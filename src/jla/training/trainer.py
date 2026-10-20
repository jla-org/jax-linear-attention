# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The training loop.

``Trainer(config).run()`` performs, in order:

1. mesh construction from ``config.mesh`` and per-host batch geometry;
2. model creation directly into its logical-axis shardings (no replicated
   copy is ever materialized), parameter-dtype and replication checks;
3. optimizer construction with the LR schedule and sharded optimizer state;
4. strict resume from the latest committed checkpoint (``train.resume``);
5. the data pipeline (forked-worker loader resumed by batch count, per-host
   rows, device prefetch with ``grad_accum`` grouping);
6. compile preflight (AOT lowering + compilation whose executable is reused),
   sharding audit/dump, run manifest;
7. the loop: one donated step per optimizer update, metrics fetched once per
   logging window, MFU/HFU/MBU and throughput in every log line, an optional
   profiler window with graph-health parsing, periodic jitted evaluation and
   checkpoints, and a deadline/signal/telemetry stop coordinated across hosts;
8. final evaluation and checkpoint.
"""

from __future__ import annotations

import contextlib
import dataclasses
import gc
import logging
import signal
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from jla.data.pipeline import (
    DataState,
    DevicePrefetcher,
    data_fingerprint,
    eval_batches,
    eval_dataset,
    train_dataset,
    train_loader,
)
from jla.data.streaming import stream_positions
from jla.distributed.init import process_any, process_any_host
from jla.distributed.mesh import build_mesh, plan_from_config
from jla.distributed.sharding import (
    assert_params_sufficiently_sharded,
    grouped_batch_sharding,
    make_global_batch,
    opt_state_shardings,
    param_shardings,
    path_names,
    process_row_range,
    replicated,
    replicated_fraction,
    resolve_rules,
)
from jla.distributed.kernel_sharding import use_rules as kernel_sharding_rules
from jla.training.checkpointing import CheckpointManager, wait_for_pending_checkpoints
from jla.training.graph_health import analyze_graph_health_path, graph_health_warnings, metrics_as_dict
from jla.training.loggers import metric_loggers
from jla.training.metrics_sink import MetricSink, atomic_write_json, write_run_manifest
from jla.training.mixed_precision import MixedPrecisionPolicy, check_param_dtypes, is_fp8_capable
from jla.training.optimizers import optimizer_from_config
from jla.training.performance import (
    device_profile_for,
    estimate_flops_per_token,
    estimate_hfu_flops_per_token,
    estimate_step_bytes,
    step_performance,
)
from jla.training.runtime import audit_parameter_sharding, compile_preflight, write_sharding_dump
from jla.training.schedulers import create_schedule, create_schedule_fn
from jla.training.step import LossConfig, build_eval_step, build_train_step, evaluate, lm_loss
from jla.training.telemetry import TelemetryCollector, TelemetryConfig

__all__ = ["TrainResult", "Trainer", "train"]

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    """What a run ended with (host-side history plus the final device state)."""

    step: int
    model: Any
    opt_state: Any
    rng_key: jax.Array
    data_state: DataState
    stop_reason: str
    history: list[dict[str, Any]] = field(default_factory=list)
    eval_history: list[dict[str, Any]] = field(default_factory=list)


def deterministic_config(config: Any) -> Any:
    """Route attention through XLA when ``runtime.deterministic`` is set.

    XLA's deterministic mode (``--xla_gpu_exclude_nondeterministic_ops``) excludes cuDNN's fused
    attention, and with it selected the step fails at run time (JAX 0.11.2). With XLA attention,
    a resumed run on GPU matches the uninterrupted run bit for bit.
    """
    runtime, model = getattr(config, "runtime", None), getattr(config, "model", None)
    if runtime is None or not runtime.deterministic or model is None:
        return config
    # Both attention configs: ``local_attention`` (mixed local/global stacks) would otherwise stay on cuDNN.
    changes = {
        name: dataclasses.replace(cfg, implementation="xla")
        for name in ("attention", "local_attention")
        if (cfg := getattr(model, name, None)) is not None and cfg.implementation != "xla"
    }
    if not changes:
        return config
    logger.info("runtime.deterministic: attention runs on XLA instead of cuDNN")
    return dataclasses.replace(config, model=model.replace(**changes))


def _default_model_builder() -> Callable[..., Any]:
    from jla.models import build_model

    return build_model


def _default_logical_axes() -> Callable[[Any], Any]:
    from jla.models import logical_axes

    return logical_axes


def _is_abstract_array(leaf: Any) -> bool:
    return isinstance(leaf, jax.ShapeDtypeStruct)


def _is_abstract_param(leaf: Any) -> bool:
    return isinstance(leaf, jax.ShapeDtypeStruct) and jnp.issubdtype(leaf.dtype, jnp.inexact)


def _tree_bytes(tree: Any) -> int:
    return sum(int(np.prod(leaf.shape)) * jnp.dtype(leaf.dtype).itemsize for leaf in jax.tree.leaves(tree) if hasattr(leaf, "shape"))


def _replicated_key(key: jax.Array, mesh: Any) -> jax.Array:
    """A typed PRNG key that every process holds identically, replicated over ``mesh``.

    ``jax.device_put`` cannot place a process-local array on another process's devices (it raises
    under multi-process ``jax.distributed``), so each process contributes its own copy of the data.
    """
    data = np.asarray(jax.random.key_data(key))
    sharding = replicated(mesh)
    global_data = jax.make_array_from_callback(data.shape, sharding, lambda index: data[index])
    impl = str(jax.random.key_impl(key))
    return jax.jit(lambda bits: jax.random.wrap_key_data(bits, impl=impl), out_shardings=sharding)(global_data)


class _StopSignal:
    """SIGTERM/SIGINT set a flag; a second SIGINT falls back to the default handler."""

    def __init__(self) -> None:
        self.event = threading.Event()
        self._previous: dict[int, Any] = {}

    def __enter__(self) -> _StopSignal:
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGTERM, signal.SIGINT):
                self._previous[signum] = signal.signal(signum, self._handle)
        return self

    def _handle(self, signum: int, _frame: Any) -> None:
        logger.warning("received signal %d: stopping at the next coordination boundary", signum)
        self.event.set()
        signal.signal(signum, self._previous.get(signum, signal.SIG_DFL))

    def __exit__(self, *exc: Any) -> None:
        for signum, handler in self._previous.items():
            signal.signal(signum, handler)


# Objects in the permanent generation when this module was imported (CPython 3.12 starts with a few hundred).
_FROZEN_AT_IMPORT = gc.get_freeze_count()


@contextlib.contextmanager
def _frozen_gc(enabled: bool) -> Any:
    """``train.gc_freeze``: one full collection, then the survivors move to the permanent generation.

    Tracing and compiling the train step leave hundreds of thousands of Python objects behind. Without the freeze
    the first full collection after compiling rescans them on the host while the GPU waits at the next sync, at a
    step that varies from run to run; with it that collection runs here, before the first step, and later full
    collections skip them. The steady-state step time is the same either way. ``gc.unfreeze()`` at the end returns
    every frozen object to the oldest generation, also the few the interpreter froze at startup. If more objects
    are frozen than at import (a caller that ran ``gc.freeze()`` before forking workers, say), only the collection
    runs: CPython cannot unfreeze just the objects frozen here, so a caller's freeze is left alone.

    It needs the step compiled before the loop (``train.compile_preflight``, the default): with the preflight off
    the step compiles in the loop's first step, after the freeze. The eval step always compiles at the first
    evaluation, after the freeze.
    """
    if not enabled:
        yield
        return
    gc.collect()
    if gc.get_freeze_count() > _FROZEN_AT_IMPORT:
        yield
        return
    gc.freeze()
    try:
        yield
    finally:
        gc.unfreeze()


class Trainer:
    """Runs one :class:`jla.config.ExperimentConfig`.

    ``model_builder(model_config, key=...)`` and ``logical_axes_fn(model)``
    default to :func:`jla.models.build_model` and
    :func:`jla.models.logical_axes`; ``cross_entropy`` defaults to
    :func:`jla.ops.cross_entropy.linear_cross_entropy`. ``devices``
    restricts the mesh to a subset of ``jax.devices()``.
    """

    def __init__(
        self,
        config: Any,
        *,
        model_builder: Callable[..., Any] | None = None,
        logical_axes_fn: Callable[[Any], Any] | None = None,
        cross_entropy: Callable[..., Any] | None = None,
        devices: list[Any] | None = None,
    ) -> None:
        self.config = deterministic_config(config)
        self.model_builder = model_builder or _default_model_builder()
        self.logical_axes_fn = logical_axes_fn or _default_logical_axes()
        self.cross_entropy = cross_entropy
        self.devices = list(jax.devices() if devices is None else devices)
        self.primary = jax.process_index() == 0
        self.output_dir = Path(config.train.output_dir)

    # -- setup -----------------------------------------------------------------

    def _log(self, message: str, *args: Any) -> None:
        if self.primary:
            logger.info(message, *args)

    def _init_model(self, mesh: Any, rules: dict[str, tuple[str, ...]], key: jax.Array) -> Any:
        model_config = self.config.model
        abstract = eqx.filter_eval_shape(self.model_builder, model_config, key=key)
        abstract_params = eqx.filter(abstract, _is_abstract_param)
        shardings = param_shardings(abstract_params, self.logical_axes_fn(abstract), mesh, rules)
        by_path = {
            path_names(path): sharding
            for path, sharding in jax.tree_util.tree_flatten_with_path(
                shardings, is_leaf=lambda x: isinstance(x, jax.sharding.Sharding)
            )[0]
        }
        array_shardings = jax.tree_util.tree_map_with_path(
            lambda path, leaf: by_path.get(path_names(path), replicated(mesh)),
            eqx.filter(abstract, _is_abstract_array),
        )

        def init_arrays(init_key: jax.Array) -> Any:
            return eqx.filter(self.model_builder(model_config, key=init_key), eqx.is_array)

        arrays = jax.jit(init_arrays, out_shardings=array_shardings)(key)
        return eqx.combine(arrays, eqx.filter(abstract, _is_abstract_array, inverse=True))

    def _check_params(self, params: Any, mesh: Any) -> dict[str, Any]:
        train = self.config.train
        check_param_dtypes(params, self.config.model.param_dtype)
        # dp replicates parameters by design; only the parameter-sharding axes count.
        target_axes = tuple(axis for axis in ("fsdp", "tp", "pp") if int(mesh.shape.get(axis, 1)) > 1)
        if not target_axes:
            return {"replicated_fraction": 0.0, "target_axes": []}
        fraction, leaves = replicated_fraction(params, mesh, target_axes=target_axes)
        if train.fail_on_replicated_params:
            assert_params_sufficiently_sharded(
                params, mesh, tolerance=train.replicated_params_tolerance, target_axes=target_axes
            )
        if fraction > train.replicated_params_tolerance:
            logger.warning(
                "%.1f%% of parameter elements are replicated over %s; largest: %s",
                100 * fraction,
                target_axes,
                [leaf["path"] for leaf in leaves[:3]],
            )
        return {"replicated_fraction": fraction, "target_axes": list(target_axes)}

    # -- main loop -------------------------------------------------------------

    def run(self) -> TrainResult:
        cfg = self.config
        train = cfg.train
        started = time.time()
        if self.primary:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        plan = plan_from_config(cfg.mesh, len(self.devices))
        mesh = build_mesh(plan, self.devices, topology_aware=cfg.mesh.topology_aware)
        rules = resolve_rules(cfg.mesh.rules)
        # Context parallelism used to fail at trace time for want of a context mesh; under the mesh the
        # steps now trace in, it would shard T over a dp/fsdp/tp axis with the batch gathered (untested).
        for section in ("linear_attention", "attention"):
            axis = getattr(getattr(cfg.model, section, None), "context_parallel_axis", None)
            if axis is not None:
                raise NotImplementedError(
                    f"{section}.context_parallel_axis={axis!r}: the Trainer's mesh has no context-parallel "
                    "axis (dp, fsdp and tp split the batch and the heads)"
                )
        local = getattr(cfg.model, "local_attention", None)  # a stack of local_attention layers only
        if getattr(local, "context_parallel_axis", None) is not None:
            raise NotImplementedError(
                f"local_attention.context_parallel_axis={local.context_parallel_axis!r}: the Trainer's mesh has no "
                "context-parallel axis (dp, fsdp and tp split the batch and the heads)"
            )
        global_batch = train.micro_batch_per_replica * plan.data_parallel_size
        batch_shape = (train.grad_accum, global_batch, train.seq_len)
        tokens_per_step = int(np.prod(batch_shape))
        batch_sharding = grouped_batch_sharding(mesh, rules)
        eval_sharding = grouped_batch_sharding(mesh, rules, grouped=False)
        rows = process_row_range(batch_sharding, batch_shape, axis=1)
        self._log("mesh %s on %d devices; global batch %d x %d tokens x %d microbatches", plan.axis_sizes, plan.device_count, global_batch, train.seq_len, train.grad_accum)
        pipeline_forward = None
        if plan.pp > 1:
            from jla.distributed.pipeline import (
                bubble_fraction,
                check_pipeline_attention,
                check_pipeline_layers,
                make_pipeline_forward,
                pipeline_rules,
                resolve_microbatches,
                runs_fully_manual,
                shards_head_over_stages,
                stage_layout,
            )
            from jla.distributed.sharding import batch_axes

            stage_layout(cfg.model, plan.pp)
            check_pipeline_attention(cfg.model, mesh)
            check_pipeline_layers(cfg.model, plan.pp)
            pp_micro = resolve_microbatches(plan.pp, cfg.mesh.pp_microbatches)
            if global_batch % pp_micro:
                raise ValueError(f"global batch {global_batch} is not divisible by mesh.pp_microbatches={pp_micro}")
            rules = pipeline_rules(rules)
            row_axes = batch_axes(mesh, rules)
            pipeline_forward = make_pipeline_forward(mesh, num_microbatches=pp_micro, batch_axes=row_axes, rules=rules)
            head = ("sharded over pp" if shards_head_over_stages(global_batch, mesh, num_microbatches=pp_micro, batch_axes=row_axes,
                                                                 config=cfg.model)
                    else "replicated on every stage (needs micro_batch_per_replica a multiple of pp_microbatches * pp and, "
                         "with MoE, the batch split over one mesh axis)")
            manual = runs_fully_manual(cfg.model, global_batch, mesh, num_microbatches=pp_micro, batch_axes=row_axes, rules=rules)
            self._log("pipeline: %d stages x %d microbatches (GPipe), bubble fraction %.3f; stages manual over %s; LM head and loss %s",
                      plan.pp, pp_micro, bubble_fraction(plan.pp, pp_micro), "every mesh axis" if manual else "pp only", head)

        init_key, train_key = jax.random.split(jax.random.key(train.seed))
        model = self._init_model(mesh, rules, init_key)
        params = eqx.filter(model, eqx.is_inexact_array)
        sharding_report = self._check_params(params, mesh)
        param_count = sum(int(np.prod(leaf.shape)) for leaf in jax.tree.leaves(params))
        self._log("model: %.2fM parameters (%s)", param_count / 1e6, MixedPrecisionPolicy.from_model_config(cfg.model).as_dict())

        schedule_args = {
            "warmup_steps": cfg.optim.warmup_steps,
            "min_lr_ratio": cfg.optim.min_lr_ratio,
            "stable_fraction": cfg.optim.stable_fraction,
            "decay_type": cfg.optim.decay_type,
            "d_model": cfg.model.d_model,
        }
        schedule = create_schedule(cfg.optim.lr, cfg.optim.schedule, total_steps=cfg.schedule_steps, **schedule_args)
        host_lr = create_schedule_fn(cfg.optim.schedule, max_steps=cfg.schedule_steps, **schedule_args)
        optimizer = optimizer_from_config(cfg.optim, schedule)
        param_sharding_tree = jax.tree.map(lambda leaf: leaf.sharding, params)
        opt_shardings = opt_state_shardings(jax.eval_shape(optimizer.init, params), params, param_sharding_tree, mesh)
        opt_state = jax.jit(optimizer.init, out_shardings=opt_shardings)(params)
        rng = _replicated_key(train_key, mesh)

        manager = CheckpointManager(
            self.output_dir / "checkpoints",
            keep=train.keep_checkpoints,
            asynchronous=train.async_checkpoint,
            payload_format=train.checkpoint_format,
        )
        fingerprint = data_fingerprint(cfg)
        start_step = batches_consumed = tokens_seen = 0
        stream_resume = None
        latest = manager.latest_step()
        if train.resume == "never" and latest is not None:
            raise FileExistsError(f"{manager.directory} already holds step {latest}; use another output_dir or resume")
        if train.resume == "required" and latest is None:
            raise FileNotFoundError(f"train.resume = 'required' but {manager.directory} has no committed checkpoint")
        if latest is not None:
            restored = manager.restore(latest, model=model, opt_state=opt_state, strict=True)
            state = DataState.from_dict(restored.data_state)
            state.require_compatible(global_batch_size=global_batch, seq_len=train.seq_len, fingerprint=fingerprint)
            model, opt_state = restored.model, restored.opt_state
            params = eqx.filter(model, eqx.is_inexact_array)  # release the freshly initialized arrays
            rng = _replicated_key(restored.rng_key, mesh)
            start_step, batches_consumed, tokens_seen = state.step, state.batches_consumed, state.tokens_seen
            stream_resume = state.stream
            self._log("resumed from step %d (%d batches consumed)", start_step, batches_consumed)
        if start_step >= train.max_steps:
            raise ValueError(f"checkpoint step {start_step} already reaches train.max_steps={train.max_steps}")

        def data_state(step: int) -> DataState:
            return DataState(
                step=step,
                batches_consumed=batches_consumed + (step - start_step) * train.grad_accum,
                tokens_seen=tokens_seen + (step - start_step) * tokens_per_step,
                global_batch_size=global_batch,
                seq_len=train.seq_len,
                grad_accum=train.grad_accum,
                fingerprint=fingerprint,
                stream=stream_positions(prefetch.loader, batches_consumed + (step - start_step) * train.grad_accum),
            )

        loss_config = LossConfig.from_train_config(train)
        grad_reduce = getattr(train, "grad_reduce", "micro_step")
        if grad_reduce == "step" and plan.device_count > 1:
            from jla.training.grad_reduce import check_grad_reduce

            check_grad_reduce(cfg.model, platform=str(self.devices[0].platform))
        step_fn = build_train_step(
            optimizer,
            lm_loss(loss_config, cross_entropy=self.cross_entropy, forward=pipeline_forward),
            batch_sharding=batch_sharding,
            param_shardings=param_sharding_tree,
            opt_shardings=opt_shardings,
            scan_unroll=train.scan_unroll,
            grad_reduce=grad_reduce,
            update_guard=getattr(train, "update_guard", "cond"),
        )
        eval_fn = build_eval_step(loss_config, cross_entropy=self.cross_entropy, batch_sharding=eval_sharding, forward=pipeline_forward)
        needs_eval = train.eval_every > 0 or train.final_eval
        eval_source = eval_dataset(cfg) if needs_eval else None

        prefetch = DevicePrefetcher(
            train_loader(
                cfg, train_dataset(cfg), global_batch_size=global_batch, rows=rows, skip_batches=batches_consumed, stream_state=stream_resume
            ),
            grad_accum=train.grad_accum,
            sharding=batch_sharding,
            global_shape=batch_shape,
            depth=cfg.data.device_prefetch,
        )
        telemetry = TelemetryCollector(TelemetryConfig(max_consecutive_skips=train.max_consecutive_skips))
        sink = metric_loggers(cfg, self.output_dir, run_id=f"{cfg.name}-{cfg.config_hash()[:12]}") if self.primary else None
        profile = device_profile_for(str(self.devices[0].device_kind))
        flops = estimate_flops_per_token(cfg.model, train.seq_len)
        hfu_flops = estimate_hfu_flops_per_token(flops, cfg.model.remat,
                                                 cfg.model.remat_names + cfg.model.remat_offload_names)
        bytes_per_step = estimate_step_bytes(
            config=cfg.model,
            total_params=param_count,
            optimizer_state_bytes=_tree_bytes(opt_state),
            tokens_per_microbatch=global_batch * train.seq_len,
            grad_accum=train.grad_accum,
        )
        bytes_source = "hand_model"

        # Steps trace under the mesh so kernels XLA cannot partition (Mosaic GPU custom calls) run
        # per shard instead of on gathered inputs (jla.distributed.kernel_sharding).
        # jax.set_mesh takes effect when called, so it is entered inside the try (closed in finally).
        contexts = contextlib.ExitStack()
        try:
            contexts.enter_context(jax.set_mesh(mesh if plan.device_count > 1 else None))
            contexts.enter_context(kernel_sharding_rules(rules))
            batch = next(prefetch)
            if train.compile_preflight:
                with jax.set_mesh(mesh):  # the context mesh of shard_map layers (expert parallelism)
                    report, compiled = compile_preflight("train_step", step_fn, model, opt_state, batch, rng)
                if self.primary:
                    atomic_write_json(self.output_dir / "compile_preflight.json", report)
                if compiled is None:
                    raise RuntimeError(f"train step preflight {report['status']}: {report.get('error', report.get('reason'))}")
                step_fn.use_compiled(compiled)
                compiled_bytes = report.get("cost_analysis_bytes_accessed")
                if compiled_bytes:
                    # The cost analysis describes one device's SPMD program.
                    bytes_per_step, bytes_source = compiled_bytes * plan.device_count, "compiled_cost_analysis"
                self._log(
                    "compiled train step in %.1fs (device memory %.2f GiB)",
                    report["compile_seconds"],
                    report["memory"].get("total_size_gib", float("nan")),
                )
            if train.sharding_audit and self.primary:
                audit = {**audit_parameter_sharding(params), **sharding_report}
                write_sharding_dump(self.output_dir, mesh=mesh, model=params, optimizer=opt_state, batch_sharding=batch_sharding, audit=audit)
            if self.primary:
                cfg.save(self.output_dir / "config.toml")
                write_run_manifest(
                    self.output_dir / "run.json",
                    {
                        "name": cfg.name,
                        "config_hash": cfg.config_hash(),
                        "config": cfg.to_dict(),
                        "mesh": plan.axis_sizes,
                        "process_count": jax.process_count(),
                        "devices": {
                            "count": len(self.devices),
                            "platform": str(self.devices[0].platform),
                            "kind": str(self.devices[0].device_kind),
                        },
                        "fp8_capable": is_fp8_capable(str(self.devices[0].device_kind)),
                        "parameters": param_count,
                        "flops_per_token": flops.as_dict(),
                        "start_step": start_step,
                        "jax_version": jax.__version__,
                    },
                )
            with _StopSignal() as stop_signal, jax.set_mesh(mesh), _frozen_gc(train.gc_freeze):
                result = self._loop(
                    model=model,
                    opt_state=opt_state,
                    rng=rng,
                    batch=batch,
                    step_fn=step_fn,
                    eval_fn=eval_fn,
                    eval_source=eval_source,
                    prefetch=prefetch,
                    manager=manager,
                    telemetry=telemetry,
                    sink=sink,
                    data_state=data_state,
                    start_step=start_step,
                    started=started,
                    stop_signal=stop_signal.event,
                    host_lr=host_lr,
                    perf={
                        "tokens_per_step": tokens_per_step,
                        "flops": flops.training,
                        "hfu_flops": hfu_flops,
                        "bytes": bytes_per_step,
                        "bytes_source": bytes_source,
                        "profile": profile,
                    },
                    eval_geometry=(global_batch, rows, eval_sharding),
                )
        finally:
            contexts.close()
            prefetch.close()
            wait_for_pending_checkpoints()
            if sink is not None:
                sink.close()
        return result

    def _loop(
        self,
        *,
        model: Any,
        opt_state: Any,
        rng: jax.Array,
        batch: dict[str, jax.Array],
        step_fn: Any,
        eval_fn: Any,
        eval_source: Any,
        prefetch: DevicePrefetcher,
        manager: CheckpointManager,
        telemetry: TelemetryCollector,
        sink: MetricSink | None,
        data_state: Callable[[int], DataState],
        start_step: int,
        started: float,
        stop_signal: threading.Event,
        host_lr: Callable[[int], float],
        perf: dict[str, Any],
        eval_geometry: tuple[int, tuple[int, int], Any],
    ) -> TrainResult:
        cfg = self.config
        train = cfg.train
        global_batch, rows, eval_sharding = eval_geometry
        deadline = started + 60.0 * train.time_limit_minutes if train.time_limit_minutes else None
        profile_dir = self.output_dir / "profile"
        profile_end = train.profile_start_step + train.profile_steps
        history: list[dict[str, Any]] = []
        eval_history: list[dict[str, Any]] = []
        pending: list[dict[str, jax.Array]] = []
        window_start = time.perf_counter()
        flushed_at = window_start  # when the last metrics window reached the host
        last_saved = start_step
        stop_reason = "max_steps"
        step = start_step

        def run_eval() -> dict[str, float]:
            batches = (
                make_global_batch(host_batch, eval_sharding, (global_batch, train.seq_len))
                for host_batch in eval_batches(
                    cfg, eval_source, global_batch_size=global_batch, rows=rows, num_batches=train.eval_batches
                )
            )
            metrics = evaluate(eval_fn, model, batches)
            eval_history.append({"step": step, **metrics})
            self._log("eval step %d | loss %.4f | ppl %.2f | %d tokens", step, metrics["eval_loss"], metrics["eval_perplexity"], metrics["eval_tokens"])
            if sink is not None:
                sink.append("eval", step=step, metrics=metrics)
            return metrics

        def save() -> None:
            nonlocal last_saved
            state = data_state(step)
            manager.save(
                step,
                model=model,
                opt_state=opt_state,
                data_state=state.to_dict(),
                rng_key=rng,
                metadata={"config_hash": cfg.config_hash(), "name": cfg.name},
            )
            last_saved = step

        def flush_window() -> None:
            nonlocal window_start, flushed_at
            host = jax.device_get(pending)
            flushed_at = time.perf_counter()
            elapsed = flushed_at - window_start
            first_step = step - len(host) + 1
            for offset, metrics in enumerate(host):
                telemetry_metrics = telemetry.on_step(first_step + offset, metrics)
                if telemetry_metrics and sink is not None:
                    sink.append("telemetry", step=first_step + offset, metrics=telemetry_metrics)
            window = {name: float(np.mean([float(m[name]) for m in host])) for name in host[-1]}
            window["skipped_steps"] = float(sum(not bool(m["step_applied"]) for m in host))
            window["skip_reason"] = float(int(np.bitwise_or.reduce([int(m["skip_reason"]) for m in host])))
            window["lr"] = cfg.optim.lr * host_lr(step - 1)
            speed = step_performance(
                tokens_per_step=perf["tokens_per_step"],
                step_time=max(elapsed / len(host), 1e-9),
                flops_per_token=perf["flops"],
                hfu_flops_per_token=perf["hfu_flops"],
                num_devices=len(self.devices),
                bytes_per_step=perf["bytes"],
                bytes_per_step_source=perf["bytes_source"],
                profile=perf["profile"],
                compute_dtype=cfg.model.compute_dtype,
            )
            record = {"step": step, **window, **speed.as_dict()}
            history.append(record)
            if sink is not None:
                sink.append("train", step=step, metrics=record)
            self._log(
                "step %d/%d | loss %.4f | lr %.3e | grad %.3f | %s tok/s | MFU %.1f%% HFU %.1f%% MBU %.1f%% | %.0f ms/step | skipped %d",
                step,
                train.max_steps,
                window["loss"],
                window["lr"],
                window["grad_norm"],
                f"{speed.tokens_per_sec:,.0f}",
                speed.mfu_percent,
                speed.hfu_percent,
                speed.mbu_percent,
                1000.0 * speed.step_time,
                int(window["skipped_steps"]),
            )
            pending.clear()
            window_start = time.perf_counter()

        profiling = False
        ahead: tuple[Any, Any, Any, Any] | None = None  # train.dispatch_ahead: the next step, already dispatched
        while step < train.max_steps:
            if ahead is None:
                if train.profile_steps and step == train.profile_start_step:
                    profile_dir.mkdir(parents=True, exist_ok=True)
                    jax.profiler.start_trace(str(profile_dir))
                    profiling = True
                model, opt_state, metrics, rng = step_fn(model, opt_state, batch, rng)
            else:
                (model, opt_state, metrics, rng), ahead = ahead, None
            step += 1
            pending.append(metrics)
            if profiling and step == profile_end:
                jax.block_until_ready(metrics)
                jax.profiler.stop_trace()
                profiling = False
                if train.graph_health and self.primary:
                    graph = analyze_graph_health_path(profile_dir)
                    graph_metrics = telemetry.record_graph_health(metrics_as_dict(graph), graph_health_warnings(graph), halt=False)
                    if sink is not None:
                        sink.append("graph_health", step=step, metrics=graph_metrics)
            is_log = train.log_every and step % train.log_every == 0
            is_eval = train.eval_every and step % train.eval_every == 0
            is_save = train.save_every and step % train.save_every == 0
            if step == train.max_steps or is_log or is_eval or is_save:
                if (train.dispatch_ahead and step < train.max_steps and not (is_eval or is_save) and not profiling
                        and not (train.profile_steps and step == train.profile_start_step)):
                    # Dispatch the next step before reading this one's metrics, so the devices run it while the
                    # host logs this one, instead of waiting for the host to log and dispatch (the same steps in
                    # the same order: exact).
                    batch = next(prefetch)
                    ahead = step_fn(model, opt_state, batch, rng)
                flush_window()
                local_stop = stop_signal.is_set() or telemetry.should_halt() or (deadline is not None and time.time() >= deadline)
                # With a step dispatched ahead, the device collective of process_any would wait for that step; the
                # hosts agree through the distributed runtime instead, while the devices run it.
                if process_any_host(local_stop) if ahead is not None else process_any(local_stop):
                    stop_reason = (
                        f"halt: {telemetry.halt_reason}"
                        if telemetry.should_halt()
                        else "signal"
                        if stop_signal.is_set()
                        else "time_limit"
                        if deadline is not None and time.time() >= deadline
                        else "stop requested by another process"
                    )
                    if ahead is not None:  # the step dispatched ahead has run too: count it (logged below)
                        (model, opt_state, metrics, rng), ahead = ahead, None
                        step += 1
                        pending.append(metrics)
                    self._log("stopping at step %d: %s", step, stop_reason)
                    break
                if is_eval and eval_source is not None:
                    run_eval()
                if is_save:
                    save()
                # With a step dispatched ahead, its window starts when this one's metrics arrived. The devices may
                # have started it before then (metrics can reach the host up to a step late), so the windows are
                # metric arrival to metric arrival and the last one before the end can come out short.
                window_start = flushed_at if ahead is not None else time.perf_counter()
            if step < train.max_steps and ahead is None:
                batch = next(prefetch)
        if pending:  # a step dispatched ahead of a stop
            flush_window()
        if profiling:
            jax.profiler.stop_trace()
        evaluated_now = bool(eval_history) and eval_history[-1]["step"] == step
        if stop_reason == "max_steps" and train.final_eval and eval_source is not None and not evaluated_now:
            run_eval()
        if (train.save_final or stop_reason != "max_steps") and step > last_saved:
            save()
        if telemetry.should_halt() and self.primary:
            telemetry.recorder.dump(self.output_dir / "flight_recorder.json", reason=telemetry.halt_reason)
        if sink is not None:
            sink.append("end", step=step, metrics={"stop_reason": stop_reason, "seconds": time.time() - started})
        return TrainResult(
            step=step,
            model=model,
            opt_state=opt_state,
            rng_key=rng,
            data_state=data_state(step),
            stop_reason=stop_reason,
            history=history,
            eval_history=eval_history,
        )


def train(config: Any, **kwargs: Any) -> TrainResult:
    """``Trainer(config, **kwargs).run()``."""
    return Trainer(config, **kwargs).run()
