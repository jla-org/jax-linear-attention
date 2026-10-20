# Security policy

## Supported versions

Security fixes go into the latest release of jax-linear-attention. Upgrade to it before reporting a
problem you found in an older one.

| Version | Supported |
|---|---|
| 0.1.x | Yes |

## Reporting a vulnerability

Please do not report security problems in public issues, discussions or pull requests.

Report them privately through GitHub: open the repository's
[Security tab](https://github.com/jla-org/jax-linear-attention/security) and choose
**Report a vulnerability**
([direct link](https://github.com/jla-org/jax-linear-attention/security/advisories/new)).
Only the maintainers can read the report.

Include what you can of:

- the version (`jla --version`), your platform (GPU, driver, CUDA and Python versions) and how you
  installed it;
- what an attacker can do, and what they need first (for example a crafted checkpoint, config or
  dataset file);
- steps or a small input that reproduces it.

We will confirm that we received the report, keep you informed while we fix it, and credit you in the
published advisory unless you ask us not to.

## Scope

The `jla` package and command-line tool: the kernels and ops, the training stack (configs, data
loading, checkpoints, Hugging Face import and export), the benchmark scripts and the release archives.
A problem in a third-party dependency (JAX, CUDA or the Hugging Face libraries, for example) is in
scope when jax-linear-attention exposes it; please also report it to the dependency's maintainers.
