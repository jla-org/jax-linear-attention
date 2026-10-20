# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Read every curve of Figure 2 of the Gated DeltaNet-2 paper (arXiv 2605.22791) from the PDF's vector paths.

    pip install pymupdf
    curl -L -o gdn2.pdf https://arxiv.org/pdf/2605.22791
    python benchmarks/end_to_end/gdn2_paper/read_paper_fig2.py gdn2.pdf

The figure is vector graphics, so every plotted point is stored exactly. The y axis is calibrated from the figure's
own gridlines and tick labels (25 to 45 thousand tokens per second); each curve is the stroked path through the four
x positions (2K x 8, 4K x 4, 8K x 2 and 16K x 1); each curve's name comes from the legend swatch of the same color.
The output ends with a check against the two numbers the paper's text prints (Gated DeltaNet-2: 38.0 and 36.1).
"""
import json
import sys

import pymupdf

NAMES = {"Transformer", "Mamba-2", "Mamba-3 SISO", "Mamba-3 MIMO", "Gated DeltaNet", "KDA", "Gated DeltaNet-2"}
SPLITS = ["2K x 8", "4K x 4", "8K x 2", "16K x 1"]


def main(path):
    page = next(p for p in pymupdf.open(path) if "Training throughput on a H100" in p.get_text())
    spans = [(s["text"].strip(), s["bbox"]) for b in page.get_text("dict")["blocks"]
             for line in b.get("lines", []) for s in line["spans"]]
    drawings = page.get_drawings()
    # Gridlines: long horizontal light-gray strokes. Ticks: the labels 25 ... 45 left of the plot.
    grid_y = sorted({round(it[1].y, 3) for d in drawings if d.get("color") and
                     tuple(round(c, 2) for c in d["color"]) == (0.9, 0.9, 0.9)
                     for it in d["items"] if it[0] == "l" and abs(it[1].y - it[2].y) < 0.05
                     and abs(it[2].x - it[1].x) > 100})
    ticks = {float(t): (bb[1] + bb[3]) / 2 for t, bb in spans if t in {"25", "30", "35", "40", "45"} and bb[0] > 290}
    pairs = [(v, min(grid_y, key=lambda g: abs(g - yc))) for v, yc in sorted(ticks.items())]
    mv = sum(v for v, _ in pairs) / len(pairs)
    my = sum(y for _, y in pairs) / len(pairs)
    slope = sum((v - mv) * (y - my) for v, y in pairs) / sum((v - mv) ** 2 for v, _ in pairs)
    icept = my - slope * mv
    residual = max(abs(y - (icept + slope * v)) for v, y in pairs)
    # Legend: a short horizontal swatch (line, marker, line) left of each name, on the same baseline.
    legend = {}
    for d in drawings:
        r, col = d["rect"], d.get("color")
        if col and [it[0] for it in d["items"]] == ["l", "c", "l"] and r.y1 - r.y0 < 0.05 and 12 < r.x1 - r.x0 < 25:
            for t, bb in spans:
                if t in NAMES and 0 < bb[0] - r.x1 < 15 and abs((bb[1] + bb[3]) / 2 - r.y0) < 3:
                    legend[tuple(round(c, 3) for c in col)] = t
    curves = {}
    for d in drawings:
        col, items = d.get("color"), d["items"]
        if not col or len(items) != 3 or any(it[0] != "l" for it in items) or (d.get("width") or 0) < 1.0:
            continue
        pts = sorted([(it[1].x, it[1].y) for it in items] + [(items[-1][2].x, items[-1][2].y)])
        if pts[0][0] < 330 and pts[-1][0] > 500:
            name = legend.get(tuple(round(c, 3) for c in col), f"unmatched {col}")
            curves[name] = [round((y - icept) / slope, 2) for _, y in pts]
    print(json.dumps({"splits": SPLITS, "kt_per_s": curves, "calibration_residual_pt": residual}, indent=1))
    g = curves.get("Gated DeltaNet-2")
    if g:
        print(f"check: Gated DeltaNet-2 {g[0]:.2f} at 2K x 8 (text: 38.0), {g[3]:.2f} at 16K x 1 (text: 36.1)")


if __name__ == "__main__":
    main(sys.argv[1])
