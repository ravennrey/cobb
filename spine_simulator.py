"""Tkinter-based GUI to visualize spine and compute Cobb angles.

This script loads vertebra bounding boxes from JSON files and
corresponding images, renders the spine in 2D/3D using Matplotlib, and
computes Cobb angles using helper functions from ``cobb_measurement``.

The JSON directory and image directory can be specified via command line
arguments ``--json-dir`` and ``--img-dir`` or environment variables
``JSON_DIR`` and ``IMG_DIR``. Defaults are ``data/json`` and ``data/img``
relative to the script location.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import List

import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation

from cobb_measurement import CobbConfig, compute_cobb_regions, moving_avg

log = logging.getLogger(__name__)


def load_boxes(path: Path) -> List[dict]:
    """Load bounding boxes from *path*.

    Raises ``ValueError`` if required keys are missing.
    """
    with path.open("r", encoding="utf-8") as f:
        boxes = json.load(f)

    required = {"x1", "x2", "y1", "y2"}
    for idx, b in enumerate(boxes):
        if not required.issubset(b):
            missing = required - set(b)
            raise ValueError(f"Missing keys {missing} in box #{idx} of {path}")
    return boxes


class SpineSimulator:
    def __init__(self, root: tk.Tk, json_files: List[Path], img_dir: Path, cfg: CobbConfig):
        self.root = root
        self.json_files = json_files
        self.img_dir = img_dir
        self.cfg = cfg
        self.current_index = 0

        self.root.geometry("1500x900")
        self.root.title("3D Omurga Simülasyonu")

        top = tk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X)
        tk.Button(top, text="⟵ Prev", command=self.prev_file).pack(side=tk.LEFT, padx=6, pady=6)
        tk.Button(top, text="Next ⟶", command=self.next_file).pack(side=tk.LEFT, padx=6, pady=6)
        self.summary_lbl = tk.Label(top, text="", font=("Segoe UI", 11))
        self.summary_lbl.pack(side=tk.LEFT, padx=14)

        body = tk.Frame(self.root)
        body.pack(fill=tk.BOTH, expand=True)

        self.right_frame = tk.Frame(body, width=500, borderwidth=2, relief="groove", bg="#EEEEEE")
        self.right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        self.right_frame.pack_propagate(False)
        self.img_label = tk.Label(self.right_frame, bg="#EEEEEE")
        self.img_label.pack(expand=True)

        fig = plt.figure(figsize=(9.5, 6.2), facecolor="white")
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 2], wspace=0.25)
        self.ax3d = fig.add_subplot(gs[0, 0], projection="3d", facecolor="white")
        self.ax2d = fig.add_subplot(gs[0, 1])
        self.canvas = FigureCanvasTkAgg(fig, master=body)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.fig = fig
        self.ani = None
        self.tk_img = None

        if not json_files:
            log.error("No JSON files found in %s", json_dir)
            raise SystemExit(1)
        self.load_json_and_render(self.json_files[self.current_index])

    def prev_file(self) -> None:
        self.current_index = (self.current_index - 1) % len(self.json_files)
        self.load_json_and_render(self.json_files[self.current_index])

    def next_file(self) -> None:
        self.current_index = (self.current_index + 1) % len(self.json_files)
        self.load_json_and_render(self.json_files[self.current_index])

    # ------------------- DATA PROCESSING -------------------
    def _create_spine_from_2d_boxes(self, path: Path):
        boxes = load_boxes(path)

        for b in boxes:
            b["cx"] = (b["x1"] + b["x2"]) / 2.0
            b["cy"] = (b["y1"] + b["y2"]) / 2.0
            b["h"] = abs(b["y2"] - b["y1"])

        boxes.sort(key=lambda b: b["cy"])  # top to bottom

        xs = [b["cx"] for b in boxes]
        ys = [b["cy"] for b in boxes]
        hs = [b["h"] for b in boxes]

        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        minh, maxh = min(hs), max(hs)

        verts = []
        for b in boxes:
            px_x = b["cx"]
            px_z = (maxy - b["cy"])  # up small, down big

            nx = (b["cx"] - minx) / max(1e-6, (maxx - minx)) * 2 - 1
            ny = 0.0
            nz = (1 - (b["cy"] - miny) / max(1e-6, (maxy - miny))) * 35
            nh = (b["h"] - minh) / max(1e-6, (maxh - minh))

            verts.append(
                {
                    "px_x": px_x,
                    "px_z": px_z,
                    "x3d": round(nx * 5, 4),
                    "y3d": ny,
                    "z3d": round(nz, 4),
                    "size": round(40 + nh * 50, 2),
                    "color": "dimgray",
                }
            )
        return verts

    def load_json_and_render(self, path: Path) -> None:
        filename = path.name
        self.root.title(f"3D Omurga Simülasyonu - {filename}")

        image_name = path.stem + ".jpeg"
        image_path = self.img_dir / image_name
        if image_path.exists():
            pil = Image.open(image_path).resize((420, 420))
            self.tk_img = ImageTk.PhotoImage(pil, master=self.root)
            self.img_label.configure(image=self.tk_img, text="")
            self.img_label.image = self.tk_img
        else:
            log.warning("Image not found: %s", image_path)
            self.img_label.configure(image="", text="Görsel bulunamadı", font=("Segoe UI", 12))
            self.img_label.image = None

        try:
            self.spine_data = self._create_spine_from_2d_boxes(path)
        except Exception as exc:
            log.exception("Could not process %s: %s", path, exc)
            self.summary_lbl.config(text="Hata: JSON okunamadı")
            return

        self.pts = np.array([[v["x3d"], v["y3d"], v["z3d"]] for v in self.spine_data])
        self.colors = [v["color"] for v in self.spine_data]
        self.sizes = [v["size"] for v in self.spine_data]

        self._draw_all()
        self._start_anim()

    # ------------------- DRAWING -------------------
    def _draw_all(self) -> None:
        ax = self.ax3d
        ax.clear()
        ax.plot(self.pts[:, 0], self.pts[:, 1], self.pts[:, 2], "-", c="gray", alpha=0.6, linewidth=2.2)
        ax.scatter(self.pts[:, 0], self.pts[:, 1], self.pts[:, 2], c=self.colors, s=self.sizes, depthshade=True)
        ax.axis("off")

        max_range = np.array([np.ptp(self.pts[:, i]) for i in range(3)]).max() / 2.0
        mid = np.mean(self.pts, axis=0)
        scale = 1.25
        ax.set_xlim(mid[0] - max_range * scale, mid[0] + max_range * scale)
        ax.set_ylim(mid[1] - max_range * scale, mid[1] + max_range * scale)
        ax.set_zlim(mid[2] - max_range * scale, mid[2] + max_range * scale)

        self.ax2d.clear()
        spine_xy_px = np.array([[v["px_x"], v["px_z"]] for v in self.spine_data], dtype=float)

        if self.cfg.smooth_win > 1:
            spine_xy_px[:, 0] = moving_avg(spine_xy_px[:, 0], self.cfg.smooth_win)

        regions = compute_cobb_regions(spine_xy_px, self.cfg)

        plot_xy = spine_xy_px.copy()
        plot_xy[:, 0] = (plot_xy[:, 0] - plot_xy[:, 0].mean()) * 0.7 + plot_xy[:, 0].mean()

        self.ax2d.plot(plot_xy[:, 0], plot_xy[:, 1], ".-", color="tab:blue", linewidth=2)
        self.ax2d.set_title("COBB Ölçümü (°)")

        labels = []
        for r in regions:
            (t1, t2) = r["top_line"]
            (b1, b2) = r["bot_line"]

            def _sx(p):
                return np.array([(p[0] - plot_xy[:, 0].mean()) * 0.7 + plot_xy[:, 0].mean(), p[1]])

            T1, T2 = _sx(t1), _sx(t2)
            B1, B2 = _sx(b1), _sx(b2)

            self.ax2d.plot([T1[0], T2[0]], [T1[1], T2[1]], "r-", linewidth=2.3)
            self.ax2d.plot([B1[0], B2[0]], [B1[1], B2[1]], "r-", linewidth=2.3)

            cx = (T1[0] + T2[0] + B1[0] + B2[0]) / 4.0
            cz = (T1[1] + T2[1] + B1[1] + B2[1]) / 4.0
            self.ax2d.text(
                cx,
                cz,
                f"{r['cobb']:.1f}°",
                fontsize=10,
                ha="center",
                va="center",
                color="darkred",
                bbox=dict(facecolor="white", alpha=0.9, edgecolor="black"),
            )
            labels.append(f"{r['cobb']:.1f}°")

        self.ax2d.set_aspect("equal", adjustable="box")
        self.ax2d.margins(x=0.10, y=0.10)

        self.summary_lbl.config(text="Cobb: " + (", ".join(labels) if labels else "—"))
        self.canvas.draw_idle()

    def _start_anim(self) -> None:
        if self.ani:
            try:
                self.ani.event_source.stop()
            except Exception:
                pass
        self.ani = animation.FuncAnimation(
            self.fig,
            lambda f: (self.ax3d.view_init(elev=15, azim=f), self.fig)[-1],
            frames=np.arange(0, 360, 1),
            interval=40,
            blit=False,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spine simulator with Cobb angle measurement")
    parser.add_argument("--json-dir", type=Path, default=Path(os.environ.get("JSON_DIR", "data/json")))
    parser.add_argument("--img-dir", type=Path, default=Path(os.environ.get("IMG_DIR", "data/img")))
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    cfg = CobbConfig()

    json_files = sorted(args.json_dir.glob("*.json"))
    root = tk.Tk()
    SpineSimulator(root, json_files, args.img_dir, cfg)
    root.mainloop()
