#!/usr/bin/env python3
"""
YOLO dataset viewer.

Usage:
    python scripts/view_dataset.py                          # default datasets/yolo_objects
    python scripts/view_dataset.py --dir datasets/yolo_objects
    python scripts/view_dataset.py --dir datasets/yolo_objects --class backpack
    python scripts/view_dataset.py --dir datasets/yolo_objects --only_labeled

Controls:
    →  / D       next image
    ←  / A       previous image
    F            toggle show filename
    B            toggle bounding boxes
    Q / Esc      quit
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

OBJECT_NAMES = {0: "backpack", 1: "bottle", 2: "chair", 3: "cup", 4: "laptop"}

COLORS = {
    0: (52, 152, 219),   # backpack  — blue
    1: (46, 204, 113),   # bottle    — green
    2: (231, 76, 60),    # chair     — red
    3: (241, 196, 15),   # cup       — yellow
    4: (155, 89, 182),   # laptop    — purple
}


def load_labels(lbl_path: Path) -> list[tuple[int, float, float, float, float]]:
    rows = []
    if lbl_path.exists():
        for line in lbl_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) == 5:
                c, xc, yc, bw, bh = parts
                rows.append((int(c), float(xc), float(yc), float(bw), float(bh)))
    return rows


def draw_boxes(img: np.ndarray, labels: list, alpha: float = 0.15) -> np.ndarray:
    out = img.copy()
    overlay = img.copy()
    h, w = img.shape[:2]

    for cls_id, xc, yc, bw, bh in labels:
        color = COLORS.get(cls_id, (200, 200, 200))
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label = OBJECT_NAMES.get(cls_id, str(cls_id))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ty = max(y1 - 4, th + 4)
        cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 6, ty + 2), color, -1)
        cv2.putText(out, label, (x1 + 3, ty - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return out


def draw_hud(img: np.ndarray, text_lines: list[str]) -> np.ndarray:
    out = img.copy()
    y = 22
    for line in text_lines:
        cv2.putText(out, line, (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, line, (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += 20
    return out


def build_file_list(
    img_dir: Path,
    lbl_dir: Path,
    filter_class: str | None,
    only_labeled: bool,
) -> list[Path]:
    files = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))

    if only_labeled or filter_class is not None:
        filtered = []
        cls_id_filter = None
        if filter_class is not None:
            cls_id_filter = next(
                (k for k, v in OBJECT_NAMES.items() if v == filter_class), None
            )

        for p in files:
            labels = load_labels(lbl_dir / p.with_suffix(".txt").name)
            if only_labeled and not labels:
                continue
            if cls_id_filter is not None:
                if not any(l[0] == cls_id_filter for l in labels):
                    continue
            filtered.append(p)
        return filtered

    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO dataset viewer")
    parser.add_argument("--dir", default="datasets/yolo_objects",
                        help="Root dataset dir (must contain images/train and labels/train)")
    parser.add_argument("--class", dest="filter_class", default=None,
                        choices=list(OBJECT_NAMES.values()),
                        help="Show only frames containing this class")
    parser.add_argument("--only_labeled", action="store_true",
                        help="Skip frames with no annotations")
    args = parser.parse_args()

    root = Path(args.dir)
    img_dir = root / "images" / "train"
    lbl_dir = root / "labels" / "train"

    if not img_dir.exists():
        print(f"[error] images dir not found: {img_dir}", file=sys.stderr)
        sys.exit(1)

    files = build_file_list(img_dir, lbl_dir, args.filter_class, args.only_labeled)
    if not files:
        print("[error] no images found", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(files)} images from {img_dir}")
    print("Controls: ← / → navigate   B toggle boxes   F toggle filename   Q quit")

    idx = 0
    show_boxes = True
    show_fname = True

    cv2.namedWindow("viewer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("viewer", 960, 560)

    while True:
        path = files[idx]
        img = cv2.imread(str(path))
        if img is None:
            idx = (idx + 1) % len(files)
            continue

        labels = load_labels(lbl_dir / path.with_suffix(".txt").name)

        if show_boxes and labels:
            img = draw_boxes(img, labels)

        hud = [f"{idx + 1}/{len(files)}  [{len(labels)} objects]"]
        if show_fname:
            hud.append(path.name)
        if labels:
            cls_counts: dict[str, int] = {}
            for l in labels:
                n = OBJECT_NAMES.get(l[0], str(l[0]))
                cls_counts[n] = cls_counts.get(n, 0) + 1
            hud.append("  ".join(f"{n}:{c}" for n, c in cls_counts.items()))

        img = draw_hud(img, hud)
        cv2.imshow("viewer", img)

        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):        # Q / Esc
            break
        elif key in (ord("d"), 83, 32):  # D / → / Space
            idx = (idx + 1) % len(files)
        elif key in (ord("a"), 81):      # A / ←
            idx = (idx - 1) % len(files)
        elif key == ord("b"):
            show_boxes = not show_boxes
        elif key == ord("f"):
            show_fname = not show_fname

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
