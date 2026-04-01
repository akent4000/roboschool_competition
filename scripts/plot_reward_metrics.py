from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
from html import escape
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False


DEFAULT_RUNS_ROOT = Path(__file__).resolve().parents[1] / "runs" / "gait-conditioned-agility"
DEFAULT_EXPORTS_ROOT = Path(__file__).resolve().parents[1] / "plots" / "gait-conditioned-agility"
REWARD_PREFIX = "train/episode/rew "


def _parse_float(value: str) -> float | None:
    value = value.strip()
    try:
        return float(value)
    except ValueError:
        match = re.search(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", value)
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                return None
    return None


def _extract_table_cells(raw_line: str) -> Tuple[str, float] | None:
    if "│" not in raw_line:
        return None

    parts = [part.strip().strip("║│") for part in raw_line.split("│")]
    if len(parts) < 3:
        return None

    key = parts[1].strip()
    if not key:
        return None

    value = _parse_float(parts[2])
    if value is None:
        return None

    return key, value


def list_train_runs(runs_root: Path) -> List[Path]:
    if not runs_root.exists():
        return []

    runs = []
    for date_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        train_dir = date_dir / "train"
        if not train_dir.is_dir():
            continue
        for run_dir in sorted(path for path in train_dir.iterdir() if path.is_dir()):
            if (run_dir / "outputs.log").exists():
                runs.append(run_dir)
    return runs


def parse_outputs_log(path: Path) -> List[Tuple[int, Dict[str, float]]]:
    samples: List[Tuple[int, Dict[str, float]]] = []
    current_metrics: Dict[str, float] = {}

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            cell = _extract_table_cells(raw_line)
            if cell is None:
                continue

            key, value = cell
            current_metrics[key] = value

            if key == "iterations":
                samples.append((int(value), dict(current_metrics)))
                current_metrics = {}

    return samples


def build_reward_series(samples: Sequence[Tuple[int, Dict[str, float]]]) -> Dict[str, List[Tuple[int, float]]]:
    reward_keys = sorted(
        {
            key
            for _, metrics in samples
            for key in metrics
            if key.startswith(REWARD_PREFIX) and key.endswith("/mean")
        }
    )

    series: Dict[str, List[Tuple[int, float]]] = {}
    for key in reward_keys:
        values = []
        for iteration, metrics in samples:
            if key in metrics:
                values.append((iteration, metrics[key]))
        if values:
            series[key] = values
    return series


def _metric_sort_key(name: str) -> Tuple[int, str]:
    return (0 if name == f"{REWARD_PREFIX}total/mean" else 1, name.lower())


def _metric_title(name: str) -> str:
    short_name = name[len(REWARD_PREFIX):] if name.startswith(REWARD_PREFIX) else name
    short_name = short_name[:-5] if short_name.endswith("/mean") else short_name
    return short_name


def export_csv(series: Dict[str, List[Tuple[int, float]]], output: Path) -> None:
    iterations = sorted({iteration for values in series.values() for iteration, _ in values})
    value_maps = {name: dict(values) for name, values in series.items()}
    fieldnames = ["iteration"] + list(sorted(series, key=_metric_sort_key))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for iteration in iterations:
            row = {"iteration": iteration}
            for name in fieldnames[1:]:
                value = value_maps[name].get(iteration)
                row[name] = "" if value is None else f"{value:.9g}"
            writer.writerow(row)


def _plot_rewards_matplotlib(series: Dict[str, List[Tuple[int, float]]], output: Path) -> None:
    if not series:
        raise RuntimeError("No reward metrics found in outputs.log")

    ordered_items = [(name, series[name]) for name in sorted(series, key=_metric_sort_key)]
    cols = 2 if len(ordered_items) > 1 else 1
    rows = math.ceil(len(ordered_items) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7.5 * cols, 3.8 * rows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, (name, values) in zip(axes_flat, ordered_items):
        xs = [step for step, _ in values]
        ys = [value for _, value in values]
        ax.plot(xs, ys, linewidth=1.7)
        ax.set_title(_metric_title(name))
        ax.set_xlabel("iteration")
        ax.grid(True, alpha=0.3)

    for ax in axes_flat[len(ordered_items):]:
        ax.axis("off")

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _polyline_points(xs: Sequence[float], ys: Sequence[float], width: float, height: float, pad: float) -> str:
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    if x_max == x_min:
        x_max = x_min + 1.0
    if y_max == y_min:
        y_max = y_min + 1.0

    points = []
    for x, y in zip(xs, ys):
        px = pad + (x - x_min) / (x_max - x_min) * (width - 2 * pad)
        py = height - pad - (y - y_min) / (y_max - y_min) * (height - 2 * pad)
        points.append(f"{px:.2f},{py:.2f}")
    return " ".join(points)


def _plot_rewards_svg(series: Dict[str, List[Tuple[int, float]]], output: Path) -> None:
    ordered_items = [(name, series[name]) for name in sorted(series, key=_metric_sort_key)]
    cols = 2 if len(ordered_items) > 1 else 1
    rows = math.ceil(len(ordered_items) / cols)
    panel_w = 560
    panel_h = 280
    gap = 24
    width = cols * panel_w + (cols + 1) * gap
    height = rows * panel_h + (rows + 1) * gap
    pad = 42

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>',
        '.title { font: 18px sans-serif; fill: #111827; }',
        '.label { font: 12px sans-serif; fill: #4b5563; }',
        '.tick { font: 11px sans-serif; fill: #6b7280; }',
        '</style>',
    ]

    for index, (name, values) in enumerate(ordered_items):
        row = index // cols
        col = index % cols
        x0 = gap + col * (panel_w + gap)
        y0 = gap + row * (panel_h + gap)
        xs = [float(step) for step, _ in values]
        ys = [float(value) for _, value in values]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        if y_max == y_min:
            y_max = y_min + 1.0

        points = _polyline_points(xs, ys, panel_w, panel_h, pad)
        parts.extend(
            [
                f'<g transform="translate({x0},{y0})">',
                f'<rect x="0" y="0" width="{panel_w}" height="{panel_h}" fill="white" stroke="#d1d5db"/>',
                f'<line x1="{pad}" y1="{panel_h - pad}" x2="{panel_w - pad}" y2="{panel_h - pad}" stroke="#9ca3af"/>',
                f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{panel_h - pad}" stroke="#9ca3af"/>',
                f'<polyline fill="none" stroke="#2563eb" stroke-width="2" points="{points}"/>',
                f'<text class="title" x="{pad}" y="26">{escape(_metric_title(name))}</text>',
                f'<text class="label" x="{panel_w / 2:.1f}" y="{panel_h - 10}" text-anchor="middle">iteration</text>',
                f'<text class="tick" x="{pad}" y="{panel_h - pad + 18}">{int(x_min)}</text>',
                f'<text class="tick" x="{panel_w - pad}" y="{panel_h - pad + 18}" text-anchor="end">{int(x_max)}</text>',
                f'<text class="tick" x="{pad - 8}" y="{pad + 4}" text-anchor="end">{y_max:.3g}</text>',
                f'<text class="tick" x="{pad - 8}" y="{panel_h - pad + 4}" text-anchor="end">{y_min:.3g}</text>',
                '</g>',
            ]
        )

    parts.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts), encoding="utf-8")


def plot_rewards(series: Dict[str, List[Tuple[int, float]]], output: Path) -> Path:
    if HAS_MATPLOTLIB:
        _plot_rewards_matplotlib(series, output)
        return output

    svg_output = output if output.suffix.lower() == ".svg" else output.with_suffix(".svg")
    _plot_rewards_svg(series, svg_output)
    return svg_output


def _latest_iteration(run_dir: Path) -> int | None:
    log_path = run_dir / "outputs.log"
    if not log_path.exists():
        return None

    latest = None
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            cell = _extract_table_cells(raw_line)
            if cell is None:
                continue
            key, value = cell
            if key == "iterations":
                latest = int(value)
    return latest


def _print_run_menu(runs: Sequence[Path]) -> None:
    print("Доступные train-запуски:")
    for index, run_dir in enumerate(runs, start=1):
        latest_iter = _latest_iteration(run_dir)
        suffix = "" if latest_iter is None else f"  iter={latest_iter}"
        print(f"{index:>2}. {run_dir.parent.parent.name}/{run_dir.name}{suffix}")


def pick_run(runs: Sequence[Path], selected: str | None) -> Path:
    if not runs:
        raise FileNotFoundError(f"Не найдены train-раны в {DEFAULT_RUNS_ROOT}")

    if selected:
        if selected.isdigit():
            index = int(selected)
            if 1 <= index <= len(runs):
                return runs[index - 1]
            raise ValueError(f"Номер запуска вне диапазона: {index}")

        selected_path = Path(selected).expanduser()
        if selected_path.exists():
            return selected_path.resolve()

        matches = [run for run in runs if run.name == selected or selected in str(run)]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise FileNotFoundError(f"Не удалось найти запуск по значению: {selected}")
        raise ValueError(f"Найдено несколько запусков для '{selected}', укажи номер из списка")

    _print_run_menu(runs)
    answer = input("\nВыбери номер train-запуска: ").strip()
    if not answer:
        raise ValueError("Пустой выбор запуска")
    return pick_run(runs, answer)


def _default_export_path(run_dir: Path, runs_root: Path, filename: str) -> Tuple[Path, bool]:
    if os.access(run_dir, os.W_OK):
        return run_dir / filename, False

    try:
        relative_run = run_dir.relative_to(runs_root)
    except ValueError:
        relative_run = Path(run_dir.name)
    return DEFAULT_EXPORTS_ROOT / relative_run / filename, True


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse reward metrics from outputs.log and build iteration plots."
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=DEFAULT_RUNS_ROOT,
        help="Root directory with dated training runs.",
    )
    parser.add_argument(
        "--run",
        help="Run number from the printed list, run directory name, or full path. If omitted, an interactive menu is shown.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output PNG path. Defaults to <run_dir>/reward_metrics.png.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Optional CSV export path. Defaults to <run_dir>/reward_metrics.csv.",
    )
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    runs_root = args.runs_root.resolve()
    runs = list_train_runs(runs_root)
    run_dir = pick_run(runs, args.run)

    log_path = run_dir / "outputs.log"
    if not log_path.exists():
        raise FileNotFoundError(f"Не найден лог: {log_path}")

    samples = parse_outputs_log(log_path)
    if not samples:
        raise RuntimeError(f"Не удалось распарсить итерации из {log_path}")

    series = build_reward_series(samples)
    if not series:
        raise RuntimeError(f"В {log_path} нет reward-метрик формата '{REWARD_PREFIX}.../mean'")

    default_plot_name = "reward_metrics.png" if HAS_MATPLOTLIB else "reward_metrics.svg"
    if args.output:
        plot_path = args.output.resolve()
        plot_fallback = False
    else:
        plot_path, plot_fallback = _default_export_path(run_dir, runs_root, default_plot_name)

    if args.csv:
        csv_path = args.csv.resolve()
        csv_fallback = False
    else:
        csv_path, csv_fallback = _default_export_path(run_dir, runs_root, "reward_metrics.csv")

    export_csv(series, csv_path)
    saved_plot_path = plot_rewards(series, plot_path)

    print(f"Run: {run_dir}")
    print(f"Plot: {saved_plot_path}")
    print(f"CSV: {csv_path}")
    print(f"Iterations parsed: {len(samples)}")
    print(f"Reward metrics plotted: {len(series)}")
    if not HAS_MATPLOTLIB:
        print("matplotlib не найден, поэтому график сохранен в SVG.")
    if plot_fallback or csv_fallback:
        print("Run-директория недоступна на запись с хоста, поэтому файлы сохранены в plots/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
