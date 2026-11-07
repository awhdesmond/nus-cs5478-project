"""Report figures from the result CSVs (no simulation runs).

    uv run python scripts/plots.py                 # all figures → docs/figures/
    uv run python scripts/plots.py --only search   # one figure by name

`results/` is gitignored, so figures are written to docs/figures/ and travel with the report.
Palette: categorical slots 1-3 of the reference data-viz palette (validated for this many series); aqua and
yellow sit below 3:1 on the light surface, so every series is also direct-labelled or named on its own axis.
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from insertion.config import RESULTS_DIR, ROOT
from insertion.stats import wilson

FIGURES = ROOT / "docs" / "figures"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#d9d8d2"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
CONTROLLERS = {"straight": "open loop", "guarded": "guarded retry", "impedance": "impedance", "spiral": "spiral search"}
CONTROLLER_COLOUR = {"straight": INK_2, "guarded": AQUA, "impedance": ORANGE, "spiral": BLUE}


def style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.titlecolor": INK,
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8.5,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0,
        "figure.dpi": 200, "savefig.bbox": "tight",
    })


def save(fig, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"→ {path.relative_to(ROOT)}")
    return path


def box(ax, groups: list[np.ndarray], labels: list[str], colour, horizontal: bool = False):
    """Recessive box plot: thin whiskers, filled boxes, no fliers (the points are shown separately)."""
    colours = colour if isinstance(colour, list) else [colour] * len(groups)
    parts = ax.boxplot(groups, tick_labels=labels, orientation="horizontal" if horizontal else "vertical",
                       widths=0.55, showfliers=False,
                       patch_artist=True, medianprops=dict(color=SURFACE, linewidth=1.5),
                       whiskerprops=dict(color=INK_2, linewidth=1.0), capprops=dict(color=INK_2, linewidth=1.0))
    for patch, c in zip(parts["boxes"], colours):
        patch.set(facecolor=c, edgecolor=c, alpha=0.85, linewidth=0)
    return parts


def fig_scene() -> None:
    """The full-task scene: arm at the `ready` pose, and a close-up of the fixture with both sockets."""
    import mujoco

    from insertion.assembly_scene import build_assembly_model, sample_layout
    from insertion.assembly_sim import AssemblySim
    from insertion.config import AssemblyConfig, JointPDConfig
    from insertion.control.joint_pd import joint_pd_torque

    cfg = sample_layout(AssemblyConfig(), 4, orientation="mixed")
    model = build_assembly_model(cfg)
    sim = AssemblySim(model, cfg)
    sim.reset()
    q0 = sim.q()
    for _ in range(500):  # let the connectors settle
        sim.step(joint_pd_torque(sim, q0, JointPDConfig()))
    renderer = mujoco.Renderer(model, 720, 960)
    views = [("the scene at the `ready` pose", [-0.02, 0.38, 0.26], 0.95, 152, -20),
             ("fixture: round and square sockets", list(sim.data.site("round_socket_entrance").xpos), 0.20, 150, -28)]
    images = []
    for _, lookat, distance, azimuth, elevation in views:
        camera = mujoco.MjvCamera()
        camera.lookat[:] = lookat
        camera.distance, camera.azimuth, camera.elevation = distance, azimuth, elevation
        renderer.update_scene(sim.data, camera)
        images.append(renderer.render().copy())
    renderer.close()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for ax, image, (title, *_) in zip(axes, images, views):
        ax.imshow(image)
        ax.set_title(title, loc="left", fontsize=9)
        ax.set_axis_off()
    fig.subplots_adjust(wspace=0.04)
    save(fig, "f1_scene")


# --- weeks 1-5: the insertion module ------------------------------------------------------------------------


def fig_controller_success() -> None:
    """Success rate vs believed-pose uncertainty, one line per controller, one panel per clearance (Plan A grid)."""
    frames = []
    for name in CONTROLLERS:
        df = pd.read_csv(RESULTS_DIR / f"plan_a_{name}" / "trials.csv")
        df["controller"] = name
        frames.append(df)
    df = pd.concat(frames)
    df = df.rename(columns={"cfg.scene.clearance": "clearance", "cfg.uncertainty.translation_max": "uncertainty"})
    clearances = sorted(df.clearance.unique(), reverse=True)
    fig, axes = plt.subplots(1, len(clearances), figsize=(9.5, 3.2), sharey=True)
    for ax, clearance in zip(axes, clearances):
        for name, label in CONTROLLERS.items():
            cell = df[(df.controller == name) & (df.clearance == clearance)]
            rates = cell.groupby("uncertainty").success.agg(["mean", "sum", "size"]).reset_index()
            x = rates.uncertainty * 1e3
            lo, hi = zip(*[wilson(int(s), int(n)) for s, n in zip(rates["sum"], rates["size"])])
            mean = rates["mean"].values * 100
            yerr = np.clip([mean - np.array(lo) * 100, np.array(hi) * 100 - mean], 0, None)  # Wilson centre is shifted
            ax.errorbar(x, mean, yerr=yerr,
                        color=CONTROLLER_COLOUR[name], marker="o", markersize=5, capsize=3, elinewidth=1.0, label=label)
            if ax is axes[-1]:  # direct labels (the palette's aqua/yellow are below 3:1 on this surface)
                ax.annotate(label, (x.iloc[-1], rates["mean"].iloc[-1] * 100), xytext=(6, 0), textcoords="offset points",
                            color=CONTROLLER_COLOUR[name], fontsize=8, va="center")
        ax.set_title(f"{clearance * 1e3:.0f} mm clearance")
        ax.set_xlabel("pose uncertainty (± mm)")
        ax.set_xticks([1, 3, 5])
        ax.set_ylim(0, 105)
    axes[0].set_ylabel("insertions seated (%)")
    axes[0].legend(frameon=False, loc="lower left")
    fig.suptitle("Insertion module: success vs pose uncertainty (50 seeds per point, both friction levels; 95% Wilson intervals)",
                 fontsize=10, y=1.04)
    fig.subplots_adjust(right=0.86)
    save(fig, "f2_controller_success")


def fig_controller_cycle_time() -> None:
    """Cycle time of successful insertions per controller (Plan A grid), split by clearance."""
    frames = []
    for name in CONTROLLERS:
        df = pd.read_csv(RESULTS_DIR / f"plan_a_{name}" / "trials.csv")
        df["controller"] = name
        frames.append(df)
    df = pd.concat(frames).rename(columns={"cfg.scene.clearance": "clearance", "cfg.uncertainty.translation_max": "uncertainty",
                                           "cfg.scene.friction": "friction"})
    # Paired: only trials every controller solved, or the easy trials open loop wins on would bias its average.
    trial = ["clearance", "uncertainty", "friction", "seed"]
    solved_by_all = df.groupby(trial).success.transform("sum") == len(CONTROLLERS)
    df = df[solved_by_all & df.success]
    clearances = sorted(df.clearance.unique(), reverse=True)
    fig, axes = plt.subplots(1, len(clearances), figsize=(9.5, 3.0), sharey=True)
    for ax, clearance in zip(axes, clearances):
        cell = df[df.clearance == clearance]
        groups = [cell[cell.controller == name].cycle_time.values for name in CONTROLLERS]
        box(ax, groups, list(CONTROLLERS.values()), [CONTROLLER_COLOUR[n] for n in CONTROLLERS])
        ax.set_title(f"{clearance * 1e3:.0f} mm clearance")
        ax.tick_params(axis="x", rotation=20)
    axes[0].set_ylabel("cycle time of seated insertions (s)")
    fig.suptitle(f"Insertion module: cycle time on the {len(df) // len(CONTROLLERS)} of 900 trials every controller solved",
                 fontsize=10, y=1.02)
    save(fig, "f3_controller_cycle_time")


def fig_force_trace() -> None:
    """One insertion, same seed: impedance alone stalls on the fixture face; spiral search finds the hole.

    Force is the 10 ms moving average that the force cap scores (raw 1 kHz trace faint behind it); the lower panel
    is the peg tip's depth relative to the socket entrance, so "searching on the surface" and "seated" are visible.
    """
    import json

    traces = {"impedance": RESULTS_DIR / "demos" / "impedance_mu03_seed8.npz",
              "spiral": RESULTS_DIR / "demos" / "spiral_mu03_seed8.npz"}
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.8, 4.6), sharex=True, gridspec_kw=dict(height_ratios=[1.6, 1]))
    for name, path in traces.items():
        data = np.load(path)
        cfg = json.loads(str(data["config"]))
        entrance = cfg["scene"]["socket_pos"][2] + cfg["scene"]["socket_depth"]  # socket_pos is the hole bottom
        depth = cfg["scene"]["socket_depth"]
        time = data["time"] - data["time"][0]
        force = np.linalg.norm(data["contact_force"], axis=1)
        window = max(1, int(round(0.010 / np.median(np.diff(time)))))
        smooth = np.convolve(force, np.ones(window) / window, mode="same")
        colour = CONTROLLER_COLOUR[name]
        ax.plot(time, force, color=colour, linewidth=0.6, alpha=0.25)
        ax.plot(time, smooth, color=colour, linewidth=1.8, label=CONTROLLERS[name])
        ax2.plot(time, (data["tip_pos"][:, 2] - entrance) * 1e3, color=colour, linewidth=1.8, label=CONTROLLERS[name])
    ax.axhline(30, color=INK_2, linestyle="--", linewidth=1.0)
    ax.annotate("30 N force cap", (0.3, 30), xytext=(0, -12), textcoords="offset points", color=INK_2, fontsize=8)
    ax.set_ylabel("contact force (N)\n10 ms mean")
    ax.set_ylim(0, 33)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.02, 0.82), ncol=2)
    ax.set_title("Same seed, 1 mm clearance, µ = 0.3, ±5 mm pose uncertainty", loc="left")
    ax.annotate("spiral search sweeping on the fixture face", (12.0, 12.0), xytext=(0, 26), textcoords="offset points",
                color=BLUE, fontsize=8, ha="center", arrowprops=dict(arrowstyle="->", color=BLUE, linewidth=1.0))

    ax2.axhline(0, color=INK_2, linewidth=0.8)
    ax2.axhline(-depth * 1e3, color=INK_2, linestyle="--", linewidth=1.0)
    ax2.annotate("socket entrance", (0.3, 0), xytext=(0, 4), textcoords="offset points", color=INK_2, fontsize=8)
    ax2.annotate("hole bottom (seated)", (0.3, -depth * 1e3), xytext=(0, 4), textcoords="offset points", color=INK_2, fontsize=8)
    ax2.annotate("impedance alone never enters", (17.0, -1.0), xytext=(0, 14), textcoords="offset points",
                 color=ORANGE, fontsize=8, ha="center", arrowprops=dict(arrowstyle="->", color=ORANGE, linewidth=1.0))
    ax2.set_ylabel("peg tip depth (mm)\nbelow the entrance")
    ax2.set_xlabel("time since the insertion started (s)")
    ax2.set_ylim(-depth * 1e3 - 4, 6)
    save(fig, "f4_force_trace")


def fig_compliance_checks() -> None:
    """Impedance in isolation: the spring saturates with setpoint depth, and impact force scales with touchdown speed."""
    press = pd.read_csv(RESULTS_DIR / "compliance_check" / "surface_press.csv")
    touchdown = pd.read_csv(RESULTS_DIR / "compliance_check" / "touchdown_speed.csv").sort_values("touchdown_speed")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.2))

    impedance = press[press.controller == "impedance"]
    depth = impedance.setpoint_depth * 1e3
    ax.plot(depth, impedance.setpoint_depth * 1000, color=GRID, linewidth=1.2, linestyle=":")  # K·depth, 1 N per mm
    ax.annotate("K·depth (unsaturated)", (depth.iloc[-1], depth.iloc[-1]), xytext=(-4, -12), textcoords="offset points",
                color=INK_2, fontsize=8, ha="right")
    ax.plot(depth, impedance.mean_force, color=ORANGE, marker="o", markersize=6, label="impedance")
    pd_row = press[press.controller == "pd"]
    ax.plot(pd_row.setpoint_depth * 1e3, pd_row.mean_force, color=INK_2, marker="s", markersize=6, linestyle="none", label="joint PD")
    ax.annotate("saturated at 15 N", (depth.iloc[-1], impedance.mean_force.iloc[-1]), xytext=(-6, 8),
                textcoords="offset points", color=ORANGE, fontsize=8, ha="right")
    ax.set_xlabel("setpoint below the surface (mm)")
    ax.set_ylabel("steady contact force (N)")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Pressing on a surface", loc="left")

    speed = touchdown.touchdown_speed * 1e3
    ax2.plot(speed, touchdown.peak_force, color=BLUE, marker="o", markersize=6, label="single step (1 ms)")
    ax2.plot(speed, touchdown.peak_force_10ms, color=ORANGE, marker="o", markersize=6, label="10 ms mean (scored)")
    ax2.axhline(5.0, color=GRID, linewidth=1.2, linestyle=":")
    ax2.annotate("5 N steady press", (speed.iloc[-1], 5.0), xytext=(-4, -12), textcoords="offset points",
                 color=INK_2, fontsize=8, ha="right")
    ax2.set_xlabel("touchdown speed (mm/s)")
    ax2.set_ylabel("peak contact force (N)")
    ax2.legend(frameon=False, loc="upper left")
    ax2.set_title("Impact at touchdown", loc="left")
    save(fig, "f1b_compliance_checks")


# --- weeks 6-7: the two-connector task ----------------------------------------------------------------------


def _m5() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "m5" / "connectors.csv")


def fig_search_usage() -> None:
    """How often spiral search runs in the full task: perception error is what triggers it."""
    df = _m5()
    df["orientation_label"] = np.where(df.lying, "lying", "upright")
    cells = [(0.0, "ideal"), (0.0, "realistic"), (0.002, "ideal"), (0.002, "realistic")]
    labels = ["ideal\ncamera", "realistic\ncamera", "ideal\ncamera", "realistic\ncamera"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.4), gridspec_kw=dict(width_ratios=[1.35, 1]))
    x = np.arange(len(cells))
    for offset, (name, colour) in zip((-0.19, 0.19), (("upright", BLUE), ("lying", ORANGE))):
        shares = [df[(df.object_error == e) & (df.condition == c) & (df.orientation_label == name)]["search.searched"].mean() * 100
                  for e, c in cells]
        bars = ax.bar(x + offset, shares, width=0.34, color=colour, label=name, zorder=3)
        ax.bar_label(bars, fmt="%.0f%%", padding=2, fontsize=8, color=INK_2)
    ax.set_xticks(x, labels)
    ax.set_ylabel("insertions that searched (%)")
    ax.set_ylim(0, 72)
    ax.axvline(1.5, color=GRID, linewidth=1.0)
    ax.text(0.5, 68, "no per-object error", ha="center", fontsize=8.5, color=INK_2)
    ax.text(2.5, 68, "±2 mm per-object error", ha="center", fontsize=8.5, color=INK_2)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.88))
    ax.set_title("Spiral search usage", loc="left")

    # Right panel: the non-cancelling connector-vs-socket error that drives it.
    groups, labels2, colours = [], [], []
    for error in (0.0, 0.002):
        for name, colour in (("upright", BLUE), ("lying", ORANGE)):
            cell = df[(df.object_error == error) & (df.orientation_label == name)]
            groups.append(cell.relative_xy_error_mm.dropna().values)
            labels2.append(f"{name}\n{'0' if error == 0 else '±2 mm'}")
            colours.append(colour)
    box(ax2, groups, labels2, colours)
    ax2.set_ylabel("connector − socket error (mm)")
    ax2.set_title("Perception error that does not cancel", loc="left")
    save(fig, "f5_search_usage")


def fig_cycle_and_force() -> None:
    """Full task: cycle time per connector and peak sustained contact force, upright vs lying."""
    df = _m5()
    df["orientation_label"] = np.where(df.lying, "lying", "upright")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.2))
    for axis, column, label, limit in ((ax, "cycle_time", "cycle time per connector (s)", None),
                                       (ax2, "peak_force_10ms", "peak sustained force (N, 10 ms mean)", 30)):
        groups = [df[df.orientation_label == name][column].dropna().values for name in ("upright", "lying")]
        box(axis, groups, ["upright", "lying"], [BLUE, ORANGE])
        axis.set_ylabel(label)
        for position, values in enumerate(groups, start=1):  # boxes hide the tail; the worst case matters here
            axis.plot(position, values.max(), marker="v", markersize=6, color=INK_2, linestyle="none")
            axis.annotate(f"max {values.max():.0f}", (position, values.max()), xytext=(8, -3),
                          textcoords="offset points", color=INK_2, fontsize=8)
        if limit:
            axis.axhline(limit, color=INK_2, linestyle="--", linewidth=1.0)
            axis.annotate("30 N cap", (0.55, limit), xytext=(0, 4), textcoords="offset points", color=INK_2, fontsize=8)
            axis.set_ylim(0, limit * 1.15)
    fig.suptitle("Full task (640 runs): a lying connector costs ~7 s more; force stays far below the cap", fontsize=10, y=1.02)
    save(fig, "f6_cycle_and_force")


def fig_perception_error() -> None:
    """Perception error vs depth noise, for upright connectors, lying connectors and sockets."""
    df = pd.read_csv(RESULTS_DIR / "perception_check" / "rows.csv")
    df = df[df.detected & df.xy_error.notna()]
    kinds = [("connector", False, "upright connector", BLUE), ("connector", True, "lying connector", ORANGE),
             ("socket", False, "socket", AQUA)]
    sigmas = sorted(df.noise_sigma.unique())
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    width = 0.24
    for index, (part, lying, label, colour) in enumerate(kinds):
        offset = (index - 1) * (width + 0.03)
        groups, positions = [], []
        for pos, sigma in enumerate(sigmas):
            cell = df[(df.part == part) & (df.lying == lying) & (df.noise_sigma == sigma)]
            groups.append(cell.xy_error.values * 1e3)
            positions.append(pos + offset)
        parts = ax.boxplot(groups, positions=positions, widths=width, showfliers=False, patch_artist=True,
                           medianprops=dict(color=SURFACE, linewidth=1.2),
                           whiskerprops=dict(color=INK_2, linewidth=0.9), capprops=dict(color=INK_2, linewidth=0.9))
        for patch in parts["boxes"]:
            patch.set(facecolor=colour, edgecolor=colour, alpha=0.85, linewidth=0)
        ax.plot([], [], color=colour, linewidth=6, label=label)  # legend proxy
    ax.set_xticks(range(len(sigmas)), [f"σ = {s * 1e3:.0f} mm" for s in sigmas])
    ax.set_ylabel("position error vs ground truth (mm)")
    ax.set_xlabel("depth noise")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.set_title("Perception: position error (40 layouts per level; identification was 100% everywhere)", loc="left")
    save(fig, "f7_perception_error")


def fig_completion_checks() -> None:
    """The visual check separates a seated connector from one held up by debris; the force check cannot."""
    clean, blocked = _m5(), pd.read_csv(RESULTS_DIR / "m5_debris" / "connectors.csv")
    blocked_here = blocked[blocked.debris_here]
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    bins = np.arange(28.5, 36.5, 0.25)
    ax.hist(clean.visual_height_mm.dropna(), bins=bins, color=BLUE, alpha=0.9, label=f"clean socket (n = {len(clean)})", zorder=3)
    ax.hist(blocked_here.visual_height_mm.dropna(), bins=bins, color=ORANGE, alpha=0.9,
            label=f"4 mm debris in the socket (n = {len(blocked_here)})", zorder=3)
    ax.axvline(33.0, color=INK_2, linestyle="--", linewidth=1.0)
    ax.set_yscale("log")
    ax.set_ylim(1, 3000)
    ax.text(32.85, 1.4, "visual check threshold (33 mm)", color=INK_2, fontsize=8, rotation=90, va="bottom", ha="right")
    ax.annotate("the force check accepted\nevery one of these", (blocked_here.visual_height_mm.mean(), 25),
                xytext=(31.9, 320), color=ORANGE, fontsize=8, ha="center",
                arrowprops=dict(arrowstyle="->", color=ORANGE, linewidth=1.0))
    ax.set_xlabel("connector top above its socket (mm), from the camera after release")
    ax.set_ylabel("connectors")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Completion checks: vision catches what force cannot", loc="left")
    save(fig, "f8_completion_checks")


FIGURE_FUNCTIONS = {
    "scene": fig_scene,
    "compliance": fig_compliance_checks,
    "controller_success": fig_controller_success,
    "controller_cycle_time": fig_controller_cycle_time,
    "force_trace": fig_force_trace,
    "search": fig_search_usage,
    "cycle_force": fig_cycle_and_force,
    "perception": fig_perception_error,
    "completion": fig_completion_checks,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=list(FIGURE_FUNCTIONS), help="figures to draw (default: all)")
    args = parser.parse_args()
    style()
    for name in args.only or FIGURE_FUNCTIONS:
        FIGURE_FUNCTIONS[name]()


if __name__ == "__main__":
    main()
