import matplotlib.pyplot as plt 
import numpy as np
from functools import wraps
import itertools
from scipy.stats import linregress
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import math
from demo.figures.fig_params import PlotStyle, MY_STYLE, MODEL_MAPPINGS


def apply_plot_style(plot_style: PlotStyle):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Ensure external styles are loaded
            plot_style._ensure_styles_available()

            # Apply styles and rcParams
            with plt.style.context(plot_style.styles):
                with plt.rc_context(rc=plot_style.rc):
                    return func(*args, **kwargs)
        return wrapper
    return decorator

# FIGURE 1

## 1.A
def _curve(ax, src, dst, rad, colour, y_pos, x0):
    ax.add_patch(FancyArrowPatch(
        (x0, y_pos[src]), (x0, y_pos[dst]),
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-", linewidth=MY_STYLE.linewidth,
        color=colour, alpha=0.9, zorder=1))

def _decorate_axis(ax, y_pos, *, arrow_label_fs, axis_label_fs, lw):
    y_min, y_max = min(y_pos.values()) - 0.12, max(y_pos.values()) + 0.12

    ax.plot([-0.55, -0.55], [y_min, y_max], color="black", lw=lw)
    ax.text(-0.55, y_max + 0.02, "Top",    ha="center", va="bottom", fontsize=arrow_label_fs)
    ax.text(-0.55, y_min - 0.02, "Bottom", ha="center", va="top",    fontsize=arrow_label_fs)
    ax.set_xlabel("Areas", fontsize=axis_label_fs)
    ax.set_ylabel("Anatomical hierarchy score", fontsize=axis_label_fs)

    arrow_x = 0.25
    start, end = y_pos["V1"] + 0.1, y_pos["AL"] - 0.1
    mid_y = (start + end) / 2

    ax.add_patch(FancyArrowPatch((-arrow_x, start), (-arrow_x, end),
                                 arrowstyle='-|>', linewidth=1.,
                                 mutation_scale=6, color='black'))
    ax.add_patch(FancyArrowPatch(( arrow_x, end),  ( arrow_x, start),
                                 arrowstyle='-|>', linewidth=1.,
                                 mutation_scale=6, color='black'))
    ax.text(-arrow_x - 0.04, mid_y, "Feedforward", rotation=90,
            va="center", ha="right", fontsize=arrow_label_fs+1)
    ax.text( arrow_x + 0.04, mid_y, "Feedback",   rotation=270,
            va="center", ha="left",  fontsize=arrow_label_fs+1)

    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

@apply_plot_style(MY_STYLE)
def plot_hierarchy(
        areas= ["V1", "LM", "RL", "AL"],
        savepath: str = None
    ):
    """
    Draw the anatomical-hierarchy spindle panel with standard style.
    """

    # Fallback colour palette (V1→AL order)
    default_palette = ["#8e44ad", "#2980b9", "#16a085", "#c0392b"]
    colours = {
        a: MY_STYLE.area_colors.get(a, default_palette[i]) for i, a in enumerate(areas)
    }

    # Harris-like vertical coordinates
    y_pos = MY_STYLE.area_hierarchy_scores

    fig, ax = plt.subplots(figsize=MY_STYLE.figsize_single, dpi=300)
    ax.set_aspect("equal")

    # Font sizes
    base_fs  = plt.rcParams.get("font.size", 10)
    label_fs = plt.rcParams.get("axes.labelsize", base_fs)
    axis_fs  = label_fs + 2
    node_fs  = base_fs 
    arrow_fs = base_fs 

    # Nodes (areas)
    x0, r = 0.0, 0.045
    for area in areas:
        ax.add_patch(Circle((x0, y_pos[area]), r,
                            facecolor=colours[area], edgecolor="black",
                            lw=MY_STYLE.linewidth, zorder=3))
        ax.text(x0 + 0.05, y_pos[area], area,
                va="center", ha="left", fontweight="bold", fontsize=node_fs)

    # Draw arcs (both directions), colored by source area
    base_rad, spread = 0.30, 0.10
    for src, dst in itertools.permutations(areas, 2):
        if src == dst:
            continue

        src_idx = areas.index(src)
        dst_idx = areas.index(dst)
        rank_dist = abs(dst_idx - src_idx) - 1
        rad = base_rad + spread * rank_dist

        if src_idx < dst_idx:
            # Feedforward: draw on left, color by source
            _curve(ax, src, dst, -rad, colours[src], y_pos, x0)
        elif src_idx > dst_idx:
            # Feedback: draw on right, color by source
            _curve(ax, src, dst, -rad, colours[src], y_pos, x0)

    # Axis decorations
    _decorate_axis(ax, y_pos,
                   arrow_label_fs=arrow_fs,
                   axis_label_fs=axis_fs,
                   lw=MY_STYLE.linewidth)

    # plt.tight_layout()
    plt.show()
    if savepath:
        fig.savefig(savepath, transparent=True, dpi=300)
        
# 1.B
@apply_plot_style(MY_STYLE)
def plot_dotplot_from_dict(data_dict, ylabel, title=None, xlabel=None, ylim=None, xlim=None, use_x=False, save_path=None):
    areas = list(data_dict.keys())
    xy_pairs = list(data_dict.values())
    x_vals, y_vals = zip(*xy_pairs)
    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)
    sems = np.zeros_like(y_vals)  # placeholder SEMs

    colors = [MY_STYLE.area_colors.get(a, 'gray') for a in areas]
    markers = ['8'] * len(areas)

    fig, ax = plt.subplots(figsize=MY_STYLE.figsize_single, dpi=300)

    if use_x:
        x_plot = x_vals
    else:
        x_plot = np.arange(len(areas))

    # Plot dots with SEMs
    for i, (x, y, y_err, color, marker) in enumerate(zip(x_plot, y_vals, sems, colors, markers)):
        ax.errorbar(
            x, y,
            yerr=y_err,
            fmt=marker,
            capsize=MY_STYLE.capsize,
            color=color,
            markersize=MY_STYLE.markersize,
            linewidth=MY_STYLE.linewidth,
            elinewidth=MY_STYLE.linewidth,
            markeredgewidth=MY_STYLE.linewidth,
            label=areas[i]
        )

    # Regression line
    if use_x:
        coeffs = np.polyfit(x_vals, y_vals, 1)
        poly = np.poly1d(coeffs)
        x_fit = np.linspace(np.minimum(x_vals, xlim[0]), np.maximum(x_vals, xlim[1]), 100)
        ax.plot(x_fit, poly(x_fit), linestyle='--', color='gray', linewidth=MY_STYLE.linewidth, alpha=0.75)

        slope, intercept, r_value, p_value, std_err = linregress(x_vals, y_vals)
        print(f"Slope: {slope}, Intercept: {intercept}, R-squared: {r_value**2}, P-value: {p_value}, Std Err: {std_err}")
    else:
        plt.plot(x_vals, y_vals, linestyle='--', color='gray', linewidth=MY_STYLE.linewidth, alpha=0.75)
        ax.set_xticks(np.arange(len(areas)))
        ax.set_xticklabels(areas)

    ax.set_ylabel(ylabel, fontsize=MY_STYLE.rc.get("axes.labelsize", 12))
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=MY_STYLE.rc.get("axes.labelsize", 12))
    if title:
        ax.set_title(title, fontsize=MY_STYLE.rc.get("axes.titlesize", 14))
    if ylim:
        ax.set_ylim(ylim)
    if xlim:
        ax.set_xlim(xlim)
        
    # Add upward ticks with labels
    y0 = ax.get_ylim()[0]
    tick_height = 0.025 * (ax.get_ylim()[1] - y0)
    for xpos, area in zip(x_vals, areas):
        ax.plot([xpos, xpos], [y0, y0 + tick_height], color='black', linewidth=MY_STYLE.linewidth)
        ax.text(
            xpos+ 0.007,
            y0 + tick_height + 0.005 * (ax.get_ylim()[1] - y0),
            area,
            ha='center',
            va='bottom',
            fontsize=MY_STYLE.rc.get("font.size", 18),
            rotation=60,
            clip_on=False
        )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    # plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight', dpi=300, )
    plt.show()

# FIGURE 2/3

##  Main orchestrator
def draw_model_legend(
    ax,
    model_ids,
    *,
    ncols=1,          # how many columns to use
    xpad=0.8,         # horizontal spacing between columns
    ypad=0.8,         # vertical spacing between rows
    marker_x=0.10,    # x‑position of marker inside each cell (0 → left edge)
    text_x=0.25,      # x‑position where the label text starts
):
    """
    Render a shared model legend (marker + label) inside *ax* using an
    n‑column layout.

    Parameters
    ----------
    ax        : matplotlib.axes.Axes
    model_ids : list[str]
        Keys recognised by MODEL_MAPPINGS / MY_STYLE.model_colors / model_markers
    ncols     : int
        Number of columns in the legend grid (≥1).
    xpad, ypad : float
        Spacing between columns / rows in axis‑coordinate units.
    marker_x, text_x : float
        Horizontal offsets (axis coords) for marker and label within a cell.
    """
    ax.axis("off")

    n_models = len(model_ids)
    ncols = max(1, min(ncols, n_models))
    nrows = math.ceil(n_models / ncols)

    # set axis limits so all cells fit
    ax.set_xlim(-0.5, (ncols - 1) * xpad + 0.5)
    ax.set_ylim(-0.5, (nrows - 1) * ypad + 0.5)

    for idx, mid in enumerate(model_ids):
        row = nrows - 1 - (idx // ncols)       # top → bottom
        col = idx % ncols

        x0 = col * xpad
        y0 = row * ypad

        label  = MODEL_MAPPINGS.get(mid, mid)
        color  = MY_STYLE.model_colors.get(label, "gray")
        marker = MY_STYLE.model_markers.get(label, "o")

        ax.scatter(
            x0 + marker_x, y0,
            s=MY_STYLE.markersize ** 2,
            marker=marker, color=color,
            linewidths=MY_STYLE.linewidth,
        )
        ax.text(
            x0 + text_x, y0, label,
            ha="left", va="center",
            fontsize=MY_STYLE.rc["font.size"],
        )

def _draw_model_row_legend(ax, model_ids, *, xpad=0.8):
    """
    Render a shared legend inside `ax`, showing each model’s marker & colour.

    Parameters
    ----------
    ax        : matplotlib.axes.Axes
    model_ids : list[str]   – keys that appear in MODEL_MAPPINGS /
                              MY_STYLE.model_colors / MY_STYLE.model_markers
    xpad      : float       – spacing between consecutive entries
    """
    ax.axis("off")
    ax.set_xlim(-0.5, (len(model_ids) - 1) * xpad + 0.5)
    ax.set_ylim(-1, 1)

    for i, mid in enumerate(model_ids):
        label  = MODEL_MAPPINGS.get(mid, mid)
        # if label == "hierarchical (anatomy)":
        #     label = "hierarchical (V1-LM/RL-AL (anatomy))"
        color  = MY_STYLE.model_colors.get(label, "gray")
        marker = MY_STYLE.model_markers.get(label, "o")

        x = i * xpad
        ax.scatter(
            x, 0, s=MY_STYLE.markersize**2,
            marker=marker, color=color, linewidths=MY_STYLE.linewidth,
        )
        ax.text(
            x + 0.12, 0, label,
            ha="left", va="center",
            fontsize=MY_STYLE.rc["font.size"],
        )
        
def _draw_model_column_legend(ax, model_ids, *, ypad=0.8):
    """
    Render a shared legend inside `ax`, showing each model’s marker & colour
    as a vertical list.

    Parameters
    ----------
    ax        : matplotlib.axes.Axes
    model_ids : list[str]   – keys that appear in MODEL_MAPPINGS /
                              MY_STYLE.model_colors / MY_STYLE.model_markers
    ypad      : float       – vertical spacing between consecutive entries
    """
    ax.axis("off")
    ax.set_ylim(-0.5, (len(model_ids) - 1) * ypad + 0.5)
    ax.set_xlim(0, 1)                       # narrow horizontal band is enough

    for i, mid in enumerate(model_ids):
        label  = MODEL_MAPPINGS.get(mid, mid)
        # if label == "hierarchical (anatomy)":
        #     label = "hierarchical (V1-LM/RL-AL (anatomy))"
        color  = MY_STYLE.model_colors.get(label, "gray")
        marker = MY_STYLE.model_markers.get(label, "o")

        y = (len(model_ids) - 1 - i) * ypad  # top‑to‑bottom order
        ax.scatter(
            0.1, y, s=MY_STYLE.markersize**2,
            marker=marker, color=color, linewidths=MY_STYLE.linewidth,
        )
        ax.text(
            0.25, y, label,
            ha="left", va="center",
            fontsize=MY_STYLE.rc["font.size"],
        )

@apply_plot_style(MY_STYLE) 
def build_summary_figure(
    barplot_fn,
    row1_fns,
    row2_fns,
    model_ids,
    *,
    single_row_fns=None,          # ← NEW
    figsize=(12, 8),
    wspace=0.35,
    hspace=0.45,
    legend_type="vertical",
):
    """
    If *single_row_fns* is provided (iterable of 3 callables), a compact
    1‑row × 3‑column figure is produced; otherwise the original 3‑row
    composite is built.
    -------------------------------------------------------------------
    All original parameters and return‑values remain unchanged.
    """

    # ──────────────────────────────────────────────────────────────────
    # 1. ─ Single‑row mode  (early‑exit)                               ─
    # ──────────────────────────────────────────────────────────────────
    if single_row_fns is not None:
        if len(single_row_fns) != 3:
            raise ValueError("single_row_fns must contain exactly three callables")
        fig = plt.figure(figsize=figsize, constrained_layout=False, dpi=300)
        gs  = fig.add_gridspec(
            nrows=1, ncols=6,
            width_ratios=[1]*6,
            wspace=wspace, hspace=0,
        )

        ax_c1 = fig.add_subplot(gs[0, 0:2])
        ax_c2 = fig.add_subplot(gs[0, 2:4])
        ax_c3 = fig.add_subplot(gs[0, 4:6])

        for fn, ax in zip(single_row_fns, (ax_c1, ax_c2, ax_c3)):
            fn(ax)
            ax.set_box_aspect(1)

        axes = {
            "col1": ax_c1,
            "col2": ax_c2,
            "col3": ax_c3,
        }
        
        letters = ["A", "B", "C"]
        row_anchor_axes = [ax_c1, ax_c2, ax_c3]   # first panel of each row

        for letter, _ax in zip(letters, row_anchor_axes):
            if letter == "A":
                x0_change = -0.052
            elif letter == "B":
                x0_change = 0.01
            else:
                x0_change = 0.0
            pos = _ax.get_position()                   # in figure coords
            fig.text(
                pos.x0 + x0_change,                         # a little left of the panel
                pos.y0 + pos.height + 0.0445,                   # top of the panel
                letter,
                ha="right", va="top",
                fontweight="bold",
                fontsize=MY_STYLE.rc["axes.titlesize"],
        )
        return fig, axes

    # ──────────────────────────────────────────────────────────────────
    # 2. ─ Original 3‑row composite (unchanged)                        ─
    # ──────────────────────────────────────────────────────────────────
    if len(row1_fns) != 3 or len(row2_fns) != 3:
        raise ValueError("row1_fns and row2_fns must each contain exactly three callables")

    fig = plt.figure(figsize=figsize, constrained_layout=False, dpi=300)
    gs  = fig.add_gridspec(
        nrows=5, ncols=6,
        height_ratios=[1, 1, 1, 1, 1],
        width_ratios=[1]*6,
        wspace=wspace, hspace=hspace,
    )

    # (everything below is byte‑for‑byte the same as before …)
    # ── Top row ───────────────────────────────────────────────────────
    ax_bar    = fig.add_subplot(gs[0, 0:3])
    ax_legend = fig.add_subplot(gs[0, 3:6])
    if legend_type == "vertical":
        _draw_model_column_legend(ax_legend, model_ids)
    else:
        draw_model_legend(ax_legend, model_ids, ncols=len(model_ids))

    barplot_fn(ax_bar)

    # ── Middle row ────────────────────────────────────────────────────
    ax_r1c1 = fig.add_subplot(gs[1:3, 0:2])
    ax_r1c2 = fig.add_subplot(gs[1:3, 2:4])
    ax_r1c3 = fig.add_subplot(gs[1:3, 4:6])
    for fn, ax in zip(row1_fns, (ax_r1c1, ax_r1c2, ax_r1c3)):
        fn(ax); ax.set_box_aspect(1)

    # ── Bottom row ────────────────────────────────────────────────────
    ax_r2c1 = fig.add_subplot(gs[3:, 0:2])
    ax_r2c2 = fig.add_subplot(gs[3:, 2:4])
    ax_r2c3 = fig.add_subplot(gs[3:, 4:6])
    for fn, ax in zip(row2_fns, (ax_r2c1, ax_r2c2, ax_r2c3)):
        fn(ax); ax.set_box_aspect(1)

    # vertical nudge for bottom row
    _gap = 0.03
    for ax in (ax_r2c1, ax_r2c2, ax_r2c3):
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 + _gap, pos.width, pos.height])

    axes = {
        "bar"        : ax_bar,
        "legend"     : ax_legend,
        "row1_rf"    : ax_r1c1,
        "row1_hoell" : ax_r1c2,
        "row1_froud" : ax_r1c3,
        "row2_rf"    : ax_r2c1,
        "row2_hoell" : ax_r2c2,
        "row2_froud" : ax_r2c3,
    }

    letters = ["A", "B", "C"]
    row_anchor_axes = [ax_bar, ax_r1c1, ax_r2c1]   # first panel of each row

    for letter, _ax in zip(letters, row_anchor_axes):
        pos = _ax.get_position()                   # in figure coords
        fig.text(
            pos.x0 - 0.08,                         # a little left of the panel
            pos.y0 + pos.height + 0.033,                   # top of the panel
            letter,
            ha="right", va="top",
            fontweight="bold",
            fontsize=MY_STYLE.rc["axes.titlesize"],
    )

    return fig, axes

## Single plot functions

### unitperformance
@apply_plot_style(MY_STYLE)
def plot_grouped_dotplot_model_area_with_sems(
    summary,
    ylabel,
    *,
    ax=None,
    area_order=('V1', 'LM', 'RL', 'AL'),
    sems=True,
    ylim=None,
    xlabel=None,
    title=None,
    width=0.18,             # bar width per model
):
    """
    Grouped bar‑and‑error plot with *areas* on the x‑axis and
    individual *models* as the coloured bars inside each group.

    Parameters
    ----------
    summary : dict
        {model_name: {area_name: values}}
    """

    # ── 0. Axes bookkeeping ────────────────────────────────────────────
    if ax is None:
        fig, ax = plt.subplots(figsize=MY_STYLE.figsize_single, dpi=300)
    else:
        fig = ax.figure

    # ── 1. Prepare data ────────────────────────────────────────────────
    areas   = list(area_order)
    models  = list(summary.keys())

    x       = np.arange(len(areas))                 # group positions (areas)
    offsets = np.linspace(
        -width * (len(models) - 1) / 2,
         width * (len(models) - 1) / 2,
        len(models)
    )

    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # ── 2. Draw bars per model inside each area group ──────────────────
    for j, model in enumerate(models):
        label  = MODEL_MAPPINGS.get(model, model)
        color  = MY_STYLE.model_colors.get(label, 'gray')

        for i, area in enumerate(areas):
            values = summary[model][area]
            mean   = np.mean(values)
            err    = (np.std(values, ddof=1) /
                      np.sqrt(len(values)) if sems and len(values) > 1 else 0)

            xpos = x[i] + offsets[j]

            ax.bar(
                xpos, mean,
                width=width,
                yerr=err,
                color=color,
                alpha=0.7,
                edgecolor='black',
                linewidth=MY_STYLE.linewidth,
                zorder=1,
            )
            ax.errorbar(
                xpos, mean,
                yerr=err,
                fmt='none',
                capsize=MY_STYLE.capsize,
                color=color,
                linewidth=MY_STYLE.linewidth,
                elinewidth=MY_STYLE.linewidth,
                zorder=2,
            )

    # ── 3. Cosmetics ──────────────────────────────────────────────────
    ax.set_xticks(x)
    ax.set_xticklabels(areas, rotation=0, fontsize=MY_STYLE.rc.get("xtick.labelsize", 10))

    ax.set_ylabel(ylabel, fontsize=MY_STYLE.rc.get("axes.labelsize", 12))
    if xlabel:
        ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title, fontsize=MY_STYLE.rc.get("axes.titlesize", 14))
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)

    return ax

### ARF
@apply_plot_style(MY_STYLE)
def plot_median_area_per_model(
    model_medians,
    *,
    ax=None,                          # ←────────── NEW
    area_order=('V1', 'LM', 'RL', 'AL'),
    regline=False,
    ylim=None,
    xlim=None,
    ylabel=None,
    xlabel=None,
    title=None,
    loc='best',
    bbox_to_anchor=None,
    legend=False,
    yticks=None,
):
    """
    Draw median (or mean) receptive‑field size vs anatomical hierarchy for
    each model, with optional regression lines.

    Parameters
    ----------
    model_medians : dict
        {model_name: {area_name: 1‑D array‑like}}
    ax : matplotlib.axes.Axes or None
        Existing axes to draw on; if None a new figure/axes is created.
    """

    # ── 0. Axes bookkeeping ────────────────────────────────────────────
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=MY_STYLE.figsize_single, dpi=300)
        created_fig = True
    else:
        fig = ax.figure

    # ── 1. Prepare data ────────────────────────────────────────────────
    areas = list(area_order)
    x = np.array([MY_STYLE.area_hierarchy_scores[a] for a in areas])

    ax.grid(axis='y', linestyle='--', alpha=0.5)
    labels = list(model_medians.keys())
    colors = [MY_STYLE.model_colors.get(label, 'gray') for label in labels]
    markers = [MY_STYLE.model_markers.get(label, 'o') for label in labels]
    # ── 2. Plot per‑model curves / regression lines ────────────────────
    for model, medians in model_medians.items():
        label  = MODEL_MAPPINGS.get(model, model)
        color  = MY_STYLE.model_colors.get(label)
        marker = MY_STYLE.model_markers.get(label, 'o')

        y      = [np.mean(medians[a]) for a in areas]
        y_err  = [np.std(medians[a]) / np.sqrt(len(medians[a])) for a in areas]

        if regline:
            # Flatten data for regression
            y_reg = np.concatenate([medians[a] for a in areas])
            x_reg = np.concatenate([
                np.full(len(medians[a]), MY_STYLE.area_hierarchy_scores[a])
                for a in areas
            ])

            slope, intercept, r_val, p_val, std_err = linregress(x_reg, y_reg)

            # choose fit domain
            xmin = xlim[0] if xlim is not None else x.min()
            xmax = xlim[1] if xlim is not None else x.max()
            x_fit = np.linspace(xmin, xmax, 200)
            y_fit = slope * x_fit + intercept

            # significance label (monospace spacing)
            asterisks = {0.001: '*** ', 0.01: '**  ', 0.05: '*   ', 0.1: '.   '}
            significance = next((s for thr, s in asterisks.items() if p_val < thr), 'n.s.')
            label_leg = f"{significance}"
            ax.plot(
                x_fit, y_fit, linestyle='--', color=color,
                linewidth=MY_STYLE.linewidth, alpha=0.5, zorder=0,
            )
        else:
            ax.plot(
                x, y, marker=marker, color=color,
                linewidth=MY_STYLE.linewidth, markersize=MY_STYLE.markersize,
                alpha=0.8, zorder=2,
                label=label if legend else None,
            )

        # Error bars
        ax.errorbar(
            x, y, yerr=y_err,
            fmt=marker if regline else 'none',
            capsize=MY_STYLE.capsize,
            markersize=MY_STYLE.markersize,
            color=color, alpha=0.8,
            elinewidth=1, zorder=1,
            label=label_leg
        )

    # ── 3. Cosmetic axes tweaks ────────────────────────────────────────
    ax.set_xlabel(xlabel)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)
    if yticks is not None:
        ax.set_yticks(yticks)
    if title:
        ax.set_title(title,
                    fontsize=MY_STYLE.rc['axes.titlesize'],
                    pad=10)                    # first row (larger)

    if ylabel:
        # second row, slightly below the first and inside the axes frame
        ax.text(0.5, 1.1, ylabel,
                transform=ax.transAxes,
                ha='center', va='top',
                fontsize=MY_STYLE.rc['axes.labelsize'])

    # custom little upward ticks + labels under x‑axis
    y0 = ax.get_ylim()[0]
    tick_h = 0.025 * (ax.get_ylim()[1] - y0)
    for xpos, area in zip(x, areas):
        ax.plot([xpos, xpos], [y0, y0 + tick_h],
                color='black', linewidth=MY_STYLE.rc['ytick.major.width'])
        ax.text(
            xpos + 0.008, y0 + tick_h * 1.3, area,
            ha='center', va='bottom',
            fontsize=MY_STYLE.rc['axes.labelsize'], rotation=60, clip_on=False,
        )

    # Legend (optional, unique labels only)
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        uniq = dict()
        for h, l in zip(handles, labels):
            if h not in uniq:
                uniq[h] = l
            # if l not in uniq:
            #     uniq[l] = h
        ax.legend(
            uniq.keys(), uniq.values(),
            frameon=False, fontsize=5, handlelength=0.5,
            loc=loc, bbox_to_anchor=bbox_to_anchor,
            prop={
                'family': 'monospace',
                'size': MY_STYLE.rc['legend.fontsize'],
            },

        )
        

    # Clean up spines / ticks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)

    return ax

### HOELLER/FROUD
@apply_plot_style(MY_STYLE)
def plot_single_neuron_perf_per_model_with_sems(
    summary,
    ylabel,
    *,
    ax=None,                               # ←─ NEW: accept existing Axes
    area_order=('V1', 'LM', 'RL', 'AL'),
    sems=False,
    ylim=None,
    xlim=None,
    xlabel=None,
    ytick=None,
    title=None,
    legend=False,
    loc='best',
    bbox_to_anchor=None,
    regline=False,
):
    """
    Plot per‑model single‑neuron performance (Hoeller / Froudarakis tasks)
    against anatomical hierarchy score, with optional regression lines
    and SEM error bars.

    Parameters
    ----------
    summary : dict
        {model_name: {area_name: 1‑D array‑like}}
    ylabel : str
        Label for y‑axis.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  If None, a new figure/axes is created so the
        function still works stand‑alone.
    """

    # ── 0. Figure / axes bookkeeping ───────────────────────────────────
    if ax is None:
        fig, ax = plt.subplots(figsize=MY_STYLE.figsize_single, dpi=300)
    else:
        fig = ax.figure

    # ── 1. Prepare data ────────────────────────────────────────────────
    areas = list(area_order)
    x = np.array([MY_STYLE.area_hierarchy_scores[a] for a in areas])

    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # ── 2. Plot per‑model curves / regressions ─────────────────────────
    for model, area_data in summary.items():
        label  = MODEL_MAPPINGS.get(model, model)
        color  = MY_STYLE.model_colors.get(label)
        marker = MY_STYLE.model_markers.get(label, 'o')

        y      = [np.mean(area_data[a]) for a in areas]
        y_err  = [np.std(area_data[a], ddof=1) / np.sqrt(len(area_data[a]))
                  for a in areas] if sems else None

        # Default strings in case regline=False (avoid UnboundLocal)
        significance, p_val = 'n.s.', np.nan

        if regline:
            # flatten data for regression
            y_reg = np.concatenate([area_data[a] for a in areas])
            x_reg = np.concatenate([
                np.full(len(area_data[a]), MY_STYLE.area_hierarchy_scores[a])
                for a in areas
            ])
            slope, intercept, r_val, p_val, _ = linregress(x_reg, y_reg)

            # Fit line range
            xmin = xlim[0] if xlim is not None else x.min()
            xmax = xlim[1] if xlim is not None else x.max()
            x_fit = np.linspace(xmin, xmax, 200)
            y_fit = slope * x_fit + intercept

            # significance stars
            for thr, sym in {0.001:'*** ', 0.01:'**  ', 0.05:'*   ', 0.1:'.   '}.items():
                if p_val < thr:
                    significance = sym
                    break
            label = f"{significance}"
            ax.plot(
                x_fit, y_fit,
                color=color, linestyle='--',
                linewidth=MY_STYLE.linewidth, alpha=0.5, zorder=0,
            )
            
        else:
            # main markers / line
            ax.plot(
                x, y,
                marker=marker,
                color=color,
                linewidth=MY_STYLE.linewidth,
                markersize=MY_STYLE.markersize,
                alpha=0.8,
                zorder=2,
                label=label if not regline and legend else None,
            )

        # error bars
        if sems and y_err is not None:
            ax.errorbar(
                x, y, yerr=y_err,
                fmt=marker if regline else 'none',
                capsize=MY_STYLE.capsize,
                markersize=MY_STYLE.markersize,
                color=color,
                alpha=0.8,
                elinewidth=1,
                zorder=1,
                label=label if legend else None,
            )

    # ── 3. Cosmetic axes tweaks ────────────────────────────────────────
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ytick is not None:
        ax.set_yticks(ytick)
        
    if title:
        ax.set_title(title,
                    fontsize=MY_STYLE.rc['axes.titlesize'],
                    pad=10)                    # first row (larger)

    if ylabel:
        # second row, slightly below the first and inside the axes frame
        ax.text(0.5, 1.1, ylabel,
                transform=ax.transAxes,
                ha='center', va='top',
                fontsize=MY_STYLE.rc['axes.labelsize'])

    # upward hierarchy ticks under x‑axis
    y0 = ax.get_ylim()[0]
    tick_h = 0.025 * (ax.get_ylim()[1] - y0)
    for xpos, area in zip(x, areas):
        ax.plot([xpos, xpos], [y0, y0 + tick_h],
                color='black', linewidth=MY_STYLE.rc['ytick.major.width'])
        ax.text(
            xpos + 0.0075,
            y0 + tick_h * 1.3,
            area, rotation=60,
            ha='center', va='bottom',
            fontsize=MY_STYLE.rc['axes.labelsize'],
            clip_on=False,
        )

    # legend (deduplicated) — optional
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        uniq = {}
        for h, l in zip(handles, labels):
            if h not in uniq:
                uniq[h] = l
        ax.legend(
            uniq.keys(), uniq.values(),
            frameon=False, fontsize=MY_STYLE.rc['legend.fontsize'],
            handlelength=0.5, loc=loc, bbox_to_anchor=bbox_to_anchor,
            prop={
                'family': 'monospace',
                'size': MY_STYLE.rc['legend.fontsize'],
            }   
        )

    # clean spines / ticks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)

    return ax

### KENDALLS TAU
@apply_plot_style(MY_STYLE)
def dotplot_name_vs_model_tau_with_sem(
    x_labels, y_values, y_sems,
    ylabel,
    *,
    ax=None,                    # ←─ NEW: accept an Axes
    xlabel=None,
    title=None,
    ylim=None,
    yticks=None,
    xlim=None,
):
    """
    Scatter + error‑bar dot‑plot of model‑data alignment scores
    (e.g. Kendall’s τ) for multiple models.

    Parameters
    ----------
    x_labels : list[str]      – model identifiers
    y_values : list[float]    – central values
    y_sems   : list[float]    – SEMs (same length as y_values)
    ax       : matplotlib.axes.Axes or None
    """

    # ── 0. Figure / axes bookkeeping ───────────────────────────────────
    if ax is None:
        fig, ax = plt.subplots(figsize=MY_STYLE.figsize_single, dpi=300)
    else:
        fig = ax.figure

    # ── 1. Prep colours / markers / positions ─────────────────────────
    labels  = [MODEL_MAPPINGS.get(lbl, lbl) for lbl in x_labels]
    colors  = [MY_STYLE.model_colors.get(lbl, 'gray') for lbl in labels]
    markers = [MY_STYLE.model_markers.get(lbl, 'o')   for lbl in labels]

    x = np.linspace(0, 1, len(labels))        # evenly spaced between 0 and 1

    # ── 2. Background guideline grid ──────────────────────────────────
    ax.axhline(0, color='gray', linestyle='--',
               linewidth=MY_STYLE.linewidth, alpha=0.8)

    vline_ymin = ylim[0] if ylim is not None else -1
    vline_ymax = ylim[1] if ylim is not None else 1
    ax.vlines(
        x,
        ymin=vline_ymin, ymax=vline_ymax,
        color='gray', linewidth=MY_STYLE.linewidth, alpha=0.1,
    )

    # ── 3. Scatter points + error bars ────────────────────────────────
    for x_val, y_val, y_err, c, m in zip(x, y_values, y_sems, colors, markers):
        ax.errorbar(
            x_val, y_val,
            yerr=y_err,
            fmt=m,
            capsize=MY_STYLE.capsize,
            color=c,
            markersize=MY_STYLE.markersize,
            linewidth=MY_STYLE.linewidth,
            elinewidth=MY_STYLE.linewidth,
            markeredgewidth=MY_STYLE.linewidth,
        )

    # ── 4. Axis labelling / limits / ticks ────────────────────────────
    ax.set_xticks(x)
    ax.set_xticklabels([])          # hide labels (can be added later if desired)

    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title)

    if ylim is not None:
        ax.set_ylim(ylim)
    if yticks is not None:
        ax.set_yticks(yticks)

    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(-0.25, 1.25)    # default padding around first/last point

    ax.grid(axis='y', alpha=0.1, linewidth=MY_STYLE.linewidth)

    # ── 5. Cosmetic clean‑up ──────────────────────────────────────────
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)

    return ax

# FIGURE 4
@apply_plot_style(MY_STYLE)
def scatter_name_vs_model_tau_with_sem(
    x_labels,
    x_values,
    y_values,
    y_sems,
    ylabel,
    *,
    ax=None,                           # NEW – supply external axes
    xlabel=None,
    title=None,
    ylim=None,
    xlim=None,
    regline=False,
    legend=False,
):
    """
    Scatter plot of model–data alignment (τ) with SEM bars and optional
    regression line.

    Parameters
    ----------
    x_labels : list[str]      – model identifiers
    x_values : list[float]    – x coordinates (e.g. hierarchy score)
    y_values : list[float]    – y values (τ)
    y_sems   : list[float]    – SEMs for y
    ax       : matplotlib.axes.Axes or None
    """

    # ── 0. Axes bookkeeping ───────────────────────────────────────────
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=MY_STYLE.figsize_single, dpi=300)
        created_fig = True
    else:
        fig = ax.figure

    # ── 1. Styling lookup ─────────────────────────────────────────────
    labels  = [MODEL_MAPPINGS.get(lbl, lbl) for lbl in x_labels]
    colors  = [MY_STYLE.model_colors.get(lbl, "gray") for lbl in labels]
    markers = [MY_STYLE.model_markers.get(lbl, "o")  for lbl in labels]

    # ── 2. Baseline & points ──────────────────────────────────────────
    # ax.axhline(0, color="gray", linestyle="--",
    #            linewidth=MY_STYLE.linewidth, alpha=0.8)

    for x, y, yerr, label, color, marker in zip(
            x_values, y_values, y_sems, labels, colors, markers):
        yerr = np.std(y, ddof=1) / np.sqrt(len(y)) 
        y = np.mean(y)
        ax.errorbar(
            x, y, yerr=yerr,
            fmt=marker,
            color=color,
            markersize=MY_STYLE.markersize,
            capsize=MY_STYLE.capsize,
            linewidth=MY_STYLE.linewidth,
            elinewidth=MY_STYLE.linewidth,
            markeredgewidth=MY_STYLE.linewidth,
            label=label[14:-1] if legend else None,
        )

    # ── 3. Optional regression line ──────────────────────────────────
    if regline:

        x_reg = np.array(
            [[el] * len(y_values[i]) for i, el in enumerate(x_values)]
        )
        slope, intercept, r_val, p_val, _ = linregress(x_reg.flatten(), np.array(y_values).flatten())
        x_min = xlim[0] if xlim is not None else min(x_values)
        x_max = xlim[1] if xlim is not None else max(x_values)
        x_fit = np.linspace(x_min, x_max, 200)
        y_fit = slope * x_fit + intercept
        ax.plot(
            x_fit, y_fit,
            color="gray", linestyle="--",
            linewidth=MY_STYLE.linewidth, alpha=0.75,
        )
        significance = next(
            (s for thr, s in {0.001: "*** ", 0.01: "**  ", 0.05: "*   ", 0.1: ".   "}.items() if p_val < thr),
            "n.s."
        )
        ax.text(
            -1, 0.8,
            f"p={p_val:.1e} {significance}",
            ha="left", va="center",
            fontsize=MY_STYLE.rc["font.size"],
            color="gray",
        )
        # ax.text(
        #     -1, 0.8,
        #     f"p = {p_val:.2e}",
        #     ha="left", va="center",
        #     fontsize=MY_STYLE.rc["font.size"],
        #     color="gray",
        # )

    # ── 4. Axis labels, limits, title ─────────────────────────────────
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=MY_STYLE.rc["axes.labelsize"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=MY_STYLE.rc["axes.labelsize"])
    if title:
        ax.set_title(title, fontsize=MY_STYLE.rc["axes.titlesize"])

    # ── 5. Cosmetics ──────────────────────────────────────────────────
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="both", top=False, right=False)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    if legend:
    # unique labels in the same order you plotted them
        uniq = {}
        for lab, col, mar in zip(labels, colors, markers):
            if lab == "hierarchical (anatomy)":
                lab = "hierarchical (V1-LM/RL-AL\n(anatomy))"
            if lab not in uniq:
                # line2D with no line, just the marker
                uniq[lab] = Line2D([], [], marker=mar, linestyle='',
                                markersize=MY_STYLE.markersize,
                                markerfacecolor=col,
                                markeredgecolor=col)

        ax.legend(uniq.values(), [l[14:-1] for l in uniq.keys()],
                frameon=False, bbox_to_anchor=(1.0, 1.01), loc="upper left")

    return ax

# SUPPLEMENTARY 
@apply_plot_style(MY_STYLE)
def build_single_figure(
    barplot_fn,
    row1_fns,
    row2_fns,
    model_ids,
    *,
    single_row_fns=None,          # ← NEW
    figsize=(12, 8),
    wspace=0.35,
    hspace=0.45,
    legend_type="vertical",
):
    # ──────────────────────────────────────────────────────────────────
    # 1. ─ Single-row mode  (early-exit)  → 4 columns × 1 row         ─
    # ──────────────────────────────────────────────────────────────────
    if single_row_fns is not None:
        if len(single_row_fns) != 4:
            raise ValueError("single_row_fns must contain exactly FOUR callables")

        fig = plt.figure(figsize=figsize, constrained_layout=False, dpi=300)
        gs  = fig.add_gridspec(
            nrows=1, ncols=8,                # 4 panels, each spans 2 columns
            width_ratios=[1] * 8,
            wspace=wspace, hspace=0,
        )

        ax_c1 = fig.add_subplot(gs[0, 0:2])
        ax_c2 = fig.add_subplot(gs[0, 2:4])
        ax_c3 = fig.add_subplot(gs[0, 4:6])
        ax_c4 = fig.add_subplot(gs[0, 6:8])

        for fn, ax in zip(single_row_fns, (ax_c1, ax_c2, ax_c3, ax_c4)):
            fn(ax)
            ax.set_box_aspect(1)

        axes = {
            "col1": ax_c1,
            "col2": ax_c2,
            "col3": ax_c3,
            "col4": ax_c4,
        }

        # ── panel letters ────────────────────────────────────────────────
        letters = ["A", "B", "C", "D"]
        for letter, _ax in zip(letters, (ax_c1, ax_c2, ax_c3, ax_c4)):
            pos = _ax.get_position()           # figure coordinates
            fig.text(
                pos.x0 - 0.01,                # a bit left of the panel
                pos.y0 + pos.height + 0.075,   # just above the panel
                letter,
                ha="right", va="top",
                fontweight="bold",
                fontsize=MY_STYLE.rc["axes.titlesize"],
            )

        return fig, axes

def plot_metric_vs_models_tau(
    yvals, xvals, ylabel, selected_area,
    *,                         # keyword-only from here
    ax=None,
    xlabel=None,
    legend=False,
    title=None,
    ylim=None,
    bbox_to_anchor=None,
    text_pos=(-0.5, 0.8),   
    loc="best",
):
    """Grouped error-bar plot of τ vs model hierarchy (subplot-ready)."""

    # ── 0. axes bookkeeping ───────────────────────────────────────────
    if ax is None:
        fig, ax = plt.subplots(figsize=MY_STYLE.figsize_single, dpi=300)
    else:
        fig = ax.figure

    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # ── 1. data points ────────────────────────────────────────────────
    for model, area_data in yvals.items():
        model_label = MODEL_MAPPINGS.get(model, model)
        marker      = MY_STYLE.model_markers.get(model_label, "o")
        for area, values in area_data.items():
            if area != selected_area:
                continue
            x     = xvals[model]
            y     = np.mean(values)
            yerr  = np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
            color = MY_STYLE.model_colors.get(model, "gray")
            if model_label == "hierarchical (anatomy)":
                model_label = "hierarchical (V1-LM/RL-AL\n(anatomy))"
            ax.errorbar(
                x, y, yerr=yerr,
                fmt=marker,
                color=color,
                markersize=MY_STYLE.markersize,
                capsize=MY_STYLE.capsize,
                linewidth=MY_STYLE.linewidth,
                elinewidth=MY_STYLE.linewidth,
                markeredgewidth=MY_STYLE.linewidth,
                # label=model_label[14:-1],
            )

    # ── 2. regression lines per area ──────────────────────────────────
    xs, ys = [], []
    for model, area_data in yvals.items():
        xs.extend([xvals[model]] * len(area_data[selected_area]))
        ys.extend(area_data[selected_area])

    slope, intercept, r, p, _ = linregress(xs, ys)
    x_rng = np.linspace(min(xs), max(xs), 100)
    ax.plot(
        x_rng, slope * x_rng + intercept,
        linestyle="--", linewidth=MY_STYLE.linewidth,
        color='gray', alpha=0.75,
    )
    ax.text(
        text_pos[0], text_pos[1],
        f"p={p:.1e} " + (
            "***" if p < .001 else
            "**"  if p < .01  else
            "*"   if p < .05  else
            "."   if p < .1  else
            "n.s."
        ),
        ha="left", va="center",
        fontsize=MY_STYLE.rc["font.size"],
        color="gray",
    )

    # ── 3. labels, limits, legend ─────────────────────────────────────
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=MY_STYLE.rc["axes.labelsize"])
        ax.xaxis.set_label_coords(1.15, -0.22)
    ax.set_ylabel(ylabel, fontsize=MY_STYLE.rc["axes.labelsize"])
    if title:
        ax.set_title(selected_area, fontsize=MY_STYLE.rc["axes.titlesize"])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="both", top=False, right=False)

    if legend:
        for label in MY_STYLE.model_colors.keys():
            if label not in yvals:
                continue
            color  = MY_STYLE.model_colors.get(label, "gray")
            marker = MY_STYLE.model_markers.get(label, "o")
            if label == "hierarchical (anatomy)":
                label = "hierarchical (V1-LM/RL-AL\n(anatomy))"
            ax.plot(
                [], [], marker=marker, linestyle='',
                markersize=MY_STYLE.markersize,
                color=color,
                label=label[14:-1],
            )
        ax.legend(
            loc=loc,
            bbox_to_anchor=bbox_to_anchor,
            frameon=False,
            fontsize=MY_STYLE.rc["legend.fontsize"],
        )

    return ax

def plot_receptive_field(arf, arf_gaussian_fit, area, figsize, session='4_7', rows=10, cols=10, k=0, savepath=None):
    """
    Plots a receptive field with a Gaussian fit overlay if params are provided.
    
    Args:
    - arf (dict): Dictionary containing receptive field data.
    - arf_gaussian_fit (dict): Dictionary containing Gaussian fit parameters.
    - brain_area (numpy array): Array specifying brain areas for neurons.
    - area (str): Brain area to filter neurons.
    - session (str): Session key to access data in arf and arf_gaussian_fit.
    """
    # Create figure and axis
    fig, ax = plt.subplots(
        rows, 
        cols, 
        figsize=figsize, 
        dpi=300, 
        sharex=True, sharey=True,
        subplot_kw={"aspect": "equal"},          # ← important
        gridspec_kw={"wspace": 0.1, "hspace": 0.1},  # no gaps
        # constrained_layout=False, 
    )
    
    for i in range(rows):
        for j in range(cols):
            idx = i * k + j * (k+1)
            image = arf[session][area][idx]
            params = arf_gaussian_fit[session][area][idx]
            
            ax[i, j].imshow(image, cmap='gray', origin='upper')
            
            if params is not None:
                A, x0, y0, sigma_x, sigma_y, theta = params
                
                # Plot center point
                # ax[i, j].scatter(x0, y0, color='blue', s=3, label="Center", marker='o', alpha=0.85)
                
                # Plot ellipses for 1 sigma and 2 sigma
                for scale in [1, 2]:
                    ellipse = Ellipse(
                        xy=(x0, y0),
                        width=2 * scale * sigma_x,
                        height=2 * scale * sigma_y,
                        angle=np.degrees(theta),
                        edgecolor='red',
                        facecolor='none',
                        alpha=0.75,
                        linewidth=0.5
                    )
                    ax[i, j].add_patch(ellipse)
            
            # Remove axes for clarity
            # option A – turn the whole axis frame off
            ax[i, j].axis("off")

            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
    
    
    # plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.show()
