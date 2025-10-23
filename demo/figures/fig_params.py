from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence, Mapping, Tuple
import importlib

import matplotlib.pyplot as plt

from cycler import cycler  

# 1 ──────────  model appearance  ──────────────────────────────────
MODEL_COLORS  = {
    "baseline (4 layers)"                       : "#000000",
    "loss normalized"                           : "#0072B2",
    "all-layers readout"                        : "#56B4E9",
    "baseline (6 layers)"                       : "#009E73",
    "baseline (8 layers)"                       : "#D4B600",
    "hierarchical (anatomy)"                    : "#D55E00",
    "hierarchical (two streams)"                : "#CC79A7",
    "hierarchical (V1-AL-LM/RL)"                : "#CC79A7",
    "hierarchical (AL-V1-LM/RL)"                : "#9AD0F3",
    "hierarchical (AL-LM/RL-V1)"                : "#8DD3C7",
    "hierarchical (LM/RL-AL-V1)"                : "#FDB462",
    "hierarchical (LM/RL-V1-AL)"                : "#B3DE69",
    "hierarchical (9 layer)"                    : "#1C852F",
    "Wang (2025)"                               : "#474646",
}

MODEL_MAPPINGS = {
    "benchmark_4_all_seed42"                : "baseline (4 layers)",
    "benchmark_4_all_seed228"               : "baseline (4 layers)",
    "benchmark_4_all_seed351"               : "baseline (4 layers)",

    "benchmark_6_all_seed42"                : "baseline (6 layers)",
    "benchmark_6_all_seed228"               : "baseline (6 layers)",
    "benchmark_6_all_seed351"               : "baseline (6 layers)",

    "benchmark_8_all_seed42"                : "baseline (8 layers)",
    "benchmark_8_all_seed228"               : "baseline (8 layers)",
    "benchmark_8_all_seed351"               : "baseline (8 layers)",

    "all_layer_4_all_seed42"                : "all-layers readout",
    "all_layer_4_all_seed228"               : "all-layers readout",
    "all_layer_4_all_seed351"               : "all-layers readout", 

    "normalized_4_all_seed42"               : "loss normalized",
    "normalized_4_all_seed228"              : "loss normalized",    
    "normalized_4_all_seed351"              : "loss normalized",

    "8layer_678_V1_LMRL_AL_all_seed42"      : "hierarchical (anatomy)",
    "8layer_678_V1_LMRL_AL_all_seed228"     : "hierarchical (anatomy)",
    "8layer_678_V1_LMRL_AL_all_seed351"     : "hierarchical (anatomy)",

    "8layer_678_V1_AL_LMRL_all_seed42"      : "hierarchical (V1-AL-LM/RL)",
    "8layer_678_V1_AL_LMRL_all_seed228"     : "hierarchical (V1-AL-LM/RL)",
    "8layer_678_V1_AL_LMRL_all_seed351"     : "hierarchical (V1-AL-LM/RL)",

    "8layer_678_AL_V1_LMRL_all_seed42"      : "hierarchical (AL-V1-LM/RL)",
    "8layer_678_AL_V1_LMRL_all_seed228"     : "hierarchical (AL-V1-LM/RL)",
    "8layer_678_AL_V1_LMRL_all_seed351"     : "hierarchical (AL-V1-LM/RL)",

    "8layer_678_AL_LMRL_V1_all_seed42"      : "hierarchical (AL-LM/RL-V1)",
    "8layer_678_AL_LMRL_V1_all_seed228"     : "hierarchical (AL-LM/RL-V1)",
    "8layer_678_AL_LMRL_V1_all_seed351"     : "hierarchical (AL-LM/RL-V1)",

    "8layer_678_LMRL_AL_V1_all_seed42"      : "hierarchical (LM/RL-AL-V1)",
    "8layer_678_LMRL_AL_V1_all_seed228"     : "hierarchical (LM/RL-AL-V1)",
    "8layer_678_LMRL_AL_V1_all_seed351"     : "hierarchical (LM/RL-AL-V1)",

    "8layer_678_LMRL_V1_AL_all_seed42"      : "hierarchical (LM/RL-V1-AL)",
    "8layer_678_LMRL_V1_AL_all_seed228"     : "hierarchical (LM/RL-V1-AL)",
    "8layer_678_LMRL_V1_AL_all_seed351"     : "hierarchical (LM/RL-V1-AL)",
    
    "two_streams_67_678_V1LM_V1RLAL_all_seed42"  : "hierarchical (two streams)",
    "two_streams_67_678_V1LM_V1RLAL_all_seed228" : "hierarchical (two streams)",
    "two_streams_67_678_V1LM_V1RLAL_all_seed351" : "hierarchical (two streams)",
    
    "9layer_6789_V1_LM_RL_AL_all_seed42"    : "hierarchical (9 layer)",
    "9layer_6789_V1_LM_RL_AL_all_seed228"   : "hierarchical (9 layer)",
    "9layer_6789_V1_LM_RL_AL_all_seed351"   : "hierarchical (9 layer)",

    "towards_wang"                          : "Wang (2025)",    
}

MODEL_MAPPINGS_OLD = {
    "benchmark_4_subset6_seed42"                : "baseline (4 layers)",
    "benchmark_4_subset6_seed228"               : "baseline (4 layers)",
    "benchmark_4_subset6_seed351"               : "baseline (4 layers)",
    
    "benchmark_6_subset6_seed42"                : "baseline (6 layers)",
    "benchmark_6_subset6_seed228"               : "baseline (6 layers)",
    "benchmark_6_subset6_seed351"               : "baseline (6 layers)",
    
    "benchmark_8_subset6_seed42"                : "baseline (8 layers)",
    "benchmark_8_subset6_seed227"               : "baseline (8 layers)",
    "benchmark_8_subset6_seed351"               : "baseline (8 layers)",
    
    "all_layer_4_subset6_seed42"                : "all-layers readout",
    "all_layer_4_subset6_seed228"               : "all-layers readout",
    "all_layer_4_subset6_seed351"               : "all-layers readout",
    
    "normalized_4_subset6_seed42"               : "loss normalized",
    "normalized_4_subset6_seed228"              : "loss normalized",
    "normalized_4_subset6_seed351"              : "loss normalized",
    
    "8layer_678_V1_LMRL_AL_subset6_seed42"      : "hierarchical (anatomy)",
    "8layer_678_V1_LMRL_AL_subset6_seed228"     : "hierarchical (anatomy)",
    "8layer_678_V1_LMRL_AL_subset6_seed351"     : "hierarchical (anatomy)",
    
    "8layer_678_V1_AL_LMRL_subset6_seed42"      : "hierarchical (V1-AL-LM/RL)",
    "8layer_678_V1_AL_LMRL_subset6_seed228"     : "hierarchical (V1-AL-LM/RL)",
    "8layer_678_V1_AL_LMRL_subset6_seed351"     : "hierarchical (V1-AL-LM/RL)",
    
    "8layer_678_AL_V1_LMRL_subset6_seed42"      : "hierarchical (AL-V1-LM/RL)",
    "8layer_678_AL_V1_LMRL_subset6_seed228"     : "hierarchical (AL-V1-LM/RL)",
    "8layer_678_AL_V1_LMRL_subset6_seed351"     : "hierarchical (AL-V1-LM/RL)",
    
    "8layer_678_AL_LMRL_V1_subset6_seed42"      : "hierarchical (AL-LM/RL-V1)",
    "8layer_678_AL_LMRL_V1_subset6_seed228"     : "hierarchical (AL-LM/RL-V1)",
    "8layer_678_AL_LMRL_V1_subset6_seed351"     : "hierarchical (AL-LM/RL-V1)",
    
    "8layer_678_LMRL_AL_V1_subset6_seed42"      : "hierarchical (LM/RL-AL-V1)",
    "8layer_678_LMRL_AL_V1_subset6_seed228"     : "hierarchical (LM/RL-AL-V1)",
    "8layer_678_LMRL_AL_V1_subset6_seed351"     : "hierarchical (LM/RL-AL-V1)",
    
    "8layer_678_LMRL_V1_AL_subset6_seed42"      : "hierarchical (LM/RL-V1-AL)",
    "8layer_678_LMRL_V1_AL_subset6_seed228"     : "hierarchical (LM/RL-V1-AL)",
    "8layer_678_LMRL_V1_AL_subset6_seed351"     : "hierarchical (LM/RL-V1-AL)",

    "9layer_67_678_V1_LM_RL_AL_subset6_seed42"  : "hierarchical (two streams)",
}

MODEL_MARKERS = {
    "baseline (4 layers)"                       : "o",
    "loss normalized"                           : "s",
    "all-layers readout"                        : "D",
    "baseline (6 layers)"                       : "^",
    "baseline (8 layers)"                       : "v",
    "hierarchical (anatomy)"                    : "X",
    "hierarchical (V1-AL-LM/RL)"                : "p",
    "hierarchical (AL-V1-LM/RL)"                : ">",
    "hierarchical (AL-LM/RL-V1)"                : "<",
    "hierarchical (LM/RL-AL-V1)"                : "*",
    "hierarchical (LM/RL-V1-AL)"                : "h",
    "Wang (2025)"                               : "x",
    "hierarchical (two streams)"                : "P",
    "hierarchical (9 layer)"                    : "8",
}

MODEL_RANKINGS = {
    'hierarchical (anatomy)'                    : {'V1': 1, 'LM': 2, 'RL': 2, 'AL': 3},
    'hierarchical (two streams)'                : {'V1': 1, 'LM': 2, 'RL': 2, 'AL': 3},
    'hierarchical (9 layer)'                    : {'V1': 1, 'LM': 2, 'RL': 3, 'AL': 4},
    'hierarchical (V1-AL-LM/RL)'                : {'V1': 1, 'AL': 2, 'LM': 3, 'RL': 3},
    'hierarchical (LM/RL-AL-V1)'                : {'LM': 1, 'RL': 1, 'AL': 2, 'V1': 3},
    'hierarchical (LM/RL-V1-AL)'                : {'LM': 1, 'RL': 1, 'V1': 2, 'AL': 3},
    'hierarchical (AL-LM/RL-V1)'                : {'AL': 1, 'LM': 2, 'RL': 2, 'V1': 3},
    'hierarchical (AL-V1-LM/RL)'                : {'AL': 1, 'V1': 2, 'LM': 3, 'RL': 3},
}

# 2 ──────────  area colours  ──────────────────────────────────────
AREA_COLORS = {
    "AL"                                        : "#0072B2", 
    "LM"                                        : "#E69F00", 
    "RL"                                        : "#009E73", 
    "V1"                                        : "#D55E00"
}

AREA_HIERARCHY_SCORES = {
    "AL"                                        : 0.1524064171122994,
    "LM"                                        : -0.0935828877005348,
    "RL"                                        : -0.05614973262032086,
    "V1"                                        : -0.35383244206773623
}

AREA_MARRKERS = {
    "AL"                                        : "o",
    "LM"                                        : "s",
    "RL"                                        : "^",
    "V1"                                        : "D"
}

# 3 ──────────  plot style  ───────────────────────────────────────

FONT_SIZE = 6
LINE_WIDTH = 1.1
AXIS_LINE_WIDTH = 1.
LEGEND_FONT_SIZE = 4.5
TICK_LABEL_SIZE = 7
AXIS_LABEL_SIZE = 6.5
AXIS_TITLE_SIZE = 9
MARKERSIZE = 4
CAPSIZE = 1.5
FIGSIZE_SINGLE = (5.5, 3.5)
FIGSIZE_PER_SUBPLOT = (3.5, 2.5)

# 4 ──────────  plot style  ───────────────────────────────────────

@dataclass
class PlotStyle:
    styles: Sequence[str] = field(default_factory=list)
    rc: Mapping[str, object] = field(default_factory=dict)
    figsize_single: Tuple[float, float] = FIGSIZE_SINGLE
    figsize_per_subplot: Tuple[float, float] = FIGSIZE_PER_SUBPLOT
    capsize: float = CAPSIZE
    markersize: float = MARKERSIZE
    linewidth: float = LINE_WIDTH
    area_colors: Mapping[str, str] = field(default_factory=dict)
    area_hierarchy_scores: Mapping[str, float] = field(default_factory=dict)
    model_colors: Mapping[str, str] = field(default_factory=dict)
    model_markers: Mapping[str, str] = field(default_factory=dict)
    def _ensure_styles_available(self):
        missing = [s for s in self.styles if s not in plt.style.library]
        if missing:
            if any(m in {"science", "ieee", "nature", "grid"} for m in missing):
                if not importlib.util.find_spec("scienceplots"):
                    raise ImportError("SciencePlots style requested but package not installed. Run `pip install SciencePlots` or remove the style name.")
                import scienceplots # noqa: F401
            still_missing = [s for s in missing if s not in plt.style.library]
            if still_missing: raise ValueError(f"Unknown Matplotlib style(s): {still_missing}")

MY_STYLE = PlotStyle(
    styles=["nature", "no-latex"],
    rc={
        "font.size": FONT_SIZE,
        "axes.labelsize": AXIS_LABEL_SIZE,
        "axes.linewidth": LINE_WIDTH,
        "xtick.labelsize": TICK_LABEL_SIZE,
        "ytick.labelsize": TICK_LABEL_SIZE,
        "axes.titlesize": AXIS_TITLE_SIZE,
        "legend.fontsize": LEGEND_FONT_SIZE,
        "axes.prop_cycle": cycler(color=list(AREA_COLORS.values()) + list(MODEL_COLORS.values())),
        "axes.linewidth": LINE_WIDTH,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.width": 0,      
        "ytick.minor.width": 1,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.minor.size": 0,
        "ytick.minor.size": 2.5,

    },
    area_colors=AREA_COLORS,
    area_hierarchy_scores=AREA_HIERARCHY_SCORES,
    model_colors=MODEL_COLORS,
    model_markers=MODEL_MARKERS,
    markersize=MARKERSIZE,
    capsize=CAPSIZE,
)

