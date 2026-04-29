from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "initial_analysis" / "tables"
FIG_DIR = ROOT / "outputs" / "manuscript" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


GROUP_MAP = {
    "平和质": "Balanced",
    "痰湿质": "Phlegm-dampness",
    "阳虚质": "Yang-deficiency",
    "气虚质": "Qi-deficiency",
    "阴虚质": "Yin-deficiency",
    "其他偏颇质": "Other biased",
}

TARGET_MAP = {
    "痰湿质": "Phlegm-dampness",
    "阳虚质": "Yang-deficiency",
    "气虚质": "Qi-deficiency",
    "阴虚质": "Yin-deficiency",
}

CONSTITUTION_ORDER = [
    "Balanced",
    "Phlegm-dampness",
    "Yang-deficiency",
    "Qi-deficiency",
    "Yin-deficiency",
    "Other biased",
]

CONSTITUTION_TICKS = {
    "Balanced": "Balanced",
    "Phlegm-dampness": "Phlegm-\ndampness",
    "Yang-deficiency": "Yang-\ndeficiency",
    "Qi-deficiency": "Qi-\ndeficiency",
    "Yin-deficiency": "Yin-\ndeficiency",
    "Other biased": "Other\nbiased",
}

OUTCOME_TICKS = {
    "Recorded cerebrovascular disease": "Cerebrovascular\ndisease",
    "Recorded heart disease": "Heart\ndisease",
    "Recorded eye disease": "Eye\ndisease",
    "Recorded neurological disease": "Neurological\ndisease",
    "Abnormal ECG": "Abnormal\nECG",
    "Abnormal abdominal ultrasound": "Abdominal\nultrasound",
    "Urine protein trace/positive": "Proteinuria",
    "Urine glucose trace/positive": "Glycosuria",
    "Urine occult blood trace/positive": "Hematuria",
    "Liver enzyme elevation": "Liver enzyme\nelevation",
    "Anemia": "Anemia",
    "Kidney impairment or proteinuria": "Kidney marker",
    "Diabetes-related marker": "Diabetes\nmarker",
    "Cardiometabolic risk cluster": "Cardiometabolic\ncluster",
    "High BP (>=140/90)": "High BP\n(>=140/90)",
    "FPG >=6.1 mmol/L": "FPG >=6.1\nmmol/L",
    "FPG >=7.0 mmol/L": "FPG >=7.0\nmmol/L",
    "High TG": "High TG",
    "Low HDL-C": "Low HDL-C",
    "High LDL-C": "High LDL-C",
    "Any dyslipidemia": "Any\ndyslipidemia",
    "eGFR <60": "eGFR <60",
}

PREFERRED_DISEASES = [
    "Recorded cerebrovascular disease",
    "Recorded heart disease",
    "Abnormal ECG",
    "Abnormal abdominal ultrasound",
    "Urine protein trace/positive",
    "Urine glucose trace/positive",
    "Liver enzyme elevation",
    "Anemia",
    "Kidney impairment or proteinuria",
    "Diabetes-related marker",
    "Cardiometabolic risk cluster",
]

PREFERRED_LONGITUDINAL = [
    ("High TG", "bio"),
    ("Low HDL-C", "bio"),
    ("FPG >=6.1 mmol/L", "bio"),
    ("FPG >=7.0 mmol/L", "bio"),
    ("High BP (>=140/90)", "bio"),
    ("Abnormal abdominal ultrasound", "disease"),
    ("Cardiometabolic risk cluster", "disease"),
    ("Diabetes-related marker", "disease"),
    ("Urine protein trace/positive", "disease"),
    ("Kidney impairment or proteinuria", "disease"),
    ("Recorded cerebrovascular disease", "disease"),
    ("Anemia", "disease"),
]


TEXT = "#222222"
GRID = "#D6D6D6"
BLUE = "#2F6FAE"
RED = "#B74346"
GREEN = "#4F8F72"
GOLD = "#C9922D"
PURPLE = "#7569A7"
GRAY = "#777777"


def load_table(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / name)


def set_theme() -> None:
    sns.set_theme(style="white", context="paper")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6.4,
            "axes.titlesize": 7.4,
            "axes.labelsize": 6.4,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "legend.fontsize": 5.8,
            "figure.titlesize": 7.4,
            "axes.linewidth": 0.55,
            "xtick.major.width": 0.45,
            "ytick.major.width": 0.45,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "lines.linewidth": 0.8,
            "patch.linewidth": 0.45,
            "savefig.dpi": 600,
            "figure.dpi": 160,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def cmap_percent() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "percent_scale",
        ["#F7FBF7", "#C7E2D4", "#7CBBA5", "#2F6FAE"],
        N=256,
    )


def cmap_transition() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "transition_scale",
        ["#F8FBFD", "#CFE3EE", "#7BAFCB", "#1F5F99"],
        N=256,
    )


def cmap_diverging() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "or_diverging",
        ["#2F6FAE", "#F7F7F7", "#B74346"],
        N=256,
    )


def save_figure(fig: plt.Figure, name: str) -> None:
    for ext in ["pdf", "svg", "png", "tiff"]:
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.025, "facecolor": "white"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        if ext == "tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(FIG_DIR / f"{name}.{ext}", **kwargs)
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.03) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=8.8,
        fontweight="bold",
        va="bottom",
        ha="left",
        color=TEXT,
    )


def clean_axes(ax: plt.Axes, left: bool = True, bottom: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if not left:
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    if not bottom:
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", length=0)


def luminance_color(cmap, norm, value: float) -> str:
    if pd.isna(value):
        return TEXT
    rgba = cmap(norm(value))
    lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    return "white" if lum < 0.52 else TEXT


def annotated_matrix(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    cmap,
    norm=None,
    vmin: float | None = None,
    vmax: float | None = None,
    fmt: str = "{:.1f}",
    annotations: pd.DataFrame | None = None,
    text_size: float = 5.4,
    grid_lw: float = 0.45,
) -> mpl.image.AxesImage:
    values = data.astype(float).to_numpy()
    local_cmap = cmap.copy()
    local_cmap.set_bad("#FFFFFF")

    if norm is None:
        norm = Normalize(
            vmin=np.nanmin(values) if vmin is None else vmin,
            vmax=np.nanmax(values) if vmax is None else vmax,
        )
    im = ax.imshow(
        np.ma.masked_invalid(values),
        cmap=local_cmap,
        norm=norm,
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(list(data.columns))
    ax.set_yticklabels(list(data.index))
    ax.set_xticks(np.arange(-0.5, data.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, data.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=grid_lw)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = values[i, j]
            if pd.isna(value):
                label = ""
            elif annotations is not None:
                label = str(annotations.iloc[i, j])
            else:
                label = fmt.format(value)
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=text_size,
                color=luminance_color(local_cmap, norm, value),
            )
    return im


def or_annotation(values: pd.DataFrame, qvalues: pd.DataFrame | None) -> pd.DataFrame:
    annot = values.copy().astype("object")
    for i in values.index:
        for j in values.columns:
            val = values.loc[i, j]
            if pd.isna(val):
                annot.loc[i, j] = ""
                continue
            star = ""
            if qvalues is not None and not pd.isna(qvalues.loc[i, j]) and qvalues.loc[i, j] < 0.05:
                star = "*"
            annot.loc[i, j] = f"{math.exp(val):.2f}{star}"
    return annot


def add_colorbar(fig: plt.Figure, im, cax: plt.Axes, label: str) -> None:
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.outline.set_linewidth(0.4)
    cbar.ax.tick_params(labelsize=5.4, width=0.4, length=2)
    cbar.set_label(label, fontsize=5.6, labelpad=1.5)


def figure_1_cohort_overview() -> None:
    overview = json.loads((ROOT / "outputs" / "initial_analysis" / "data_overview.json").read_text(encoding="utf-8"))
    dist = load_table("constitution_group_distribution.csv").dropna(subset=["constitution_group"]).copy()
    dist = dist[dist["constitution_group"].isin(GROUP_MAP)]
    dist["group_en"] = dist["constitution_group"].map(GROUP_MAP)
    dist["group_en"] = pd.Categorical(dist["group_en"], categories=CONSTITUTION_ORDER, ordered=True)
    dist = dist.sort_values("group_en")

    fig = plt.figure(figsize=(7.2, 3.0))
    gs = gridspec.GridSpec(
        1,
        2,
        figure=fig,
        width_ratios=[1.0, 1.25],
        wspace=0.38,
        left=0.06,
        right=0.985,
        top=0.86,
        bottom=0.17,
    )
    ax_flow = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])

    ax_flow.set_axis_off()
    milestones = [
        ("Records", int(overview["n_rows"])),
        ("Participants", int(overview["n_unique_persons"])),
        ("Valid constitution\nlabels", int(overview["n_constitution_nonmissing"])),
        ("Annual adjacent\npairs", 32648),
    ]
    y_positions = np.linspace(0.82, 0.18, len(milestones))
    ax_flow.plot([0.16, 0.16], [y_positions[-1], y_positions[0]], color=GRID, lw=1.2, zorder=1)
    for idx, ((label, value), y) in enumerate(zip(milestones, y_positions)):
        color = [BLUE, GREEN, GOLD, PURPLE][idx]
        ax_flow.scatter([0.16], [y], s=34, color=color, zorder=3)
        ax_flow.text(0.24, y + 0.035, f"{value:,}", ha="left", va="center", fontsize=10.2, fontweight="bold", color=TEXT)
        ax_flow.text(0.24, y - 0.045, label, ha="left", va="center", fontsize=6.2, color="#555555", linespacing=1.05)
    ax_flow.text(0.24, 0.02, "Routine health examinations, 2017-2026", fontsize=5.9, color="#555555")
    ax_flow.set_xlim(0, 1)
    ax_flow.set_ylim(0, 1)
    ax_flow.set_title("Cohort structure", loc="left", pad=2, fontweight="bold")
    panel_label(ax_flow, "A", x=-0.16, y=1.02)

    colors = [BLUE, RED, GREEN, GOLD, PURPLE, GRAY]
    y = np.arange(len(dist))
    ax_bar.barh(y, dist["percent"], color=colors, height=0.58)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels([CONSTITUTION_TICKS[str(x)].replace("\n", " ") for x in dist["group_en"]])
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Share of valid records, %")
    ax_bar.set_xlim(0, 65)
    ax_bar.set_xticks([0, 20, 40, 60])
    ax_bar.grid(axis="x", color="#E7E7E7", lw=0.45)
    ax_bar.set_axisbelow(True)
    for yi, (_, row) in zip(y, dist.iterrows()):
        ax_bar.text(
            min(row["percent"] + 1.1, 64.5),
            yi,
            f"{row['percent']:.1f}%  ({int(row['records']):,})",
            va="center",
            ha="left",
            fontsize=5.8,
            color=TEXT,
        )
    clean_axes(ax_bar)
    ax_bar.set_title("Primary constitution distribution", loc="left", pad=2, fontweight="bold")
    panel_label(ax_bar, "B", x=-0.14, y=1.02)

    save_figure(fig, "figure_1_cohort_overview")


def figure_2_bidirectional_disease_map() -> None:
    d2c = load_table("disease_to_constitution_distribution_baseline.csv")
    c2d = load_table("constitution_to_disease_associations_baseline.csv")

    diseases = [d for d in PREFERRED_DISEASES if d in set(d2c["disease_or_marker"])]
    rows = [OUTCOME_TICKS[d] for d in diseases]
    case_groups = ["Balanced", "Phlegm-dampness", "Yang-deficiency", "Qi-deficiency", "Yin-deficiency"]
    case_cols = [CONSTITUTION_TICKS[g] for g in case_groups]
    d2c_heat = pd.DataFrame(index=rows, columns=case_cols, dtype=float)
    for d in diseases:
        row = d2c.loc[d2c["disease_or_marker"].eq(d)].iloc[0]
        for zh, en in GROUP_MAP.items():
            if en in case_groups:
                d2c_heat.loc[OUTCOME_TICKS[d], CONSTITUTION_TICKS[en]] = row.get(f"{zh}_percent_among_cases", np.nan)

    target_groups = ["Phlegm-dampness", "Yang-deficiency", "Qi-deficiency", "Yin-deficiency"]
    target_cols = [CONSTITUTION_TICKS[g] for g in target_groups]
    c2d_sub = c2d[c2d["disease_or_marker"].isin(diseases)].copy()
    c2d_sub["target_en"] = c2d_sub["target_constitution"].map(TARGET_MAP)
    c2d_sub["target_tick"] = c2d_sub["target_en"].map(CONSTITUTION_TICKS)
    c2d_sub["outcome_tick"] = c2d_sub["disease_or_marker"].map(OUTCOME_TICKS)
    or_heat = (
        c2d_sub.pivot_table(index="outcome_tick", columns="target_tick", values="estimate", aggfunc="first")
        .reindex(index=rows, columns=target_cols)
    )
    q_heat = (
        c2d_sub.pivot_table(index="outcome_tick", columns="target_tick", values="q_value_all", aggfunc="first")
        .reindex(index=rows, columns=target_cols)
    )

    fig = plt.figure(figsize=(7.4, 4.75))
    gs = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[1.18, 1.0],
        height_ratios=[1.0, 0.052],
        wspace=0.32,
        hspace=0.16,
        left=0.17,
        right=0.985,
        top=0.88,
        bottom=0.14,
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    cax_blank = fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[1, 1])
    cax_blank.axis("off")

    annotated_matrix(
        ax1,
        d2c_heat,
        cmap=cmap_percent(),
        norm=Normalize(vmin=0, vmax=80),
        fmt="{:.1f}",
        text_size=5.2,
    )
    ax1.set_title("Disease/marker -> constitution\ncase composition, %", loc="left", pad=3, fontweight="bold")
    ax1.tick_params(axis="x", pad=2)
    panel_label(ax1, "A", x=-0.16, y=1.03)

    im2 = annotated_matrix(
        ax2,
        or_heat,
        cmap=cmap_diverging(),
        norm=TwoSlopeNorm(vmin=-1.15, vcenter=0, vmax=1.15),
        annotations=or_annotation(or_heat, q_heat),
        text_size=5.2,
    )
    ax2.set_title("Constitution -> disease/marker\nage- and sex-adjusted OR", loc="left", pad=3, fontweight="bold")
    ax2.tick_params(axis="x", pad=2)
    panel_label(ax2, "B", x=-0.18, y=1.03)
    add_colorbar(fig, im2, cax, "log(OR)")

    fig.text(0.17, 0.055, "Cell labels show percentages or ORs; *FDR q<0.05.", fontsize=5.7, color="#555555")
    save_figure(fig, "figure_2_bidirectional_disease_map")


def figure_3_longitudinal_risk_map() -> None:
    lag_bio = load_table("lagged_constitution_future_risk_models.csv")
    lag_dis = load_table("lagged_constitution_future_disease_risk_models.csv")
    rows = []
    for outcome, kind in PREFERRED_LONGITUDINAL:
        df = lag_bio if kind == "bio" else lag_dis.rename(columns={"disease_or_marker": "outcome"})
        rows.append(df[df["outcome"].eq(outcome)].copy())
    lag = pd.concat(rows, ignore_index=True)
    lag["target_en"] = lag["target_constitution"].map(TARGET_MAP)
    lag["target_tick"] = lag["target_en"].map(CONSTITUTION_TICKS)
    lag["outcome_tick"] = lag["outcome"].map(OUTCOME_TICKS)

    row_order = [OUTCOME_TICKS[o] for o, _ in PREFERRED_LONGITUDINAL]
    col_order = [CONSTITUTION_TICKS[g] for g in ["Phlegm-dampness", "Yang-deficiency", "Qi-deficiency", "Yin-deficiency"]]
    heat = (
        lag.pivot_table(index="outcome_tick", columns="target_tick", values="estimate", aggfunc="first")
        .reindex(index=row_order, columns=col_order)
    )
    q = (
        lag.pivot_table(index="outcome_tick", columns="target_tick", values="q_value_all", aggfunc="first")
        .reindex(index=row_order, columns=col_order)
    )

    fig = plt.figure(figsize=(5.75, 4.55))
    gs = gridspec.GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[1.0, 0.055],
        hspace=0.18,
        left=0.29,
        right=0.965,
        top=0.89,
        bottom=0.15,
    )
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[1, 0])
    im = annotated_matrix(
        ax,
        heat,
        cmap=cmap_diverging(),
        norm=TwoSlopeNorm(vmin=-0.8, vcenter=0, vmax=0.8),
        annotations=or_annotation(heat, q),
        text_size=5.35,
    )
    ax.set_title(
        "Current constitution and next-visit disease/risk markers",
        loc="left",
        pad=3,
        fontweight="bold",
    )
    ax.set_xlabel("")
    panel_label(ax, "A", x=-0.22, y=1.03)
    add_colorbar(fig, im, cax, "log(OR), adjusted for current marker status")
    fig.text(0.29, 0.055, "Cell labels show ORs; *FDR q<0.05.", fontsize=5.7, color="#555555")
    save_figure(fig, "figure_3_longitudinal_risk_map")


def figure_4_transition_dynamics() -> None:
    trans_pct = pd.read_csv(TABLES / "adjacent_constitution_transition_matrix_percent.csv", index_col=0)
    trans_pct.index = [GROUP_MAP.get(x, x) for x in trans_pct.index]
    trans_pct.columns = [GROUP_MAP.get(x, x) for x in trans_pct.columns]
    trans_pct = trans_pct.reindex(index=CONSTITUTION_ORDER, columns=CONSTITUTION_ORDER)
    trans_pct.index = [CONSTITUTION_TICKS[x] for x in CONSTITUTION_ORDER]
    trans_pct.columns = [CONSTITUTION_TICKS[x] for x in CONSTITUTION_ORDER]

    predictors = load_table("transition_predictor_models.csv")
    pred_names = [
        "BMI",
        "Waist circumference",
        "Age",
        "Triglycerides",
        "HDL-C",
        "Fasting glucose",
        "ALT",
        "Creatinine",
        "eGFR",
    ]
    pred = predictors[predictors["predictor"].isin(pred_names)].copy()
    pred["task_en"] = pred["transition_task"].replace(
        {
            "Balanced to biased": "Balanced -> biased",
            "Phlegm-dampness to balanced": "Phlegm-dampness -> balanced",
        }
    )
    selected = []
    for task in ["Balanced -> biased", "Phlegm-dampness -> balanced"]:
        sub = pred[pred["task_en"].eq(task)].copy()
        sub["abs_log_or"] = sub["or_per_sd"].map(lambda x: abs(math.log(float(x))))
        keep_n = 6 if task == "Balanced -> biased" else 5
        selected.append(sub.sort_values("abs_log_or", ascending=False).head(keep_n))
    forest = pd.concat(selected, ignore_index=True)

    fig = plt.figure(figsize=(7.35, 4.25))
    gs = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[1.0, 1.08],
        height_ratios=[1.0, 0.055],
        wspace=0.42,
        hspace=0.15,
        left=0.13,
        right=0.98,
        top=0.88,
        bottom=0.15,
    )
    ax1 = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[:, 1])

    im = annotated_matrix(
        ax1,
        trans_pct,
        cmap=cmap_transition(),
        norm=Normalize(vmin=0, vmax=72),
        fmt="{:.1f}",
        text_size=5.15,
    )
    ax1.set_title("Observed state transitions, %", loc="left", pad=3, fontweight="bold")
    ax1.set_xlabel("Next observed state", labelpad=4)
    ax1.set_ylabel("Current state", labelpad=4)
    panel_label(ax1, "A", x=-0.16, y=1.03)
    add_colorbar(fig, im, cax, "Transition, %")

    task_palette = {"Balanced -> biased": RED, "Phlegm-dampness -> balanced": BLUE}
    y_positions: list[float] = []
    y_labels: list[str] = []
    xvals: list[float] = []
    xlow: list[float] = []
    xhigh: list[float] = []
    colors: list[str] = []
    task_ranges: dict[str, tuple[float, float]] = {}
    y = 0.0
    for task in ["Balanced -> biased", "Phlegm-dampness -> balanced"]:
        sub = forest[forest["task_en"].eq(task)].sort_values("abs_log_or", ascending=False)
        start = y
        for _, row in sub.iterrows():
            y_positions.append(y)
            y_labels.append(row["predictor"])
            xvals.append(float(row["or_per_sd"]))
            xlow.append(float(row["or_ci_low"]))
            xhigh.append(float(row["or_ci_high"]))
            colors.append(task_palette[task])
            y += 1.0
        task_ranges[task] = (start, y - 1.0)
        y += 1.1

    ax2.errorbar(
        xvals,
        y_positions,
        xerr=[np.array(xvals) - np.array(xlow), np.array(xhigh) - np.array(xvals)],
        fmt="none",
        ecolor="#555555",
        elinewidth=0.75,
        capsize=2,
        zorder=2,
    )
    ax2.scatter(xvals, y_positions, s=22, color=colors, edgecolor="white", linewidth=0.35, zorder=3)
    ax2.axvline(1, color="#333333", lw=0.7, ls=(0, (3, 2)))
    ax2.set_xscale("log")
    ax2.set_xlim(0.20, 2.35)
    ax2.set_xticks([0.25, 0.5, 1.0, 2.0])
    ax2.set_xticklabels(["0.25", "0.5", "1", "2"])
    ax2.set_xlabel("OR per SD increment", labelpad=4)
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels(y_labels)
    ax2.invert_yaxis()
    ax2.set_title("Predictors of constitution-state transitions", loc="left", pad=3, fontweight="bold")
    clean_axes(ax2)
    for task, (start, end) in task_ranges.items():
        ax2.text(
            0.205,
            start - 0.45,
            task,
            ha="left",
            va="bottom",
            fontsize=6.2,
            fontweight="bold",
            color=task_palette[task],
        )
        ax2.axhspan(start - 0.5, end + 0.5, color=task_palette[task], alpha=0.045, lw=0)
    panel_label(ax2, "B", x=-0.16, y=1.03)

    save_figure(fig, "figure_4_transition_dynamics")


def clean_feature_name(name: str) -> str:
    replacements = {
        "diet = 1,": "Diet category",
        "diet = 1": "Diet category",
        "self = health = 2.0": "Self-rated health",
        "self = health = 1.0": "Self-rated health",
        "gender = 女": "Female sex",
        "other = system = disease = 1.0": "Other system disease",
        "other = system = disease = 2.0": "Other system disease",
        "cerebrovascular = 2": "Cerebrovascular history",
        "exercise = 1.0": "Exercise category",
    }
    return replacements.get(str(name), str(name))


def figure_5_prediction_and_explainability() -> None:
    metrics = load_table("temporal_validation_model_metrics.csv")
    sensitivity = load_table("temporal_validation_model_metrics_sensitivity_no_anthro.csv")
    shap_imp = load_table("shap_top_features_phlegm_dampness_vs_others.csv").head(12).copy()
    shap_imp["feature_clean"] = shap_imp["feature_readable"].map(clean_feature_name)

    perf_rows = []
    task_map = {
        "Biased vs balanced constitution": "Biased vs\nbalanced",
        "Phlegm-dampness vs others": "Phlegm-dampness\nvs others",
    }
    for _, row in metrics.iterrows():
        perf_rows.append(
            {
                "task": task_map.get(row["task"], row["task"]),
                "model": row["model"].replace("Logistic regression", "Logistic"),
                "auc": float(row["auc"]),
            }
        )
    for _, row in sensitivity[sensitivity["model"].eq("XGBoost")].iterrows():
        perf_rows.append(
            {
                "task": task_map.get(row["task"], row["task"]),
                "model": "XGBoost, no anthrop.",
                "auc": float(row["auc"]),
            }
        )
    perf = pd.DataFrame(perf_rows)
    model_order = ["Logistic", "XGBoost", "XGBoost, no anthrop."]
    perf["model"] = pd.Categorical(perf["model"], categories=model_order, ordered=True)
    perf = perf.sort_values(["task", "model"])

    fig = plt.figure(figsize=(7.3, 3.75))
    gs = gridspec.GridSpec(
        1,
        2,
        figure=fig,
        width_ratios=[1.0, 1.1],
        wspace=0.42,
        left=0.085,
        right=0.985,
        top=0.85,
        bottom=0.18,
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    task_order = ["Biased vs\nbalanced", "Phlegm-dampness\nvs others"]
    task_y = {task: i for i, task in enumerate(task_order)}
    offsets = {"Logistic": -0.16, "XGBoost": 0.0, "XGBoost, no anthrop.": 0.16}
    model_colors = {"Logistic": BLUE, "XGBoost": RED, "XGBoost, no anthrop.": GRAY}
    markers = {"Logistic": "o", "XGBoost": "s", "XGBoost, no anthrop.": "D"}
    for model in model_order:
        sub = perf[perf["model"].eq(model)]
        yy = [task_y[t] + offsets[model] for t in sub["task"]]
        ax1.scatter(
            sub["auc"],
            yy,
            s=24,
            color=model_colors[model],
            marker=markers[model],
            edgecolor="white",
            linewidth=0.35,
            label=model,
            zorder=3,
        )
        for xval, yval in zip(sub["auc"], yy):
            ax1.text(xval + 0.008, yval, f"{xval:.3f}", va="center", ha="left", fontsize=5.4, color=TEXT)
    ax1.set_yticks([task_y[t] for t in task_order])
    ax1.set_yticklabels(task_order)
    ax1.set_xlim(0.55, 0.97)
    ax1.set_xticks([0.6, 0.7, 0.8, 0.9])
    ax1.set_xlabel("Temporal validation AUC")
    ax1.set_title("Digital constitution screening performance", loc="left", pad=3, fontweight="bold")
    ax1.grid(axis="x", color="#E7E7E7", lw=0.45)
    ax1.set_axisbelow(True)
    clean_axes(ax1)
    ax1.legend(frameon=False, loc="lower right", handletextpad=0.4, borderpad=0.1)
    panel_label(ax1, "A", x=-0.15, y=1.03)

    top = shap_imp.sort_values("mean_abs_shap", ascending=True)
    ax2.hlines(top["feature_clean"], 0, top["mean_abs_shap"], color="#B8C9D8", lw=1.7)
    ax2.scatter(top["mean_abs_shap"], top["feature_clean"], s=22, color=BLUE, edgecolor="white", linewidth=0.35, zorder=3)
    ax2.set_xlabel("Mean |SHAP value|")
    ax2.set_title("Model explanation for phlegm-dampness", loc="left", pad=3, fontweight="bold")
    ax2.set_xlim(0, max(top["mean_abs_shap"]) * 1.10)
    ax2.grid(axis="x", color="#E7E7E7", lw=0.45)
    ax2.set_axisbelow(True)
    clean_axes(ax2)
    panel_label(ax2, "B", x=-0.16, y=1.03)
    fig.text(0.085, 0.055, "The SHAP model excludes TCM questionnaire items.", fontsize=5.7, color="#555555")

    save_figure(fig, "figure_5_prediction_and_explainability")


def write_figure_legends() -> None:
    legend_text = """# Draft Figure Legends

**Figure 1. Study cohort and constitution distribution.** (A) Overview of the routine health examination cohort, including total records, unique participants, valid constitution labels, and annual adjacent record pairs used for longitudinal analyses. (B) Distribution of the primary TCM constitution groups among valid records. Values beside bars show percentage and record count.

**Figure 2. Bidirectional constitution-disease map at baseline.** (A) Constitution composition among participants with each recorded disease or examination-derived marker. (B) Age- and sex-adjusted odds ratios for each biased constitution group compared with balanced constitution. Cell labels show odds ratios; asterisks indicate false-discovery-rate-adjusted q<0.05.

**Figure 3. Current constitution and next-visit disease/risk markers.** Heatmap of lagged associations between current constitution and subsequent disease-related markers or biochemical risk markers at the next annual visit. Models were adjusted for age, sex, follow-up interval, and current marker status. Cell labels show odds ratios; asterisks indicate false-discovery-rate-adjusted q<0.05.

**Figure 4. Constitution-state transition dynamics.** (A) Adjacent constitution-state transition matrix across annual visits. Rows represent the current observed state and columns represent the next observed state. (B) Standardized predictors of two clinically relevant transitions: balanced to biased constitution and phlegm-dampness to balanced constitution. Points show odds ratios per standard-deviation increment; horizontal lines show 95% confidence intervals.

**Figure 5. Digital constitution screening and model explanation.** (A) Temporal validation performance for routine-examination-based constitution screening models, with a sensitivity analysis excluding anthropometric variables. (B) Mean absolute SHAP values for the XGBoost phlegm-dampness classifier after excluding TCM questionnaire items, showing the leading routine-examination predictors.
"""
    (FIG_DIR.parent / "figure_legends_draft.md").write_text(legend_text, encoding="utf-8")


def main() -> None:
    set_theme()
    figure_1_cohort_overview()
    figure_2_bidirectional_disease_map()
    figure_3_longitudinal_risk_map()
    figure_4_transition_dynamics()
    figure_5_prediction_and_explainability()
    write_figure_legends()

    index = pd.DataFrame(
        [
            {
                "figure": "Figure 1",
                "file_stem": "figure_1_cohort_overview",
                "title": "Study cohort and constitution distribution",
                "purpose": "Main-text overview figure",
            },
            {
                "figure": "Figure 2",
                "file_stem": "figure_2_bidirectional_disease_map",
                "title": "Bidirectional constitution-disease map",
                "purpose": "Main-text bidirectional disease/marker map",
            },
            {
                "figure": "Figure 3",
                "file_stem": "figure_3_longitudinal_risk_map",
                "title": "Current constitution and next-visit risk",
                "purpose": "Main-text longitudinal risk figure",
            },
            {
                "figure": "Figure 4",
                "file_stem": "figure_4_transition_dynamics",
                "title": "Constitution-state transition dynamics",
                "purpose": "Main-text dynamic health-state figure",
            },
            {
                "figure": "Figure 5",
                "file_stem": "figure_5_prediction_and_explainability",
                "title": "Digital screening model and explainability",
                "purpose": "Main or supplementary figure depending on journal limits",
            },
        ]
    )
    index.to_csv(FIG_DIR / "publication_figure_index.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
