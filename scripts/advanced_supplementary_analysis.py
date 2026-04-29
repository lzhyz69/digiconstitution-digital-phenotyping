from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "initial_analysis" / "tables"
OUT_DIR = ROOT / "outputs" / "advanced_analysis"
OUT_TABLES = OUT_DIR / "tables"
OUT_FIGS = OUT_DIR / "figures"
MANUSCRIPT_DIR = ROOT / "outputs" / "manuscript"
for path in [OUT_TABLES, OUT_FIGS, MANUSCRIPT_DIR]:
    path.mkdir(parents=True, exist_ok=True)

sys.path.append(str(ROOT / "scripts"))
import initial_constitution_analysis as ica  # noqa: E402


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
    "其他偏颇质": "Other biased",
}

TEXT = "#222222"
GRID = "#D6D6D6"
BLUE = "#2F6FAE"
RED = "#B74346"
GRAY = "#777777"

OUTCOME_LABELS = {
    "history_cerebrovascular": "Recorded cerebrovascular disease",
    "history_kidney_disease": "Recorded kidney disease",
    "history_heart_disease": "Recorded heart disease",
    "history_vascular_disease": "Recorded vascular disease",
    "history_eye_disease": "Recorded eye disease",
    "history_neuro_disease": "Recorded neurological disease",
    "high_bp": "High BP (>=140/90)",
    "high_fpg": "FPG >=6.1 mmol/L",
    "diabetes_fpg": "FPG >=7.0 mmol/L",
    "high_tg": "High TG",
    "high_ldl": "High LDL-C",
    "low_hdl": "Low HDL-C",
    "low_egfr": "eGFR <60",
    "any_dyslipidemia": "Any dyslipidemia",
    "obesity": "Obesity",
    "abnormal_ecg": "Abnormal ECG",
    "abnormal_abdominal_ultrasound": "Abnormal abdominal ultrasound",
    "proteinuria_trace_or_positive": "Proteinuria trace/positive",
    "glycosuria_trace_or_positive": "Glycosuria trace/positive",
    "hematuria_trace_or_positive": "Hematuria trace/positive",
    "liver_enzyme_elevated": "Liver enzyme elevation",
    "anemia": "Anemia",
    "kidney_impairment_or_proteinuria": "Kidney impairment or proteinuria",
    "diabetes_related_marker": "Diabetes-related marker",
    "cardiometabolic_risk_cluster": "Cardiometabolic risk cluster",
}


def set_theme() -> None:
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


def p_adjust_bh(p_values: pd.Series) -> pd.Series:
    values = pd.to_numeric(p_values, errors="coerce")
    adjusted = pd.Series(np.nan, index=values.index, dtype=float)
    mask = values.notna()
    p = values.loc[mask].to_numpy(dtype=float)
    if len(p) == 0:
        return adjusted
    order = np.argsort(p)
    ranked = p[order]
    m = len(p)
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty_like(q)
    out[order] = q
    adjusted.loc[mask] = out
    return adjusted


def fit_binary_logit(data: pd.DataFrame, outcome: str, target: str, covariates: list[str]) -> dict | None:
    cols = [outcome, target] + covariates
    model_df = data[cols].dropna().copy()
    if len(model_df) < 80 or model_df[outcome].nunique() < 2 or model_df[target].nunique() < 2:
        return None
    x = sm.add_constant(model_df[[target] + covariates].astype(float), has_constant="add")
    y = model_df[outcome].astype(float)
    try:
        fit = sm.GLM(y, x, family=sm.families.Binomial()).fit(cov_type="HC1", maxiter=100)
    except Exception:
        return None
    coef = float(fit.params[target])
    se = float(fit.bse[target])
    return {
        "estimate": coef,
        "ci_low": coef - 1.96 * se,
        "ci_high": coef + 1.96 * se,
        "p_value": float(fit.pvalues[target]),
        "or": float(math.exp(coef)),
        "or_ci_low": float(math.exp(coef - 1.96 * se)),
        "or_ci_high": float(math.exp(coef + 1.96 * se)),
        "n": int(len(model_df)),
        "events": int(y.sum()),
    }


def fit_poisson_burden_model(
    data: pd.DataFrame,
    count_col: str,
    available_col: str,
    target_group: str,
) -> dict | None:
    sub = data[data["constitution_group"].isin(["平和质", target_group])].copy()
    sub["target"] = sub["constitution_group"].eq(target_group).astype(int)
    sub["female"] = sub["female"].astype(float)
    sub = sub[[count_col, available_col, "target", "age", "female"]].dropna()
    sub = sub[sub[available_col] > 0].copy()
    if len(sub) < 100 or sub["target"].nunique() < 2:
        return None
    x = sm.add_constant(sub[["target", "age", "female"]].astype(float), has_constant="add")
    y = sub[count_col].astype(float)
    offset = np.log(sub[available_col].astype(float))
    try:
        fit = sm.GLM(y, x, family=sm.families.Poisson(), offset=offset).fit(cov_type="HC1", maxiter=100)
    except Exception:
        return None
    coef = float(fit.params["target"])
    se = float(fit.bse["target"])
    return {
        "target_constitution": GROUP_MAP[target_group],
        "reference": "Balanced",
        "estimate": coef,
        "ci_low": coef - 1.96 * se,
        "ci_high": coef + 1.96 * se,
        "p_value": float(fit.pvalues["target"]),
        "irr": float(math.exp(coef)),
        "irr_ci_low": float(math.exp(coef - 1.96 * se)),
        "irr_ci_high": float(math.exp(coef + 1.96 * se)),
        "n": int(len(sub)),
        "total_events": int(y.sum()),
        "mean_available_markers": float(sub[available_col].mean()),
    }


def make_burden_scores(baseline: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    recorded_cols = [
        "history_cerebrovascular",
        "history_kidney_disease",
        "history_heart_disease",
        "history_vascular_disease",
        "history_eye_disease",
        "history_neuro_disease",
    ]
    cardiometabolic_cols = [
        "obesity",
        "high_bp",
        "high_fpg",
        "high_tg",
        "high_ldl",
        "low_hdl",
        "low_egfr",
    ]
    exam_marker_cols = [
        "high_bp",
        "diabetes_related_marker",
        "any_dyslipidemia",
        "abnormal_ecg",
        "abnormal_abdominal_ultrasound",
        "proteinuria_trace_or_positive",
        "glycosuria_trace_or_positive",
        "hematuria_trace_or_positive",
        "liver_enzyme_elevated",
        "anemia",
        "kidney_impairment_or_proteinuria",
    ]
    burden_sets = {
        "recorded_disease_burden": recorded_cols,
        "cardiometabolic_marker_burden": cardiometabolic_cols,
        "examination_marker_burden": exam_marker_cols,
    }
    data = baseline.copy()
    for burden_name, cols in burden_sets.items():
        data[f"{burden_name}_available"] = data[cols].notna().sum(axis=1)
        data[f"{burden_name}_count"] = data[cols].fillna(0).sum(axis=1)
        data[f"{burden_name}_fraction"] = np.where(
            data[f"{burden_name}_available"] > 0,
            data[f"{burden_name}_count"] / data[f"{burden_name}_available"],
            np.nan,
        )

    desc_rows = []
    for group_zh, group_en in GROUP_MAP.items():
        sub = data[data["constitution_group"].eq(group_zh)]
        if sub.empty:
            continue
        for burden_name in burden_sets:
            desc_rows.append(
                {
                    "constitution": group_en,
                    "burden": burden_name,
                    "n": int(sub[f"{burden_name}_fraction"].notna().sum()),
                    "mean_count": float(sub[f"{burden_name}_count"].mean()),
                    "sd_count": float(sub[f"{burden_name}_count"].std()),
                    "median_count": float(sub[f"{burden_name}_count"].median()),
                    "mean_fraction": float(sub[f"{burden_name}_fraction"].mean()),
                }
            )
    desc = pd.DataFrame(desc_rows)

    model_rows = []
    target_groups = ["痰湿质", "阳虚质", "气虚质", "阴虚质"]
    for burden_name in burden_sets:
        for target in target_groups:
            result = fit_poisson_burden_model(
                data,
                f"{burden_name}_count",
                f"{burden_name}_available",
                target,
            )
            if result is None:
                continue
            result["burden"] = burden_name
            model_rows.append(result)
    models = pd.DataFrame(model_rows)
    if not models.empty:
        models["q_value"] = models.groupby("burden", group_keys=False)["p_value"].apply(p_adjust_bh)

    dictionary = pd.DataFrame(
        [
            {"burden": name, "included_markers": "; ".join(OUTCOME_LABELS.get(c, c) for c in cols)}
            for name, cols in burden_sets.items()
        ]
    )
    return data, desc, models, dictionary


def disease_to_constitution_enrichment(baseline: pd.DataFrame) -> pd.DataFrame:
    d2c = pd.read_csv(TABLES / "disease_to_constitution_distribution_baseline.csv")
    baseline_dist = baseline["constitution_group"].value_counts(normalize=True).mul(100)
    rows = []
    for _, row in d2c.iterrows():
        candidates = []
        for zh, en in GROUP_MAP.items():
            pct = float(row.get(f"{zh}_percent_among_cases", np.nan))
            base_pct = float(baseline_dist.get(zh, np.nan))
            enrichment = pct / base_pct if base_pct and not np.isnan(base_pct) else np.nan
            candidates.append((en, pct, base_pct, enrichment))
        top_by_percent = max(candidates, key=lambda x: -np.inf if np.isnan(x[1]) else x[1])
        biased_candidates = [c for c in candidates if c[0] != "Balanced"]
        top_biased_by_percent = max(biased_candidates, key=lambda x: -np.inf if np.isnan(x[1]) else x[1])
        top_enriched = max(biased_candidates, key=lambda x: -np.inf if np.isnan(x[3]) else x[3])
        rows.append(
            {
                "disease_or_marker": row["disease_or_marker"],
                "n_cases": int(row["n_cases"]),
                "case_prevalence_percent": float(row["case_prevalence_percent"]),
                "balanced_percent_among_cases": float(row["平和质_percent_among_cases"]),
                "biased_percent_among_cases": float(row["biased_percent_among_cases"]),
                "phlegm_dampness_percent_among_cases": float(row["痰湿质_percent_among_cases"]),
                "yang_deficiency_percent_among_cases": float(row["阳虚质_percent_among_cases"]),
                "qi_deficiency_percent_among_cases": float(row["气虚质_percent_among_cases"]),
                "yin_deficiency_percent_among_cases": float(row["阴虚质_percent_among_cases"]),
                "top_constitution_by_percent": top_by_percent[0],
                "top_constitution_percent": top_by_percent[1],
                "top_biased_constitution_by_percent": top_biased_by_percent[0],
                "top_biased_constitution_percent": top_biased_by_percent[1],
                "most_enriched_biased_constitution": top_enriched[0],
                "most_enriched_biased_ratio_vs_baseline": top_enriched[3],
                "phlegm_dampness_enrichment_ratio_vs_baseline": [
                    c[3] for c in candidates if c[0] == "Phlegm-dampness"
                ][0],
            }
        )
    return pd.DataFrame(rows)


def e_value_from_or(or_value: float, ci_low: float, ci_high: float) -> tuple[float, float]:
    if np.isnan(or_value) or or_value <= 0:
        return np.nan, np.nan
    if or_value < 1:
        or_for_e = 1 / or_value
        ci_for_e = 1 / ci_high if ci_high < 1 else 1
    else:
        or_for_e = or_value
        ci_for_e = ci_low if ci_low > 1 else 1

    def calc(x: float) -> float:
        if x <= 1:
            return 1.0
        return float(x + math.sqrt(x * (x - 1)))

    return calc(or_for_e), calc(ci_for_e)


def make_evalue_table() -> pd.DataFrame:
    bio = pd.read_csv(TABLES / "lagged_constitution_future_risk_models.csv")
    bio = bio[bio["model"].eq("next_status_adjusted_for_current_status")].copy()
    bio = bio.rename(columns={"outcome": "endpoint"})
    bio["endpoint_type"] = "biochemical/examination risk marker"
    dis = pd.read_csv(TABLES / "lagged_constitution_future_disease_risk_models.csv")
    dis = dis.rename(columns={"disease_or_marker": "endpoint"})
    dis["endpoint_type"] = "recorded disease or disease-related marker"
    combined = pd.concat(
        [
            bio[
                [
                    "target_constitution",
                    "reference",
                    "endpoint",
                    "endpoint_type",
                    "or",
                    "or_ci_low",
                    "or_ci_high",
                    "p_value",
                    "q_value_all",
                    "n",
                    "events_next",
                ]
            ],
            dis[
                [
                    "target_constitution",
                    "reference",
                    "endpoint",
                    "endpoint_type",
                    "or",
                    "or_ci_low",
                    "or_ci_high",
                    "p_value",
                    "q_value_all",
                    "n",
                    "events_next",
                ]
            ],
        ],
        ignore_index=True,
    )
    primary_pairs = {
        ("痰湿质", "Abnormal abdominal ultrasound"),
        ("痰湿质", "Cardiometabolic risk cluster"),
        ("痰湿质", "Diabetes-related marker"),
        ("痰湿质", "Kidney impairment or proteinuria"),
        ("痰湿质", "Recorded cerebrovascular disease"),
        ("痰湿质", "High TG"),
        ("痰湿质", "Low HDL-C"),
        ("痰湿质", "FPG >=6.1 mmol/L"),
        ("阳虚质", "Recorded cerebrovascular disease"),
        ("气虚质", "Anemia"),
        ("气虚质", "eGFR <60"),
    }
    combined = combined[
        combined.apply(lambda r: (r["target_constitution"], r["endpoint"]) in primary_pairs, axis=1)
    ].copy()
    rows = []
    for _, row in combined.iterrows():
        e_est, e_ci = e_value_from_or(float(row["or"]), float(row["or_ci_low"]), float(row["or_ci_high"]))
        rows.append(
            {
                "target_constitution": TARGET_MAP.get(row["target_constitution"], row["target_constitution"]),
                "reference": "Balanced",
                "endpoint": row["endpoint"],
                "endpoint_type": row["endpoint_type"],
                "or": row["or"],
                "or_ci_low": row["or_ci_low"],
                "or_ci_high": row["or_ci_high"],
                "q_value": row["q_value_all"],
                "n": int(row["n"]),
                "events_next": int(row["events_next"]),
                "e_value_estimate": e_est,
                "e_value_ci_limit": e_ci,
            }
        )
    return pd.DataFrame(rows).sort_values(["target_constitution", "endpoint"])


def subgroup_phlegm_dampness_analysis(baseline: pd.DataFrame) -> pd.DataFrame:
    outcomes = [
        "cardiometabolic_risk_cluster",
        "diabetes_related_marker",
        "kidney_impairment_or_proteinuria",
        "abnormal_abdominal_ultrasound",
        "abnormal_ecg",
        "history_cerebrovascular",
    ]
    data = baseline[baseline["constitution_group"].isin(["平和质", "痰湿质"])].copy()
    data["phlegm_dampness"] = data["constitution_group"].eq("痰湿质").astype(int)
    data["older_age"] = (data["age"] >= data["age"].median()).astype(int)
    strata = {
        "overall": (data.index, ["age", "female"]),
        "female": (data.index[data["female"].eq(1)], ["age"]),
        "male": (data.index[data["female"].eq(0)], ["age"]),
        "younger_half": (data.index[data["older_age"].eq(0)], ["age", "female"]),
        "older_half": (data.index[data["older_age"].eq(1)], ["age", "female"]),
    }
    rows = []
    for outcome in outcomes:
        for stratum, (idx, covariates) in strata.items():
            result = fit_binary_logit(data.loc[idx], outcome, "phlegm_dampness", covariates)
            if result is None:
                continue
            result.update(
                {
                    "outcome": OUTCOME_LABELS[outcome],
                    "stratum": stratum,
                    "exposure": "Phlegm-dampness vs balanced",
                }
            )
            rows.append(result)

        sex_df = data[[outcome, "phlegm_dampness", "female", "age"]].dropna().copy()
        if sex_df[outcome].nunique() == 2:
            sex_df["interaction"] = sex_df["phlegm_dampness"] * sex_df["female"]
            x = sm.add_constant(sex_df[["phlegm_dampness", "female", "interaction", "age"]].astype(float))
            try:
                fit = sm.GLM(sex_df[outcome].astype(float), x, family=sm.families.Binomial()).fit(cov_type="HC1")
                rows.append(
                    {
                        "outcome": OUTCOME_LABELS[outcome],
                        "stratum": "sex_interaction",
                        "exposure": "Phlegm-dampness x female",
                        "estimate": float(fit.params["interaction"]),
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "p_value": float(fit.pvalues["interaction"]),
                        "or": float(math.exp(fit.params["interaction"])),
                        "or_ci_low": np.nan,
                        "or_ci_high": np.nan,
                        "n": int(len(sex_df)),
                        "events": int(sex_df[outcome].sum()),
                    }
                )
            except Exception:
                pass

        age_df = data[[outcome, "phlegm_dampness", "older_age", "age", "female"]].dropna().copy()
        if age_df[outcome].nunique() == 2:
            age_df["interaction"] = age_df["phlegm_dampness"] * age_df["older_age"]
            x = sm.add_constant(age_df[["phlegm_dampness", "older_age", "interaction", "age", "female"]].astype(float))
            try:
                fit = sm.GLM(age_df[outcome].astype(float), x, family=sm.families.Binomial()).fit(cov_type="HC1")
                rows.append(
                    {
                        "outcome": OUTCOME_LABELS[outcome],
                        "stratum": "age_interaction",
                        "exposure": "Phlegm-dampness x older half",
                        "estimate": float(fit.params["interaction"]),
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "p_value": float(fit.pvalues["interaction"]),
                        "or": float(math.exp(fit.params["interaction"])),
                        "or_ci_low": np.nan,
                        "or_ci_high": np.nan,
                        "n": int(len(age_df)),
                        "events": int(age_df[outcome].sum()),
                    }
                )
            except Exception:
                pass
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_value"] = out.groupby("stratum", group_keys=False)["p_value"].apply(p_adjust_bh)
    return out


def logit_clip(prob: np.ndarray) -> np.ndarray:
    eps = 1e-6
    p = np.clip(prob, eps, 1 - eps)
    return np.log(p / (1 - p))


def calibration_summary(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> tuple[dict, pd.DataFrame]:
    df = pd.DataFrame({"y": y_true.astype(int), "prob": prob.astype(float)})
    df["bin"] = pd.qcut(df["prob"].rank(method="first"), q=n_bins, labels=False) + 1
    deciles = (
        df.groupby("bin")
        .agg(n=("y", "size"), mean_predicted_probability=("prob", "mean"), observed_event_rate=("y", "mean"))
        .reset_index()
    )
    deciles["absolute_calibration_error"] = (
        deciles["mean_predicted_probability"] - deciles["observed_event_rate"]
    ).abs()
    ece = float(np.average(deciles["absolute_calibration_error"], weights=deciles["n"]))
    ici = float(np.mean(np.abs(df["prob"] - df["y"])))

    z = logit_clip(prob)
    x = sm.add_constant(pd.DataFrame({"logit_probability": z}), has_constant="add")
    try:
        fit = sm.GLM(df["y"], x, family=sm.families.Binomial()).fit()
        intercept = float(fit.params["const"])
        slope = float(fit.params["logit_probability"])
    except Exception:
        intercept = np.nan
        slope = np.nan
    summary = {"ece_10bin": ece, "ici_absolute_error": ici, "calibration_intercept": intercept, "calibration_slope": slope}
    return summary, deciles


def prediction_calibration_and_decision_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_df = pd.read_pickle(ROOT / "outputs" / "initial_analysis" / "analysis_model_frame.pkl")
    results = []
    for task_name, target_col in [
        ("Biased vs balanced constitution", "is_biased"),
        ("Phlegm-dampness vs others", "phlegm_damp_any"),
    ]:
        results.extend(ica.run_single_task_models(model_df, task_name, target_col))

    metric_rows = []
    decile_rows = []
    dca_rows = []
    for result in results:
        cal, deciles = calibration_summary(result.test_y, result.test_prob)
        row = dict(result.metrics)
        row.update(cal)
        row["task"] = result.task_name
        row["model"] = result.model_name
        metric_rows.append(row)

        deciles["task"] = result.task_name
        deciles["model"] = result.model_name
        decile_rows.append(deciles)

        for threshold in [0.10, 0.20, 0.30, 0.40, 0.50]:
            prevalence = float(np.mean(result.test_y))
            treat_all = prevalence - (1 - prevalence) * threshold / (1 - threshold)
            dca_rows.append(
                {
                    "task": result.task_name,
                    "model": result.model_name,
                    "threshold_probability": threshold,
                    "model_net_benefit": ica.net_benefit(result.test_y, result.test_prob, threshold),
                    "treat_all_net_benefit": treat_all,
                    "treat_none_net_benefit": 0.0,
                }
            )

    metrics = pd.DataFrame(metric_rows)
    deciles_all = pd.concat(decile_rows, ignore_index=True)
    dca = pd.DataFrame(dca_rows)
    return metrics, deciles_all, dca


def plot_calibration(deciles: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.65, 2.85), sharex=True, sharey=True, constrained_layout=True)
    task_order = ["Biased vs balanced constitution", "Phlegm-dampness vs others"]
    task_titles = {
        "Biased vs balanced constitution": "Biased vs balanced",
        "Phlegm-dampness vs others": "Phlegm-dampness vs others",
    }
    colors = {"Logistic regression": BLUE, "XGBoost": RED}
    handles = []
    labels = []
    for ax, task, letter in zip(axes, task_order, ["A", "B"]):
        sub = deciles[deciles["task"].eq(task)]
        for model, model_df in sub.groupby("model"):
            line = ax.plot(
                model_df["mean_predicted_probability"],
                model_df["observed_event_rate"],
                marker="o",
                ms=2.7,
                lw=0.85,
                label=model,
                color=colors.get(model, "#777777"),
                markeredgewidth=0.0,
            )
            if model not in labels:
                handles.append(line[0])
                labels.append(model)
        ax.plot([0, 1], [0, 1], ls=(0, (3.0, 2.0)), color="#555555", lw=0.65, zorder=0)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(task_titles[task], pad=4.0, color=TEXT)
        ax.set_xlabel("Mean predicted probability")
        ax.grid(True, color=GRID, linewidth=0.32, alpha=0.55)
        ax.tick_params(colors=TEXT, pad=1.5)
        for spine in ax.spines.values():
            spine.set_color("#BFBFBF")
            spine.set_linewidth(0.55)
        ax.text(
            -0.13,
            1.03,
            letter,
            transform=ax.transAxes,
            fontsize=8.8,
            fontweight="bold",
            va="bottom",
            ha="left",
            color=TEXT,
        )
    axes[0].set_ylabel("Observed event rate")
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        handlelength=1.9,
        columnspacing=1.2,
    )
    for ext in ["pdf", "svg", "png", "tiff"]:
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.025, "facecolor": "white"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        if ext == "tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(OUT_FIGS / f"supplementary_figure_calibration_curves.{ext}", **kwargs)
    plt.close(fig)


def write_addendum(
    burden_models: pd.DataFrame,
    enrichment: pd.DataFrame,
    evalues: pd.DataFrame,
    subgroup: pd.DataFrame,
    calibration: pd.DataFrame,
) -> None:
    def fmt_ci(row: pd.Series, point: str, low: str, high: str) -> str:
        return f"{row[point]:.2f} ({row[low]:.2f}-{row[high]:.2f})"

    phlegm_burden = burden_models[
        burden_models["target_constitution"].eq("Phlegm-dampness")
        & burden_models["burden"].eq("cardiometabolic_marker_burden")
    ].iloc[0]
    exam_burden = burden_models[
        burden_models["target_constitution"].eq("Phlegm-dampness")
        & burden_models["burden"].eq("examination_marker_burden")
    ].iloc[0]
    top_enriched = enrichment.sort_values("phlegm_dampness_enrichment_ratio_vs_baseline", ascending=False).head(5)
    strongest_e = evalues.sort_values("e_value_ci_limit", ascending=False).head(5)
    cal_xgb_phlegm = calibration[
        calibration["task"].eq("Phlegm-dampness vs others") & calibration["model"].eq("XGBoost")
    ].iloc[0]
    subgroup_sig = subgroup[
        subgroup["stratum"].isin(["female", "male", "younger_half", "older_half"])
        & subgroup["outcome"].eq("Cardiometabolic risk cluster")
    ].copy()

    lines = [
        "# Advanced Methods and Results Addendum",
        "",
        "## Additional Methods Text",
        "",
        "### Multimarker Burden Modeling",
        "To summarize the overall clinical burden beyond individual binary markers, we constructed three marker burden scores at baseline: recorded disease burden, cardiometabolic marker burden, and examination-derived marker burden. For each participant, the number of positive markers was divided by the number of nonmissing eligible markers. Constitution-specific burden associations were estimated using robust Poisson regression with the log number of available markers as an offset, adjusted for age and sex. Results are reported as incidence rate ratios (IRRs) relative to balanced constitution.",
        "",
        "### Disease-to-Constitution Enrichment Analysis",
        "For each recorded disease or examination-derived marker, we estimated the constitution composition among cases and calculated enrichment ratios by comparing the case-specific constitution proportion with the baseline constitution distribution. This analysis was designed to answer the patient-facing question: among people with a given disease or marker, which constitution patterns are over-represented?",
        "",
        "### Quantitative Sensitivity to Unmeasured Confounding",
        "For selected longitudinal associations, E-values were calculated to quantify the minimum strength of association that an unmeasured confounder would need to have with both constitution and the next-visit outcome, conditional on measured covariates, to explain away the observed association.",
        "",
        "### Subgroup and Interaction Analyses",
        "The main phlegm-dampness associations were repeated by sex and by age group, using the cohort median age as the cut point. Interaction terms were additionally tested to evaluate whether the association between phlegm-dampness and key outcomes differed by sex or age group.",
        "",
        "### Prediction Model Calibration and Decision-Curve Analysis",
        "For temporally validated prediction models, calibration was assessed by deciles of predicted probability, calibration intercept and slope, 10-bin expected calibration error, and integrated absolute calibration error. Clinical utility was explored using decision-curve analysis across prespecified threshold probabilities.",
        "",
        "## Additional Results Text",
        "",
        "### Multimarker Burden",
        f"Compared with balanced constitution, phlegm-dampness constitution showed a higher cardiometabolic marker burden (IRR, {fmt_ci(phlegm_burden, 'irr', 'irr_ci_low', 'irr_ci_high')}) and examination-derived marker burden (IRR, {fmt_ci(exam_burden, 'irr', 'irr_ci_low', 'irr_ci_high')}) after adjustment for age and sex. These findings support the interpretation of phlegm-dampness as a multimarker cardiometabolic risk state rather than an isolated abnormality in any single biochemical indicator.",
        "",
        "### Patient-Facing Disease-to-Constitution Enrichment",
        "The disease-to-constitution enrichment analysis identified the disease or marker groups in which phlegm-dampness was most over-represented relative to its baseline frequency:",
    ]
    for _, row in top_enriched.iterrows():
        lines.append(
            f"- {row['disease_or_marker']}: phlegm-dampness {row['phlegm_dampness_percent_among_cases']:.1f}% among cases, enrichment ratio {row['phlegm_dampness_enrichment_ratio_vs_baseline']:.2f}."
        )
    lines.extend(
        [
            "",
            "### E-Value Sensitivity Analysis",
            "Among the selected longitudinal associations, the largest E-values for the confidence-limit closest to the null were observed for:",
        ]
    )
    for _, row in strongest_e.iterrows():
        lines.append(
            f"- {row['target_constitution']} -> next {row['endpoint']}: OR {fmt_ci(row, 'or', 'or_ci_low', 'or_ci_high')}; E-value for CI limit {row['e_value_ci_limit']:.2f}."
        )
    lines.extend(
        [
            "",
            "### Subgroup Consistency",
            "The phlegm-dampness association with cardiometabolic risk clustering was directionally consistent across sex and age strata:",
        ]
    )
    for _, row in subgroup_sig.iterrows():
        lines.append(f"- {row['stratum']}: OR {fmt_ci(row, 'or', 'or_ci_low', 'or_ci_high')}.")
    lines.extend(
        [
            "",
            "### Model Calibration",
            f"For phlegm-dampness prediction, the XGBoost model had a temporal-test AUC of {cal_xgb_phlegm['auc']:.3f}, Brier score of {cal_xgb_phlegm['brier']:.3f}, 10-bin expected calibration error of {cal_xgb_phlegm['ece_10bin']:.3f}, and calibration slope of {cal_xgb_phlegm['calibration_slope']:.2f}. These results should be reported alongside discrimination metrics to avoid presenting the model as a purely rank-based classifier.",
            "",
            "## Output Files",
            "",
            "- `outputs/advanced_analysis/tables/constitution_burden_models.csv`",
            "- `outputs/advanced_analysis/tables/burden_descriptive_by_constitution.csv`",
            "- `outputs/advanced_analysis/tables/patient_facing_disease_to_constitution_enrichment.csv`",
            "- `outputs/advanced_analysis/tables/evalue_primary_longitudinal_associations.csv`",
            "- `outputs/advanced_analysis/tables/subgroup_phlegm_dampness_primary_associations.csv`",
            "- `outputs/advanced_analysis/tables/prediction_calibration_metrics.csv`",
            "- `outputs/advanced_analysis/tables/prediction_calibration_deciles.csv`",
            "- `outputs/advanced_analysis/tables/decision_curve_net_benefit_summary.csv`",
            "- `outputs/advanced_analysis/figures/supplementary_figure_calibration_curves.*`",
        ]
    )
    (MANUSCRIPT_DIR / "advanced_methods_and_results_addendum.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    set_theme()
    baseline = pd.read_csv(TABLES / "baseline_first_record_analysis_frame.csv")
    _, burden_desc, burden_models, burden_dictionary = make_burden_scores(baseline)
    burden_desc.to_csv(OUT_TABLES / "burden_descriptive_by_constitution.csv", index=False, encoding="utf-8-sig")
    burden_models.to_csv(OUT_TABLES / "constitution_burden_models.csv", index=False, encoding="utf-8-sig")
    burden_dictionary.to_csv(OUT_TABLES / "burden_score_dictionary.csv", index=False, encoding="utf-8-sig")

    enrichment = disease_to_constitution_enrichment(baseline)
    enrichment.to_csv(
        OUT_TABLES / "patient_facing_disease_to_constitution_enrichment.csv",
        index=False,
        encoding="utf-8-sig",
    )

    evalues = make_evalue_table()
    evalues.to_csv(OUT_TABLES / "evalue_primary_longitudinal_associations.csv", index=False, encoding="utf-8-sig")

    subgroup = subgroup_phlegm_dampness_analysis(baseline)
    subgroup.to_csv(
        OUT_TABLES / "subgroup_phlegm_dampness_primary_associations.csv",
        index=False,
        encoding="utf-8-sig",
    )

    calibration, deciles, dca = prediction_calibration_and_decision_tables()
    calibration.to_csv(OUT_TABLES / "prediction_calibration_metrics.csv", index=False, encoding="utf-8-sig")
    deciles.to_csv(OUT_TABLES / "prediction_calibration_deciles.csv", index=False, encoding="utf-8-sig")
    dca.to_csv(OUT_TABLES / "decision_curve_net_benefit_summary.csv", index=False, encoding="utf-8-sig")
    plot_calibration(deciles)

    write_addendum(burden_models, enrichment, evalues, subgroup, calibration)


if __name__ == "__main__":
    main()
