from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INITIAL_TABLES = ROOT / "outputs" / "initial_analysis" / "tables"
ADV_TABLES = ROOT / "outputs" / "advanced_analysis" / "tables"
MANUSCRIPT_FIGS = ROOT / "outputs" / "manuscript" / "figures"
ADV_FIGS = ROOT / "outputs" / "advanced_analysis" / "figures"
OUT = ROOT / "outputs" / "plos_submission"
OUT_TABLES = OUT / "tables"
OUT_FIGS = OUT / "figures"
OUT_SUPP = OUT / "supporting_information"

for path in [OUT_TABLES, OUT_FIGS, OUT_SUPP]:
    path.mkdir(parents=True, exist_ok=True)


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

TARGET_ORDER = ["Phlegm-dampness", "Yang-deficiency", "Qi-deficiency", "Yin-deficiency"]


def fmt_p(value: object) -> str:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("<"):
            return text
        if text == "":
            return ""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return ""
    if np.isnan(x):
        return ""
    if x < 0.001:
        return "<0.001"
    return f"{x:.3f}"


def fmt_ci(row: pd.Series, point: str = "or", low: str = "or_ci_low", high: str = "or_ci_high") -> str:
    try:
        q = float(row.get("q_value_all", row.get("q_value", np.nan)))
        star = "*" if np.isfinite(q) and q < 0.05 else ""
        return f"{float(row[point]):.2f} ({float(row[low]):.2f}-{float(row[high]):.2f}){star}"
    except Exception:
        return ""


def write_table(df: pd.DataFrame, stem: str, title: str, notes: list[str] | None = None) -> None:
    csv_path = OUT_TABLES / f"{stem}.csv"
    md_path = OUT_TABLES / f"{stem}.md"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    lines = [f"# {title}", ""]
    lines.extend(markdown_table(df))
    if notes:
        lines.extend(["", "Notes:"])
        lines.extend(f"- {note}" for note in notes)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> list[str]:
    display = df.fillna("").astype(str)
    headers = list(display.columns)

    def esc(value: str) -> str:
        return value.replace("|", "\\|").replace("\n", "<br>")

    lines = [
        "| " + " | ".join(esc(h) for h in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(esc(row[h]) for h in headers) + " |")
    return lines


def make_table1() -> None:
    table = pd.read_csv(INITIAL_TABLES / "table1_baseline_by_constitution_first_record.csv")
    table = table.rename(columns=GROUP_MAP)
    selected = [
        "N",
        "Age, years",
        "Female",
        "BMI, kg/m2",
        "Waist circumference, cm",
        "Systolic BP, mmHg",
        "Diastolic BP, mmHg",
        "Fasting glucose, mmol/L",
        "Triglycerides, mmol/L",
        "HDL-C, mmol/L",
        "LDL-C, mmol/L",
        "Overweight (BMI>=24)",
        "Obesity (BMI>=28)",
        "High BP (>=140/90)",
        "FPG >=6.1 mmol/L",
        "FPG >=7.0 mmol/L",
        "High TG",
        "Low HDL-C",
        "Any dyslipidemia",
        "eGFR <60",
        "ALT, U/L",
        "Creatinine, umol/L",
        "Hemoglobin, g/L",
    ]
    cols = [
        "variable",
        "Overall",
        "Balanced",
        "Phlegm-dampness",
        "Yang-deficiency",
        "Qi-deficiency",
        "Yin-deficiency",
        "Other biased",
        "p_value",
        "max_smd_vs_pinghe",
    ]
    out = table[table["variable"].isin(selected)][cols].copy()
    out["p_value"] = out["p_value"].map(fmt_p)
    out["max_smd_vs_pinghe"] = pd.to_numeric(out["max_smd_vs_pinghe"], errors="coerce").map(
        lambda x: "" if pd.isna(x) else f"{x:.2f}"
    )
    out = out.rename(
        columns={
            "variable": "Characteristic",
            "p_value": "P value",
            "max_smd_vs_pinghe": "Maximum SMD vs balanced",
        }
    )
    write_table(
        out,
        "Table1_baseline_characteristics",
        "Table 1. Baseline characteristics by constitution group",
        [
            "Continuous variables are shown as mean (standard deviation); binary variables are shown as number (percentage).",
            "P values compare all constitution groups using Kruskal-Wallis tests for continuous variables and chi-square tests for binary variables.",
            "SMD, standardized mean difference.",
        ],
    )


def association_pivot(
    data: pd.DataFrame,
    outcome_col: str,
    selected_outcomes: list[str],
    *,
    stem: str,
    title: str,
) -> None:
    work = data.copy()
    work["target_constitution"] = work["target_constitution"].map(TARGET_MAP).fillna(work["target_constitution"])
    work = work[work[outcome_col].isin(selected_outcomes)].copy()
    rows = []
    for outcome in selected_outcomes:
        row = {"Outcome": outcome}
        sub = work[work[outcome_col].eq(outcome)]
        for target in TARGET_ORDER:
            hit = sub[sub["target_constitution"].eq(target)]
            row[target] = fmt_ci(hit.iloc[0]) if not hit.empty else ""
        rows.append(row)
    out = pd.DataFrame(rows)
    write_table(
        out,
        stem,
        title,
        [
            "Values are odds ratios with 95% confidence intervals.",
            "*False-discovery-rate-adjusted q<0.05.",
            "Balanced constitution is the reference group.",
        ],
    )


def make_table2() -> None:
    data = pd.read_csv(INITIAL_TABLES / "constitution_to_disease_associations_baseline.csv")
    selected = [
        "Recorded cerebrovascular disease",
        "Recorded heart disease",
        "Abnormal ECG",
        "Abnormal abdominal ultrasound",
        "Liver enzyme elevation",
        "Diabetes-related marker",
        "Kidney impairment or proteinuria",
        "Cardiometabolic risk cluster",
    ]
    association_pivot(
        data,
        "disease_or_marker",
        selected,
        stem="Table2_baseline_constitution_disease_associations",
        title="Table 2. Baseline constitution-to-disease and disease-marker associations",
    )


def make_table3() -> None:
    disease = pd.read_csv(INITIAL_TABLES / "lagged_constitution_future_disease_risk_models.csv")
    disease = disease.rename(columns={"disease_or_marker": "Outcome"})
    bio = pd.read_csv(INITIAL_TABLES / "lagged_constitution_future_risk_models.csv")
    bio = bio[bio["model"].eq("next_status_adjusted_for_current_status")].rename(columns={"outcome": "Outcome"})
    combined = pd.concat(
        [
            disease[
                [
                    "target_constitution",
                    "Outcome",
                    "or",
                    "or_ci_low",
                    "or_ci_high",
                    "q_value_all",
                    "n",
                    "events_next",
                ]
            ],
            bio[
                [
                    "target_constitution",
                    "Outcome",
                    "or",
                    "or_ci_low",
                    "or_ci_high",
                    "q_value_all",
                    "n",
                    "events_next",
                ]
            ],
        ],
        ignore_index=True,
    )
    selected = [
        "High TG",
        "Low HDL-C",
        "FPG >=6.1 mmol/L",
        "FPG >=7.0 mmol/L",
        "High BP (>=140/90)",
        "Abnormal abdominal ultrasound",
        "Cardiometabolic risk cluster",
        "Diabetes-related marker",
        "Urine protein trace/positive",
        "Kidney impairment or proteinuria",
        "Recorded cerebrovascular disease",
        "Anemia",
        "eGFR <60",
    ]
    association_pivot(
        combined,
        "Outcome",
        selected,
        stem="Table3_lagged_next_visit_associations",
        title="Table 3. Current constitution and next-visit disease-related or biochemical risk markers",
    )


def make_table4() -> None:
    full = pd.read_csv(ADV_TABLES / "prediction_calibration_metrics.csv")
    full["feature_set"] = "Full routine examination feature set"
    sens = pd.read_csv(INITIAL_TABLES / "temporal_validation_model_metrics_sensitivity_no_anthro.csv")
    sens = sens.rename(columns={"feature_set": "feature_set"})
    dca = pd.read_csv(ADV_TABLES / "decision_curve_net_benefit_summary.csv")
    dca03 = dca[np.isclose(pd.to_numeric(dca["threshold_probability"]), 0.30)].copy()
    dca03 = dca03[["task", "model", "model_net_benefit", "treat_all_net_benefit"]]
    dca03 = dca03.rename(
        columns={
            "model_net_benefit": "net_benefit_threshold_0_30",
            "treat_all_net_benefit": "treat_all_net_benefit_threshold_0_30",
        }
    )
    combined = pd.concat([full, sens], ignore_index=True, sort=False)
    combined = combined.merge(dca03, on=["task", "model"], how="left")
    combined.loc[
        combined["feature_set"].ne("Full routine examination feature set"),
        ["net_benefit_threshold_0_30", "treat_all_net_benefit_threshold_0_30"],
    ] = np.nan
    cols = [
        "task",
        "model",
        "feature_set",
        "auc",
        "pr_auc",
        "balanced_accuracy",
        "brier",
        "ece_10bin",
        "calibration_slope",
        "net_benefit_threshold_0_30",
    ]
    out = combined[cols].copy()
    for col in cols[3:]:
        out[col] = pd.to_numeric(out[col], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    out = out.rename(
        columns={
            "task": "Prediction task",
            "model": "Model",
            "feature_set": "Feature set",
            "auc": "AUC",
            "pr_auc": "PR-AUC",
            "balanced_accuracy": "Balanced accuracy",
            "brier": "Brier score",
            "ece_10bin": "ECE (10-bin)",
            "calibration_slope": "Calibration slope",
            "net_benefit_threshold_0_30": "Net benefit at threshold 0.30",
        }
    )
    write_table(
        out,
        "Table4_temporal_validation_prediction_metrics",
        "Table 4. Temporal validation, calibration, and decision-curve metrics for digital constitution prediction",
        [
            "Training records were from 2017-2023, validation records from 2024, and temporal-test records from 2025-2026.",
            "Calibration and decision-curve metrics are shown for the full feature set.",
            "The sensitivity feature set excludes height, weight, BMI, and waist circumference.",
            "ECE, expected calibration error; PR-AUC, average precision.",
        ],
    )


def copy_supporting_tables() -> None:
    copies = {
        "S1_Table_disease_to_constitution_enrichment.csv": ADV_TABLES
        / "patient_facing_disease_to_constitution_enrichment.csv",
        "S2_Table_burden_score_dictionary.csv": ADV_TABLES / "burden_score_dictionary.csv",
        "S3_Table_constitution_burden_models.csv": ADV_TABLES / "constitution_burden_models.csv",
        "S4_Table_evalue_sensitivity.csv": ADV_TABLES / "evalue_primary_longitudinal_associations.csv",
        "S5_Table_subgroup_interaction_models.csv": ADV_TABLES / "subgroup_phlegm_dampness_primary_associations.csv",
        "S6_Table_prediction_sensitivity_no_anthro.csv": INITIAL_TABLES
        / "temporal_validation_model_metrics_sensitivity_no_anthro.csv",
        "S7_Table_conformal_uncertainty.csv": INITIAL_TABLES / "conformal_prediction_summary.csv",
        "S8_Table_variable_missingness.csv": INITIAL_TABLES / "key_variable_missingness_summary.csv",
        "S9_Table_disease_marker_missingness.csv": INITIAL_TABLES / "disease_marker_missingness_summary.csv",
    }
    index_rows = []
    for out_name, source in copies.items():
        dest = OUT_SUPP / out_name
        shutil.copy2(source, dest)
        index_rows.append({"file": out_name, "source": str(source.relative_to(ROOT))})
    pd.DataFrame(index_rows).to_csv(OUT_SUPP / "supporting_information_index.csv", index=False, encoding="utf-8-sig")


def copy_figures() -> None:
    figure_map = {
        "Fig1": (MANUSCRIPT_FIGS, "figure_1_cohort_overview"),
        "Fig2": (MANUSCRIPT_FIGS, "figure_2_bidirectional_disease_map"),
        "Fig3": (MANUSCRIPT_FIGS, "figure_3_longitudinal_risk_map"),
        "Fig4": (MANUSCRIPT_FIGS, "figure_4_transition_dynamics"),
        "Fig5": (MANUSCRIPT_FIGS, "figure_5_prediction_and_explainability"),
        "S1_Fig": (ADV_FIGS, "supplementary_figure_calibration_curves"),
    }
    rows = []
    for plos_name, (source_dir, stem) in figure_map.items():
        for ext in ["tiff", "png", "pdf", "svg"]:
            source = source_dir / f"{stem}.{ext}"
            if not source.exists():
                continue
            dest = OUT_FIGS / f"{plos_name}.{ext}"
            shutil.copy2(source, dest)
            rows.append(
                {
                    "figure": plos_name,
                    "format": ext,
                    "file": dest.name,
                    "source": str(source.relative_to(ROOT)),
                    "size_bytes": dest.stat().st_size,
                }
            )
    pd.DataFrame(rows).to_csv(OUT_FIGS / "plos_figure_file_index.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    make_table1()
    make_table2()
    make_table3()
    make_table4()
    copy_supporting_tables()
    copy_figures()


if __name__ == "__main__":
    main()
