from __future__ import annotations

import json
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import statsmodels.api as sm
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.multitest import multipletests
from xgboost import XGBClassifier


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "德阳数据库2026版.xlsx"
SHEET_NAME = "合并分析表"
OUT_DIR = ROOT / "outputs" / "initial_analysis"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
NB_PATH = ROOT / "output" / "jupyter-notebook" / "digiconstitution_initial_analysis.ipynb"

for path in [OUT_DIR, FIG_DIR, TABLE_DIR, NB_PATH.parent]:
    path.mkdir(parents=True, exist_ok=True)


plt.rcParams.update(
    {
        "font.sans-serif": [
            "Microsoft YaHei",
            "SimHei",
            "Arial Unicode MS",
            "DejaVu Sans",
        ],
        "axes.unicode_minus": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.linewidth": 0.8,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    }
)
sns.set_theme(style="whitegrid", font="Microsoft YaHei")


CONSTITUTIONS = [
    "平和质",
    "气虚质",
    "阳虚质",
    "阴虚质",
    "痰湿质",
    "湿热质",
    "血瘀质",
    "气郁质",
    "特禀质",
]

MAIN_GROUP_ORDER = ["平和质", "痰湿质", "阳虚质", "气虚质", "阴虚质", "其他偏颇质"]
COMPARISON_TARGETS = ["痰湿质", "阳虚质", "气虚质", "阴虚质"]


RAW_TO_ANALYSIS = {
    "person_id": "档案号",
    "exam_date": "体检_日期N",
    "tcm_date": "中医_日期N",
    "gender": "体检_性别",
    "age": "体检_年龄N",
    "height": "体检_身高N",
    "weight": "体检_体重N",
    "waist": "体检_腰围",
    "bmi": "体检_体质指数_原始",
    "sbp_left": "体检_血压（左侧）_收缩压",
    "dbp_left": "体检_血压（左侧）_舒张压",
    "sbp_right": "体检_血压（右侧）_收缩压",
    "dbp_right": "体检_血压（右侧）_舒张压",
    "hemoglobin": "体检_血红蛋白",
    "wbc": "体检_白细胞",
    "platelet": "体检_血小板",
    "fpg": "体检_空腹血糖",
    "hba1c": "体检_糖化血红蛋白",
    "alt": "体检_血清谷丙转氨酶",
    "ast": "体检_血清谷草转氨酶",
    "albumin": "体检_白蛋白",
    "tbil": "体检_总胆红素",
    "creatinine": "体检_血清肌酐",
    "urea": "体检_血尿素",
    "tc": "体检_总胆固醇",
    "tg": "体检_甘油三酯",
    "ldl": "体检_血清低密度脂蛋白胆固醇",
    "hdl": "体检_血清高密度脂蛋白胆固醇",
    "exercise": "体检_锻炼频率",
    "diet": "体检_饮食习惯",
    "smoking": "体检_吸烟状况",
    "drinking": "体检_饮酒频率",
    "self_health": "体检_老年人健康状态自我评估",
    "self_care": "体检_老年人生活自理能力自我评估",
    "cognition": "体检_老年人认知功能",
    "emotion": "体检_老年人情感状态",
    "cerebrovascular": "体检_脑血管疾病",
    "kidney_disease": "体检_肾脏疾病",
    "heart_disease": "体检_心脏疾病",
    "vascular_disease": "体检_血管疾病",
    "eye_disease": "体检_眼部疾病",
    "neuro_disease": "体检_神经系统疾病",
    "other_system_disease": "体检_其他系统疾病",
    "urine_protein": "体检_尿蛋白",
    "urine_glucose": "体检_尿糖",
    "urine_ketone": "体检_尿酮体",
    "urine_occult_blood": "体检_尿潜血",
    "ecg": "体检_心电图",
    "chest_xray": "体检_胸部X线片",
    "abdominal_ultrasound": "体检_腹部B超",
    "constitution_label": "中医_体质类型_订正",
    "constitution_label_alt": "中医_体质类型N",
}


NUMERIC_LIMITS = {
    "age": (60, 110),
    "height": (120, 200),
    "weight": (30, 150),
    "waist": (40, 150),
    "bmi": (10, 60),
    "sbp_left": (70, 260),
    "dbp_left": (35, 170),
    "sbp_right": (70, 260),
    "dbp_right": (35, 170),
    "hemoglobin": (50, 220),
    "wbc": (1, 80),
    "platelet": (20, 1000),
    "fpg": (2, 30),
    "hba1c": (3, 18),
    "alt": (1, 1000),
    "ast": (1, 1000),
    "albumin": (10, 70),
    "tbil": (1, 400),
    "creatinine": (20, 1500),
    "urea": (0.5, 60),
    "tc": (1, 20),
    "tg": (0.1, 30),
    "ldl": (0.1, 15),
    "hdl": (0.1, 5),
}


CONTINUOUS_OUTCOMES = {
    "BMI": "bmi",
    "Waist circumference": "waist",
    "Systolic BP": "sbp_mean",
    "Diastolic BP": "dbp_mean",
    "Hemoglobin": "hemoglobin",
    "White blood cells": "wbc",
    "Platelets": "platelet",
    "Fasting glucose": "fpg",
    "ALT": "alt",
    "AST": "ast",
    "Total bilirubin": "tbil",
    "Creatinine": "creatinine",
    "eGFR": "egfr",
    "Urea": "urea",
    "Total cholesterol": "tc",
    "Triglycerides": "tg",
    "LDL-C": "ldl",
    "HDL-C": "hdl",
}

BINARY_OUTCOMES = {
    "Overweight (BMI>=24)": "overweight",
    "Obesity (BMI>=28)": "obesity",
    "High BP (>=140/90)": "high_bp",
    "FPG >=6.1 mmol/L": "high_fpg",
    "FPG >=7.0 mmol/L": "diabetes_fpg",
    "High TG": "high_tg",
    "High LDL-C": "high_ldl",
    "Low HDL-C": "low_hdl",
    "eGFR <60": "low_egfr",
    "Any dyslipidemia": "any_dyslipidemia",
}

DISEASE_OUTCOMES = {
    "Recorded cerebrovascular disease": "history_cerebrovascular",
    "Recorded heart disease": "history_heart_disease",
    "Recorded kidney disease": "history_kidney_disease",
    "Recorded vascular disease": "history_vascular_disease",
    "Recorded eye disease": "history_eye_disease",
    "Recorded neurological disease": "history_neuro_disease",
    "Recorded other system disease": "history_other_system_disease",
    "Abnormal ECG": "abnormal_ecg",
    "Abnormal abdominal ultrasound": "abnormal_abdominal_ultrasound",
    "Abnormal chest X-ray": "abnormal_chest_xray",
    "Urine protein trace/positive": "proteinuria_trace_or_positive",
    "Urine glucose trace/positive": "glycosuria_trace_or_positive",
    "Urine occult blood trace/positive": "hematuria_trace_or_positive",
    "Liver enzyme elevation": "liver_enzyme_elevated",
    "Anemia": "anemia",
    "Kidney impairment or proteinuria": "kidney_impairment_or_proteinuria",
    "Diabetes-related marker": "diabetes_related_marker",
    "Cardiometabolic risk cluster": "cardiometabolic_risk_cluster",
}

TABLE1_CONTINUOUS = {
    "Age, years": "age",
    "BMI, kg/m2": "bmi",
    "Waist circumference, cm": "waist",
    "Systolic BP, mmHg": "sbp_mean",
    "Diastolic BP, mmHg": "dbp_mean",
    "Fasting glucose, mmol/L": "fpg",
    "Triglycerides, mmol/L": "tg",
    "HDL-C, mmol/L": "hdl",
    "LDL-C, mmol/L": "ldl",
    "ALT, U/L": "alt",
    "Creatinine, umol/L": "creatinine",
    "eGFR, mL/min/1.73m2": "egfr",
    "Hemoglobin, g/L": "hemoglobin",
}

TABLE1_BINARY = {
    "Female": "female",
    "Overweight (BMI>=24)": "overweight",
    "Obesity (BMI>=28)": "obesity",
    "High BP (>=140/90)": "high_bp",
    "FPG >=6.1 mmol/L": "high_fpg",
    "FPG >=7.0 mmol/L": "diabetes_fpg",
    "High TG": "high_tg",
    "High LDL-C": "high_ldl",
    "Low HDL-C": "low_hdl",
    "eGFR <60": "low_egfr",
    "Any dyslipidemia": "any_dyslipidemia",
}

LAGGED_OUTCOMES = {
    "High BP (>=140/90)": "high_bp",
    "FPG >=6.1 mmol/L": "high_fpg",
    "FPG >=7.0 mmol/L": "diabetes_fpg",
    "High TG": "high_tg",
    "Low HDL-C": "low_hdl",
    "High LDL-C": "high_ldl",
    "eGFR <60": "low_egfr",
    "Any dyslipidemia": "any_dyslipidemia",
}

LAGGED_DISEASE_OUTCOMES = {
    key: value
    for key, value in DISEASE_OUTCOMES.items()
    if key
    not in {
        "Recorded other system disease",
        "Abnormal chest X-ray",
    }
}

TRANSITION_PREDICTORS = {
    "Age": "age",
    "BMI": "bmi",
    "Waist circumference": "waist",
    "Systolic BP": "sbp_mean",
    "Diastolic BP": "dbp_mean",
    "Fasting glucose": "fpg",
    "Triglycerides": "tg",
    "HDL-C": "hdl",
    "LDL-C": "ldl",
    "ALT": "alt",
    "Creatinine": "creatinine",
    "eGFR": "egfr",
    "Hemoglobin": "hemoglobin",
}


MODEL_NUMERIC_FEATURES = [
    "age",
    "height",
    "weight",
    "waist",
    "bmi",
    "sbp_mean",
    "dbp_mean",
    "hemoglobin",
    "wbc",
    "platelet",
    "fpg",
    "alt",
    "ast",
    "tbil",
    "creatinine",
    "urea",
    "egfr",
    "tc",
    "tg",
    "ldl",
    "hdl",
]

MODEL_CATEGORICAL_FEATURES = [
    "gender",
    "exercise",
    "diet",
    "smoking",
    "drinking",
    "self_health",
    "self_care",
    "cognition",
    "emotion",
    "cerebrovascular",
    "kidney_disease",
    "heart_disease",
    "vascular_disease",
    "eye_disease",
    "neuro_disease",
    "other_system_disease",
]


def extract_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    text = series.astype("string")
    extracted = text.str.extract(r"([-+]?\d+(?:\.\d+)?)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def parse_constitution_labels(value: object) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    text = (
        text.replace("，", ",")
        .replace("、", ",")
        .replace("；", ",")
        .replace(";", ",")
        .replace("/", ",")
    )
    parts = [p.strip() for p in re.split(r"[,\s]+", text) if p.strip()]
    labels = [p for p in parts if p in CONSTITUTIONS]
    return labels


def main_group(label: object) -> object:
    if pd.isna(label):
        return np.nan
    return label if label in MAIN_GROUP_ORDER[:-1] else "其他偏颇质"


def safe_bool(condition: pd.Series, valid: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=condition.index, dtype="float64")
    out.loc[valid] = condition.loc[valid].astype(float)
    return out


def binary_or_nan(positive: pd.Series, valid: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=positive.index, dtype="float64")
    out.loc[valid] = positive.loc[valid].astype(float)
    return out


def recorded_history_present(series: pd.Series) -> pd.Series:
    missing_tokens = {
        "",
        "nan",
        "none",
        "<na>",
        "na",
        "0",
        "0.0",
        "未查",
        "拒查",
        "拒绝",
        "无",
        "/",
    }
    out = pd.Series(np.nan, index=series.index, dtype="float64")
    for idx, value in series.items():
        if pd.isna(value):
            continue
        text = str(value).strip().lower()
        if text in missing_tokens:
            continue
        if re.fullmatch(r"\d+(?:\.0+)?", text):
            tokens = [str(int(float(text)))]
        else:
            tokens = re.findall(r"\d+", text)
        if not tokens:
            continue
        tokens = [str(int(t)) for t in tokens]
        out.loc[idx] = 0.0 if all(t == "1" for t in tokens) else 1.0
    return out


def abnormal_code_two(series: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=series.index, dtype="float64")
    for idx, value in series.items():
        if pd.isna(value):
            continue
        text = str(value).strip().lower()
        if text in {"", "nan", "none", "<na>", "0", "0.0", "未查", "拒查", "拒绝", "/"}:
            continue
        if re.fullmatch(r"\d+(?:\.0+)?", text):
            tokens = [str(int(float(text)))]
        else:
            tokens = re.findall(r"\d+", text)
        if not tokens:
            continue
        tokens = [str(int(t)) for t in tokens]
        if "2" in tokens:
            out.loc[idx] = 1.0
        elif "1" in tokens:
            out.loc[idx] = 0.0
    return out


def urine_trace_or_positive(series: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=series.index, dtype="float64")
    missing_tokens = {
        "",
        "nan",
        "none",
        "<na>",
        "未查",
        "拒查",
        "拒绝",
        "无尿",
        "检查",
        "据查",
        "/",
    }
    negative_tokens = {"1", "1.0", "-", " -", "--", "正常", "阴性"}
    for idx, value in series.items():
        if pd.isna(value):
            continue
        text = str(value).strip()
        low = text.lower()
        if low in missing_tokens:
            continue
        if text in negative_tokens or low in negative_tokens:
            out.loc[idx] = 0.0
            continue
        tokens = re.findall(r"\d+", text)
        numeric_tokens = [int(t) for t in tokens] if tokens else []
        if numeric_tokens and any(t >= 2 for t in numeric_tokens):
            out.loc[idx] = 1.0
        elif "+" in text:
            out.loc[idx] = 1.0
        elif "-" in text:
            out.loc[idx] = 0.0
    return out


def ckdepi_2021_egfr(scr_umol_l: pd.Series, age: pd.Series, gender: pd.Series) -> pd.Series:
    scr_mg_dl = scr_umol_l / 88.4
    female = gender.astype("string").eq("女")
    kappa = np.where(female, 0.7, 0.9)
    alpha = np.where(female, -0.241, -0.302)
    ratio = scr_mg_dl / kappa
    egfr = (
        142
        * np.power(np.minimum(ratio, 1), alpha)
        * np.power(np.maximum(ratio, 1), -1.200)
        * np.power(0.9938, age)
        * np.where(female, 1.012, 1.0)
    )
    egfr = pd.Series(egfr, index=scr_umol_l.index)
    egfr.loc[scr_umol_l.isna() | age.isna() | gender.isna()] = np.nan
    return egfr


def p_adjust_bh(pvals: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=pvals.index, dtype="float64")
    mask = pvals.notna()
    if mask.any():
        out.loc[mask] = multipletests(pvals.loc[mask].astype(float), method="fdr_bh")[1]
    return out


def format_p(value: float | None) -> str:
    if value is None or pd.isna(value):
        return ""
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def read_and_clean() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
    df = pd.DataFrame(index=raw.index)

    for new_col, raw_col in RAW_TO_ANALYSIS.items():
        df[new_col] = raw[raw_col] if raw_col in raw.columns else np.nan

    df["exam_date"] = pd.to_datetime(df["exam_date"], errors="coerce")
    df["tcm_date"] = pd.to_datetime(df["tcm_date"], errors="coerce")
    df["year"] = df["exam_date"].dt.year

    numeric_cols = [c for c in NUMERIC_LIMITS if c in df.columns]
    for col in numeric_cols:
        df[col] = extract_numeric(df[col])
        lo, hi = NUMERIC_LIMITS[col]
        df.loc[~df[col].between(lo, hi), col] = np.nan

    df["sbp_mean"] = df[["sbp_left", "sbp_right"]].mean(axis=1)
    df["dbp_mean"] = df[["dbp_left", "dbp_right"]].mean(axis=1)
    df["sbp_max"] = df[["sbp_left", "sbp_right"]].max(axis=1)
    df["dbp_max"] = df[["dbp_left", "dbp_right"]].max(axis=1)
    df["egfr"] = ckdepi_2021_egfr(df["creatinine"], df["age"], df["gender"])
    df["female"] = safe_bool(df["gender"].astype("string").eq("女"), df["gender"].notna())

    labels = df["constitution_label"].map(parse_constitution_labels)
    alt_labels = df["constitution_label_alt"].map(parse_constitution_labels)
    labels = labels.where(labels.map(len).gt(0), alt_labels)
    df["constitution_labels"] = labels
    df["primary_constitution"] = labels.map(lambda x: x[0] if x else np.nan)
    df["constitution_group"] = df["primary_constitution"].map(main_group)
    df["is_biased"] = safe_bool(
        df["primary_constitution"].notna() & df["primary_constitution"].ne("平和质"),
        df["primary_constitution"].notna(),
    )
    df["phlegm_damp_any"] = safe_bool(
        labels.map(lambda x: "痰湿质" in x if isinstance(x, list) else False),
        df["primary_constitution"].notna(),
    )
    df["multiple_constitutions"] = labels.map(lambda x: len(x) >= 2 if isinstance(x, list) else False)

    df["overweight"] = safe_bool(df["bmi"].ge(24), df["bmi"].notna())
    df["obesity"] = safe_bool(df["bmi"].ge(28), df["bmi"].notna())
    df["high_bp"] = safe_bool(
        df["sbp_max"].ge(140) | df["dbp_max"].ge(90),
        df["sbp_max"].notna() | df["dbp_max"].notna(),
    )
    df["high_fpg"] = safe_bool(df["fpg"].ge(6.1), df["fpg"].notna())
    df["diabetes_fpg"] = safe_bool(df["fpg"].ge(7.0), df["fpg"].notna())
    df["high_tg"] = safe_bool(df["tg"].ge(1.7), df["tg"].notna())
    df["high_tc"] = safe_bool(df["tc"].ge(5.2), df["tc"].notna())
    df["high_ldl"] = safe_bool(df["ldl"].ge(3.4), df["ldl"].notna())
    df["low_hdl"] = safe_bool(df["hdl"].lt(1.0), df["hdl"].notna())
    df["low_egfr"] = safe_bool(df["egfr"].lt(60), df["egfr"].notna())
    dyslip_valid = df[["high_tg", "high_tc", "high_ldl", "low_hdl"]].notna().any(axis=1)
    df["any_dyslipidemia"] = safe_bool(
        df[["high_tg", "high_tc", "high_ldl", "low_hdl"]].eq(1).any(axis=1),
        dyslip_valid,
    )
    df["history_cerebrovascular"] = recorded_history_present(df["cerebrovascular"])
    df["history_kidney_disease"] = recorded_history_present(df["kidney_disease"])
    df["history_heart_disease"] = recorded_history_present(df["heart_disease"])
    df["history_vascular_disease"] = recorded_history_present(df["vascular_disease"])
    df["history_eye_disease"] = recorded_history_present(df["eye_disease"])
    df["history_neuro_disease"] = recorded_history_present(df["neuro_disease"])
    df["history_other_system_disease"] = recorded_history_present(df["other_system_disease"])
    df["abnormal_ecg"] = abnormal_code_two(df["ecg"])
    df["abnormal_chest_xray"] = abnormal_code_two(df["chest_xray"])
    df["abnormal_abdominal_ultrasound"] = abnormal_code_two(df["abdominal_ultrasound"])
    df["proteinuria_trace_or_positive"] = urine_trace_or_positive(df["urine_protein"])
    df["glycosuria_trace_or_positive"] = urine_trace_or_positive(df["urine_glucose"])
    df["hematuria_trace_or_positive"] = urine_trace_or_positive(df["urine_occult_blood"])
    df["liver_enzyme_elevated"] = safe_bool(
        df["alt"].gt(40) | df["ast"].gt(40),
        df["alt"].notna() | df["ast"].notna(),
    )
    anemia_condition = (
        (df["gender"].astype("string").eq("男") & df["hemoglobin"].lt(130))
        | (df["gender"].astype("string").eq("女") & df["hemoglobin"].lt(120))
    )
    df["anemia"] = safe_bool(anemia_condition, df["gender"].notna() & df["hemoglobin"].notna())
    df["kidney_impairment_or_proteinuria"] = safe_bool(
        df[["low_egfr", "proteinuria_trace_or_positive"]].eq(1).any(axis=1),
        df[["low_egfr", "proteinuria_trace_or_positive"]].notna().any(axis=1),
    )
    df["diabetes_related_marker"] = safe_bool(
        df[["diabetes_fpg", "glycosuria_trace_or_positive"]].eq(1).any(axis=1),
        df[["diabetes_fpg", "glycosuria_trace_or_positive"]].notna().any(axis=1),
    )
    cardiometabolic_components = df[["obesity", "high_bp", "high_fpg", "any_dyslipidemia"]]
    df["cardiometabolic_risk_cluster"] = safe_bool(
        cardiometabolic_components.eq(1).sum(axis=1).ge(2),
        cardiometabolic_components.notna().sum(axis=1).ge(2),
    )

    for col in MODEL_CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()
            df.loc[df[col].isin(["", "nan", "None", "<NA>"]), col] = pd.NA

    return raw, df


def write_overview_tables(raw: pd.DataFrame, df: pd.DataFrame) -> dict:
    id_counts = df["person_id"].value_counts(dropna=True)
    constitution_counts = (
        df["constitution_group"]
        .value_counts(dropna=False)
        .rename_axis("constitution_group")
        .reset_index(name="records")
    )
    constitution_counts["percent"] = constitution_counts["records"] / len(df) * 100
    constitution_counts.to_csv(TABLE_DIR / "constitution_group_distribution.csv", index=False, encoding="utf-8-sig")

    year_dist = (
        df.groupby(["year", "constitution_group"], dropna=False)
        .size()
        .rename("records")
        .reset_index()
    )
    year_dist["percent_within_year"] = year_dist["records"] / year_dist.groupby("year")["records"].transform("sum") * 100
    year_dist.to_csv(TABLE_DIR / "constitution_distribution_by_year.csv", index=False, encoding="utf-8-sig")

    variable_rows = []
    for label, col in {**CONTINUOUS_OUTCOMES, **BINARY_OUTCOMES}.items():
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce")
        variable_rows.append(
            {
                "variable": col,
                "label": label,
                "n_nonmissing": int(x.notna().sum()),
                "missing_percent": round(float(x.isna().mean() * 100), 2),
                "mean": round(float(x.mean()), 4) if x.notna().any() else np.nan,
                "sd": round(float(x.std()), 4) if x.notna().sum() > 1 else np.nan,
                "min": round(float(x.min()), 4) if x.notna().any() else np.nan,
                "max": round(float(x.max()), 4) if x.notna().any() else np.nan,
            }
        )
    variable_summary = pd.DataFrame(variable_rows)
    variable_summary.to_csv(TABLE_DIR / "key_variable_missingness_summary.csv", index=False, encoding="utf-8-sig")

    disease_rows = []
    for label, col in DISEASE_OUTCOMES.items():
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce")
        disease_rows.append(
            {
                "disease_or_marker": label,
                "source_column": col,
                "n_nonmissing": int(x.notna().sum()),
                "n_events": int(x.eq(1).sum()),
                "prevalence_percent": round(float(x.mean() * 100), 2) if x.notna().any() else np.nan,
                "missing_percent": round(float(x.isna().mean() * 100), 2),
            }
        )
    pd.DataFrame(disease_rows).to_csv(TABLE_DIR / "disease_marker_missingness_summary.csv", index=False, encoding="utf-8-sig")

    overview = {
        "n_rows": int(len(df)),
        "n_columns_raw": int(raw.shape[1]),
        "n_unique_persons": int(df["person_id"].nunique(dropna=True)),
        "missing_person_id": int(df["person_id"].isna().sum()),
        "date_min": str(df["exam_date"].min().date()),
        "date_max": str(df["exam_date"].max().date()),
        "n_constitution_nonmissing": int(df["primary_constitution"].notna().sum()),
        "n_constitution_missing": int(df["primary_constitution"].isna().sum()),
        "n_multiple_constitution_records": int(df["multiple_constitutions"].sum()),
        "records_per_person_mean": round(float(id_counts.mean()), 3),
        "records_per_person_median": round(float(id_counts.median()), 3),
        "records_per_person_max": int(id_counts.max()),
        "persons_with_2plus_records": int((id_counts >= 2).sum()),
        "persons_with_3plus_records": int((id_counts >= 3).sum()),
        "persons_with_5plus_records": int((id_counts >= 5).sum()),
    }
    (OUT_DIR / "data_overview.json").write_text(json.dumps(overview, ensure_ascii=False, indent=2), encoding="utf-8")
    return overview


def first_record_per_person(df: pd.DataFrame) -> pd.DataFrame:
    baseline = (
        df.dropna(subset=["person_id", "exam_date", "constitution_group"])
        .sort_values(["person_id", "exam_date"])
        .groupby("person_id", as_index=False)
        .head(1)
        .copy()
    )
    baseline["constitution_group"] = pd.Categorical(
        baseline["constitution_group"], categories=MAIN_GROUP_ORDER, ordered=True
    )
    return baseline.sort_values("constitution_group")


def mean_sd_text(x: pd.Series) -> str:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if x.empty:
        return ""
    return f"{x.mean():.2f} ({x.std():.2f})"


def binary_text(x: pd.Series) -> str:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if x.empty:
        return ""
    return f"{int(x.sum()):,} ({x.mean() * 100:.1f}%)"


def continuous_smd_vs_reference(data: pd.DataFrame, col: str, group_col: str = "constitution_group", reference: str = "平和质") -> float:
    ref = pd.to_numeric(data.loc[data[group_col].eq(reference), col], errors="coerce").dropna()
    if len(ref) < 2:
        return np.nan
    vals = []
    for group in MAIN_GROUP_ORDER:
        if group == reference:
            continue
        x = pd.to_numeric(data.loc[data[group_col].eq(group), col], errors="coerce").dropna()
        if len(x) < 2:
            continue
        pooled = math.sqrt((ref.var() + x.var()) / 2)
        if pooled > 0:
            vals.append(abs((x.mean() - ref.mean()) / pooled))
    return float(max(vals)) if vals else np.nan


def binary_smd_vs_reference(data: pd.DataFrame, col: str, group_col: str = "constitution_group", reference: str = "平和质") -> float:
    ref = pd.to_numeric(data.loc[data[group_col].eq(reference), col], errors="coerce").dropna()
    if ref.empty:
        return np.nan
    p0 = ref.mean()
    vals = []
    for group in MAIN_GROUP_ORDER:
        if group == reference:
            continue
        x = pd.to_numeric(data.loc[data[group_col].eq(group), col], errors="coerce").dropna()
        if x.empty:
            continue
        p1 = x.mean()
        denom = math.sqrt((p0 * (1 - p0) + p1 * (1 - p1)) / 2)
        if denom > 0:
            vals.append(abs((p1 - p0) / denom))
    return float(max(vals)) if vals else np.nan


def table1_p_value_continuous(data: pd.DataFrame, col: str) -> float:
    groups = [
        pd.to_numeric(data.loc[data["constitution_group"].eq(group), col], errors="coerce").dropna()
        for group in MAIN_GROUP_ORDER
    ]
    groups = [x for x in groups if len(x) >= 2]
    if len(groups) < 2:
        return np.nan
    try:
        return float(stats.kruskal(*groups).pvalue)
    except Exception:
        return np.nan


def table1_p_value_binary(data: pd.DataFrame, col: str) -> float:
    sub = data[["constitution_group", col]].dropna()
    if sub["constitution_group"].nunique() < 2 or sub[col].nunique() < 2:
        return np.nan
    table = pd.crosstab(sub["constitution_group"], sub[col])
    if min(table.shape) < 2:
        return np.nan
    try:
        return float(stats.chi2_contingency(table)[1])
    except Exception:
        return np.nan


def write_table1(df: pd.DataFrame) -> pd.DataFrame:
    baseline = first_record_per_person(df)
    baseline.to_csv(TABLE_DIR / "baseline_first_record_analysis_frame.csv", index=False, encoding="utf-8-sig")

    columns = ["Overall"] + MAIN_GROUP_ORDER
    rows = []
    count_row = {"variable": "N", "type": "count", "p_value": "", "max_smd_vs_pinghe": ""}
    count_row["Overall"] = f"{len(baseline):,}"
    for group in MAIN_GROUP_ORDER:
        count_row[group] = f"{baseline['constitution_group'].eq(group).sum():,}"
    rows.append(count_row)

    for label, col in TABLE1_CONTINUOUS.items():
        row = {
            "variable": label,
            "type": "mean_sd",
            "p_value": format_p(table1_p_value_continuous(baseline, col)),
            "max_smd_vs_pinghe": continuous_smd_vs_reference(baseline, col),
        }
        row["Overall"] = mean_sd_text(baseline[col])
        for group in MAIN_GROUP_ORDER:
            row[group] = mean_sd_text(baseline.loc[baseline["constitution_group"].eq(group), col])
        rows.append(row)

    for label, col in TABLE1_BINARY.items():
        row = {
            "variable": label,
            "type": "n_percent",
            "p_value": format_p(table1_p_value_binary(baseline, col)),
            "max_smd_vs_pinghe": binary_smd_vs_reference(baseline, col),
        }
        row["Overall"] = binary_text(baseline[col])
        for group in MAIN_GROUP_ORDER:
            row[group] = binary_text(baseline.loc[baseline["constitution_group"].eq(group), col])
        rows.append(row)

    table1 = pd.DataFrame(rows)
    table1 = table1[["variable", "type", "Overall"] + MAIN_GROUP_ORDER + ["p_value", "max_smd_vs_pinghe"]]
    table1.to_csv(TABLE_DIR / "table1_baseline_by_constitution_first_record.csv", index=False, encoding="utf-8-sig")

    numeric_rows = []
    for label, col in {**TABLE1_CONTINUOUS, **TABLE1_BINARY}.items():
        for group in ["Overall"] + MAIN_GROUP_ORDER:
            mask = pd.Series(True, index=baseline.index) if group == "Overall" else baseline["constitution_group"].eq(group)
            x = pd.to_numeric(baseline.loc[mask, col], errors="coerce")
            numeric_rows.append(
                {
                    "variable": label,
                    "source_column": col,
                    "group": group,
                    "n_nonmissing": int(x.notna().sum()),
                    "mean_or_proportion": float(x.mean()) if x.notna().any() else np.nan,
                    "sd": float(x.std()) if x.notna().sum() > 1 else np.nan,
                    "median": float(x.median()) if x.notna().any() else np.nan,
                    "q1": float(x.quantile(0.25)) if x.notna().any() else np.nan,
                    "q3": float(x.quantile(0.75)) if x.notna().any() else np.nan,
                }
            )
    pd.DataFrame(numeric_rows).to_csv(TABLE_DIR / "table1_baseline_numeric_long.csv", index=False, encoding="utf-8-sig")
    return table1


def fit_baseline_constitution_glm(data: pd.DataFrame, outcome_col: str) -> dict:
    model_df = data[[outcome_col, "const_target", "age", "gender"]].dropna()
    if len(model_df) < 200 or model_df["const_target"].nunique() < 2 or model_df[outcome_col].nunique() < 2:
        return {}
    y = model_df[outcome_col].astype(float)
    if y.sum() < 30 or (len(y) - y.sum()) < 30:
        return {}
    x = pd.DataFrame(index=model_df.index)
    x["const_target"] = model_df["const_target"].astype(float)
    x["age"] = model_df["age"].astype(float)
    x["male"] = model_df["gender"].astype("string").eq("男").astype(float)
    x = sm.add_constant(x, has_constant="add")
    fit = sm.GLM(y, x.astype(float), family=sm.families.Binomial()).fit(maxiter=100)
    log_or = fit.params["const_target"]
    se = fit.bse["const_target"]
    return {
        "estimate": log_or,
        "ci_low": log_or - 1.96 * se,
        "ci_high": log_or + 1.96 * se,
        "p_value": fit.pvalues["const_target"],
        "or": math.exp(log_or),
        "or_ci_low": math.exp(log_or - 1.96 * se),
        "or_ci_high": math.exp(log_or + 1.96 * se),
        "n": int(len(model_df)),
        "events": int(y.sum()),
    }


def run_disease_constitution_mapping(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = first_record_per_person(df)
    distribution_rows = []
    association_rows = []

    for disease_label, disease_col in DISEASE_OUTCOMES.items():
        if disease_col not in baseline.columns:
            continue
        sub = baseline[baseline[disease_col].notna()].copy()
        if sub.empty:
            continue
        cases = sub[sub[disease_col].eq(1)]
        controls = sub[sub[disease_col].eq(0)]
        if len(cases) < 30:
            continue
        row = {
            "disease_or_marker": disease_label,
            "source_column": disease_col,
            "n_nonmissing": int(len(sub)),
            "n_cases": int(len(cases)),
            "case_prevalence_percent": float(len(cases) / len(sub) * 100),
            "biased_percent_among_cases": float(cases["is_biased"].mean() * 100),
            "phlegm_damp_percent_among_cases": float(cases["phlegm_damp_any"].mean() * 100),
        }
        for group in MAIN_GROUP_ORDER:
            row[f"{group}_percent_among_cases"] = float(cases["constitution_group"].eq(group).mean() * 100)
            row[f"{group}_percent_among_non_cases"] = float(controls["constitution_group"].eq(group).mean() * 100) if len(controls) else np.nan
            row[f"{group}_excess_percent_points"] = (
                row[f"{group}_percent_among_cases"] - row[f"{group}_percent_among_non_cases"]
                if len(controls)
                else np.nan
            )
        biased_groups = [g for g in MAIN_GROUP_ORDER if g != "平和质"]
        top_group = max(biased_groups, key=lambda g: row[f"{g}_percent_among_cases"])
        row["top_biased_constitution_among_cases"] = top_group
        row["top_biased_constitution_percent"] = row[f"{top_group}_percent_among_cases"]
        distribution_rows.append(row)

        for target in COMPARISON_TARGETS:
            model_sub = baseline[baseline["primary_constitution"].isin(["平和质", target])].copy()
            model_sub["const_target"] = model_sub["primary_constitution"].eq(target).astype(float)
            try:
                result = fit_baseline_constitution_glm(model_sub, disease_col)
            except Exception as exc:
                result = {"error": str(exc)}
            if result:
                association_rows.append(
                    {
                        "target_constitution": target,
                        "reference": "平和质",
                        "disease_or_marker": disease_label,
                        "source_column": disease_col,
                        **result,
                    }
                )

    distribution = pd.DataFrame(distribution_rows)
    associations = pd.DataFrame(association_rows)
    if not distribution.empty:
        distribution.to_csv(TABLE_DIR / "disease_to_constitution_distribution_baseline.csv", index=False, encoding="utf-8-sig")
        plot_disease_to_constitution_heatmap(distribution)
    if not associations.empty:
        associations["q_value_all"] = p_adjust_bh(associations["p_value"])
        associations.to_csv(TABLE_DIR / "constitution_to_disease_associations_baseline.csv", index=False, encoding="utf-8-sig")
        plot_constitution_to_disease_heatmap(associations)
    write_disease_dictionary()
    return distribution, associations


def write_disease_dictionary() -> None:
    rows = [
        {
            "variable": label,
            "analysis_column": col,
            "interpretation": "Recorded history or examination abnormality from the community health examination form; coded as present vs absent when the source value was interpretable.",
        }
        for label, col in DISEASE_OUTCOMES.items()
    ]
    rows.extend(
        [
            {
                "variable": "High BP (>=140/90)",
                "analysis_column": "high_bp",
                "interpretation": "Disease-related risk marker, not a physician diagnosis; defined from measured BP.",
            },
            {
                "variable": "FPG >=7.0 mmol/L",
                "analysis_column": "diabetes_fpg",
                "interpretation": "Diabetes-range fasting glucose marker, not a diagnosis by itself.",
            },
            {
                "variable": "Any dyslipidemia",
                "analysis_column": "any_dyslipidemia",
                "interpretation": "Risk marker defined from TC/TG/LDL-C/HDL-C thresholds.",
            },
            {
                "variable": "eGFR <60",
                "analysis_column": "low_egfr",
                "interpretation": "Kidney function risk marker calculated by CKD-EPI 2021 creatinine equation.",
            },
        ]
    )
    pd.DataFrame(rows).to_csv(TABLE_DIR / "disease_marker_definition_dictionary.csv", index=False, encoding="utf-8-sig")


def plot_disease_to_constitution_heatmap(distribution: pd.DataFrame) -> None:
    preferred = [
        "Recorded cerebrovascular disease",
        "Recorded heart disease",
        "Recorded eye disease",
        "Recorded neurological disease",
        "Abnormal ECG",
        "Abnormal abdominal ultrasound",
        "Urine protein trace/positive",
        "Urine glucose trace/positive",
        "Urine occult blood trace/positive",
        "Liver enzyme elevation",
        "Anemia",
        "Kidney impairment or proteinuria",
        "Diabetes-related marker",
        "Cardiometabolic risk cluster",
    ]
    distribution = distribution[distribution["disease_or_marker"].isin(preferred)].copy()
    labels = [x for x in preferred if x in set(distribution["disease_or_marker"])]
    heat = distribution.set_index("disease_or_marker")[[f"{g}_percent_among_cases" for g in MAIN_GROUP_ORDER]]
    heat.columns = MAIN_GROUP_ORDER
    heat = heat.loc[labels]
    fig, ax = plt.subplots(figsize=(9, 8.5), constrained_layout=True)
    sns.heatmap(
        heat,
        cmap="YlGnBu",
        annot=True,
        fmt=".1f",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "% among people with disease/marker"},
        ax=ax,
    )
    ax.set_title("If a person has a disease/marker, what constitution is common?")
    ax.set_xlabel("Constitution group")
    ax.set_ylabel("")
    fig.savefig(FIG_DIR / "figure_9_disease_to_constitution_heatmap.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "figure_9_disease_to_constitution_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_constitution_to_disease_heatmap(associations: pd.DataFrame) -> None:
    selected = associations.copy()
    # Keep the most interpretable and sufficiently common outcomes for a readable manuscript figure.
    preferred = [
        "Recorded cerebrovascular disease",
        "Recorded heart disease",
        "Recorded kidney disease",
        "Recorded eye disease",
        "Abnormal ECG",
        "Abnormal abdominal ultrasound",
        "Urine protein trace/positive",
        "Urine glucose trace/positive",
        "Urine occult blood trace/positive",
        "Liver enzyme elevation",
        "Anemia",
        "Kidney impairment or proteinuria",
        "Diabetes-related marker",
        "Cardiometabolic risk cluster",
    ]
    selected = selected[selected["disease_or_marker"].isin(preferred)]
    heat = selected.pivot_table(
        index="disease_or_marker", columns="target_constitution", values="estimate", aggfunc="first"
    ).reindex(index=preferred, columns=COMPARISON_TARGETS)
    q = selected.pivot_table(
        index="disease_or_marker", columns="target_constitution", values="q_value_all", aggfunc="first"
    ).reindex(index=preferred, columns=COMPARISON_TARGETS)
    annot = heat.copy().astype("object")
    for i in heat.index:
        for j in heat.columns:
            if pd.isna(heat.loc[i, j]):
                annot.loc[i, j] = ""
            else:
                star = "*" if q.loc[i, j] < 0.05 else ""
                annot.loc[i, j] = f"{math.exp(heat.loc[i, j]):.2f}{star}"
    fig, ax = plt.subplots(figsize=(8.5, 7.2), constrained_layout=True)
    sns.heatmap(
        heat,
        cmap="vlag",
        center=0,
        vmin=-0.8,
        vmax=0.8,
        annot=annot,
        fmt="",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "log(OR), adjusted for age/sex"},
        ax=ax,
    )
    ax.set_title("If a person has a constitution, what disease/marker is more likely?")
    ax.set_xlabel("Constitution vs 平和质")
    ax.set_ylabel("")
    fig.savefig(FIG_DIR / "figure_10_constitution_to_disease_heatmap.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "figure_10_constitution_to_disease_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_constitution_trend(df: pd.DataFrame) -> None:
    plot_df = (
        df.dropna(subset=["year", "constitution_group"])
        .groupby(["year", "constitution_group"])
        .size()
        .rename("records")
        .reset_index()
    )
    plot_df["constitution_group"] = pd.Categorical(
        plot_df["constitution_group"], categories=MAIN_GROUP_ORDER, ordered=True
    )
    wide = (
        plot_df.pivot_table(index="year", columns="constitution_group", values="records", aggfunc="sum")
        .fillna(0)
        .reindex(columns=MAIN_GROUP_ORDER)
    )
    pct = wide.div(wide.sum(axis=1), axis=0) * 100

    colors = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#76B7B2", "#8A8A8A"]
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    bottom = np.zeros(len(pct))
    x = pct.index.astype(int).to_numpy()
    for group, color in zip(MAIN_GROUP_ORDER, colors):
        vals = pct[group].to_numpy()
        ax.bar(x, vals, bottom=bottom, label=group, color=color, width=0.72)
        bottom += vals
    ax.set_title("Yearly distribution of primary constitution groups")
    ax.set_ylabel("Percent of records")
    ax.set_xlabel("Examination year")
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    fig.savefig(FIG_DIR / "figure_1_constitution_year_trend.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "figure_1_constitution_year_trend.pdf", bbox_inches="tight")
    plt.close(fig)


def build_transition_matrix(df: pd.DataFrame) -> pd.DataFrame:
    records = df.dropna(subset=["person_id", "exam_date", "constitution_group"]).copy()
    records = records.sort_values(["person_id", "exam_date"])
    pairs = []
    for _, group in records.groupby("person_id", sort=False):
        vals = group["constitution_group"].tolist()
        dates = group["exam_date"].tolist()
        for a, b, d0, d1 in zip(vals, vals[1:], dates, dates[1:]):
            if pd.isna(a) or pd.isna(b):
                continue
            pairs.append(
                {
                    "from_group": a,
                    "to_group": b,
                    "from_date": d0,
                    "to_date": d1,
                    "days": (d1 - d0).days if pd.notna(d0) and pd.notna(d1) else np.nan,
                }
            )
    pair_df = pd.DataFrame(pairs)
    pair_df.to_csv(TABLE_DIR / "adjacent_constitution_transitions_long.csv", index=False, encoding="utf-8-sig")
    matrix = pd.crosstab(pair_df["from_group"], pair_df["to_group"]).reindex(
        index=MAIN_GROUP_ORDER, columns=MAIN_GROUP_ORDER, fill_value=0
    )
    pct = matrix.div(matrix.sum(axis=1).replace(0, np.nan), axis=0) * 100
    pct.to_csv(TABLE_DIR / "adjacent_constitution_transition_matrix_percent.csv", encoding="utf-8-sig")
    matrix.to_csv(TABLE_DIR / "adjacent_constitution_transition_matrix_counts.csv", encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)
    sns.heatmap(
        pct,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        cbar_kws={"label": "% of next observed records"},
        linewidths=0.4,
        linecolor="white",
        ax=ax,
    )
    ax.set_title("Adjacent constitution-state transition matrix")
    ax.set_xlabel("Next observed group")
    ax.set_ylabel("Current group")
    fig.savefig(FIG_DIR / "figure_2_transition_matrix.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "figure_2_transition_matrix.pdf", bbox_inches="tight")
    plt.close(fig)
    return pair_df


def build_lagged_frame(df: pd.DataFrame, min_days: int = 180, max_days: int = 730) -> pd.DataFrame:
    lagged_outcome_cols = list(dict.fromkeys(list(LAGGED_OUTCOMES.values()) + list(LAGGED_DISEASE_OUTCOMES.values())))
    keep_cols = (
        [
            "person_id",
            "exam_date",
            "year",
            "primary_constitution",
            "constitution_group",
            "gender",
            "female",
        ]
        + list(dict.fromkeys(["age"] + MODEL_NUMERIC_FEATURES + lagged_outcome_cols))
    )
    records = df[keep_cols].dropna(subset=["person_id", "exam_date", "constitution_group"]).copy()
    records = records.sort_values(["person_id", "exam_date"])
    rows = []
    for _, group in records.groupby("person_id", sort=False):
        if len(group) < 2:
            continue
        current = group.iloc[:-1].reset_index(drop=True)
        nxt = group.iloc[1:].reset_index(drop=True)
        for i in range(len(current)):
            days = (nxt.loc[i, "exam_date"] - current.loc[i, "exam_date"]).days
            if not (min_days <= days <= max_days):
                continue
            row = {
                "person_id": current.loc[i, "person_id"],
                "current_date": current.loc[i, "exam_date"],
                "next_date": nxt.loc[i, "exam_date"],
                "followup_days": days,
                "current_year": current.loc[i, "year"],
                "current_primary_constitution": current.loc[i, "primary_constitution"],
                "next_primary_constitution": nxt.loc[i, "primary_constitution"],
                "current_group": current.loc[i, "constitution_group"],
                "next_group": nxt.loc[i, "constitution_group"],
                "gender": current.loc[i, "gender"],
                "female": current.loc[i, "female"],
            }
            for col in ["age"] + MODEL_NUMERIC_FEATURES:
                row[f"current_{col}"] = current.loc[i, col] if col in current.columns else np.nan
            for col in lagged_outcome_cols:
                row[f"current_{col}"] = current.loc[i, col] if col in current.columns else np.nan
                row[f"next_{col}"] = nxt.loc[i, col] if col in nxt.columns else np.nan
            rows.append(row)
    lagged = pd.DataFrame(rows)
    lagged.to_csv(TABLE_DIR / "lagged_adjacent_annual_records.csv", index=False, encoding="utf-8-sig")
    if not lagged.empty:
        followup_summary = lagged["followup_days"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
        followup_summary.to_csv(TABLE_DIR / "lagged_followup_days_summary.csv", encoding="utf-8-sig")
    return lagged


def lagged_design_matrix(data: pd.DataFrame, outcome_col: str, include_current_status: bool) -> pd.DataFrame:
    x = pd.DataFrame(index=data.index)
    x["const_target"] = data["const_target"].astype(float)
    if include_current_status:
        x["current_status"] = data[f"current_{outcome_col}"].astype(float)
    x["age"] = data["current_age"].astype(float)
    x["male"] = data["gender"].astype("string").eq("男").astype(float)
    x["followup_years"] = data["followup_days"].astype(float) / 365.25
    years = pd.get_dummies(data["current_year"].astype("Int64").astype("string"), prefix="year", drop_first=True)
    x = pd.concat([x, years], axis=1)
    x = sm.add_constant(x, has_constant="add")
    return x.astype(float)


def fit_lagged_glm(data: pd.DataFrame, outcome_col: str, include_current_status: bool) -> dict:
    required = [
        f"next_{outcome_col}",
        "const_target",
        "current_age",
        "gender",
        "current_year",
        "followup_days",
        "person_id",
    ]
    if include_current_status:
        required.append(f"current_{outcome_col}")
    model_df = data[required].dropna()
    if len(model_df) < 200 or model_df["const_target"].nunique() < 2 or model_df[f"next_{outcome_col}"].nunique() < 2:
        return {}
    y = model_df[f"next_{outcome_col}"].astype(float)
    if y.sum() < 30 or (len(y) - y.sum()) < 30:
        return {}
    x = lagged_design_matrix(model_df, outcome_col, include_current_status)
    fit = sm.GLM(y, x, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": model_df["person_id"]}, maxiter=100
    )
    log_or = fit.params["const_target"]
    se = fit.bse["const_target"]
    return {
        "estimate": log_or,
        "ci_low": log_or - 1.96 * se,
        "ci_high": log_or + 1.96 * se,
        "p_value": fit.pvalues["const_target"],
        "or": math.exp(log_or),
        "or_ci_low": math.exp(log_or - 1.96 * se),
        "or_ci_high": math.exp(log_or + 1.96 * se),
        "n": int(len(model_df)),
        "events_next": int(y.sum()),
    }


def run_lagged_risk_models(lagged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if lagged.empty:
        return pd.DataFrame()
    for target in COMPARISON_TARGETS:
        sub = lagged[lagged["current_primary_constitution"].isin(["平和质", target])].copy()
        sub["const_target"] = sub["current_primary_constitution"].eq(target).astype(float)
        for outcome_label, outcome_col in LAGGED_OUTCOMES.items():
            try:
                result = fit_lagged_glm(sub, outcome_col, include_current_status=True)
            except Exception as exc:
                result = {"error": str(exc)}
            if result:
                rows.append(
                    {
                        "target_constitution": target,
                        "reference": "平和质",
                        "outcome": outcome_label,
                        "outcome_col": outcome_col,
                        "model": "next_status_adjusted_for_current_status",
                        **result,
                    }
                )

            incident_sub = sub[sub[f"current_{outcome_col}"].eq(0)].copy()
            try:
                result = fit_lagged_glm(incident_sub, outcome_col, include_current_status=False)
            except Exception as exc:
                result = {"error": str(exc)}
            if result:
                rows.append(
                    {
                        "target_constitution": target,
                        "reference": "平和质",
                        "outcome": outcome_label,
                        "outcome_col": outcome_col,
                        "model": "incident_among_currently_normal",
                        **result,
                    }
                )
    results = pd.DataFrame(rows)
    if not results.empty:
        results["q_value_all"] = p_adjust_bh(results["p_value"])
        results["q_value_by_model"] = results.groupby("model", group_keys=False)["p_value"].apply(p_adjust_bh)
        results.to_csv(TABLE_DIR / "lagged_constitution_future_risk_models.csv", index=False, encoding="utf-8-sig")
        plot_lagged_risk_heatmap(results)
    return results


def run_lagged_disease_risk_models(lagged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if lagged.empty:
        return pd.DataFrame()
    for target in COMPARISON_TARGETS:
        sub = lagged[lagged["current_primary_constitution"].isin(["平和质", target])].copy()
        sub["const_target"] = sub["current_primary_constitution"].eq(target).astype(float)
        for disease_label, disease_col in LAGGED_DISEASE_OUTCOMES.items():
            if f"next_{disease_col}" not in sub.columns:
                continue
            try:
                result = fit_lagged_glm(sub, disease_col, include_current_status=True)
            except Exception as exc:
                result = {"error": str(exc)}
            if result:
                rows.append(
                    {
                        "target_constitution": target,
                        "reference": "平和质",
                        "disease_or_marker": disease_label,
                        "source_column": disease_col,
                        "model": "next_status_adjusted_for_current_status",
                        **result,
                    }
                )
    results = pd.DataFrame(rows)
    if not results.empty:
        results["q_value_all"] = p_adjust_bh(results["p_value"])
        results.to_csv(TABLE_DIR / "lagged_constitution_future_disease_risk_models.csv", index=False, encoding="utf-8-sig")
        plot_lagged_disease_risk_heatmap(results)
    return results


def plot_lagged_disease_risk_heatmap(results: pd.DataFrame) -> None:
    preferred = [
        "Recorded cerebrovascular disease",
        "Recorded heart disease",
        "Recorded kidney disease",
        "Recorded eye disease",
        "Abnormal ECG",
        "Abnormal abdominal ultrasound",
        "Urine protein trace/positive",
        "Urine glucose trace/positive",
        "Urine occult blood trace/positive",
        "Liver enzyme elevation",
        "Anemia",
        "Kidney impairment or proteinuria",
        "Diabetes-related marker",
        "Cardiometabolic risk cluster",
    ]
    selected = results[results["disease_or_marker"].isin(preferred)].copy()
    heat = selected.pivot_table(
        index="disease_or_marker", columns="target_constitution", values="estimate", aggfunc="first"
    ).reindex(index=preferred, columns=COMPARISON_TARGETS)
    q = selected.pivot_table(
        index="disease_or_marker", columns="target_constitution", values="q_value_all", aggfunc="first"
    ).reindex(index=preferred, columns=COMPARISON_TARGETS)
    annot = heat.copy().astype("object")
    for i in heat.index:
        for j in heat.columns:
            if pd.isna(heat.loc[i, j]):
                annot.loc[i, j] = ""
            else:
                star = "*" if q.loc[i, j] < 0.05 else ""
                annot.loc[i, j] = f"{math.exp(heat.loc[i, j]):.2f}{star}"
    fig, ax = plt.subplots(figsize=(8.5, 7.2), constrained_layout=True)
    sns.heatmap(
        heat,
        cmap="vlag",
        center=0,
        vmin=-0.7,
        vmax=0.7,
        annot=annot,
        fmt="",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "log(OR), adjusted for current status/age/sex/year"},
        ax=ax,
    )
    ax.set_title("Current constitution and next-visit disease/marker risk")
    ax.set_xlabel("Current constitution vs 平和质")
    ax.set_ylabel("")
    fig.savefig(FIG_DIR / "figure_11_lagged_future_disease_risk_heatmap.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "figure_11_lagged_future_disease_risk_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_lagged_risk_heatmap(results: pd.DataFrame) -> None:
    selected = results[results["model"].eq("next_status_adjusted_for_current_status")].copy()
    order = list(LAGGED_OUTCOMES.keys())
    heat = selected.pivot_table(
        index="outcome", columns="target_constitution", values="estimate", aggfunc="first"
    ).reindex(index=order, columns=COMPARISON_TARGETS)
    q = selected.pivot_table(index="outcome", columns="target_constitution", values="q_value_all", aggfunc="first").reindex(
        index=order, columns=COMPARISON_TARGETS
    )
    annot = heat.copy().astype("object")
    for i in heat.index:
        for j in heat.columns:
            if pd.isna(heat.loc[i, j]):
                annot.loc[i, j] = ""
            else:
                star = "*" if q.loc[i, j] < 0.05 else ""
                annot.loc[i, j] = f"{math.exp(heat.loc[i, j]):.2f}{star}"
    fig, ax = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)
    sns.heatmap(
        heat,
        cmap="vlag",
        center=0,
        vmin=-0.6,
        vmax=0.6,
        annot=annot,
        fmt="",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "log(OR), adjusted for baseline status/age/sex/year"},
        ax=ax,
    )
    ax.set_title("Current constitution and next-visit risk")
    ax.set_xlabel("Current constitution vs 平和质")
    ax.set_ylabel("")
    fig.savefig(FIG_DIR / "figure_7_lagged_future_risk_heatmap.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "figure_7_lagged_future_risk_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def transition_predictor_design(data: pd.DataFrame, predictor_col: str) -> tuple[pd.DataFrame, pd.Series]:
    model_df = data[["transition_outcome", "person_id", "current_year", "gender", "followup_days", predictor_col]].dropna()
    if len(model_df) < 200 or model_df["transition_outcome"].nunique() < 2:
        return pd.DataFrame(), pd.Series(dtype=float)
    pred = pd.to_numeric(model_df[predictor_col], errors="coerce")
    sd = pred.std()
    if not sd or pd.isna(sd):
        return pd.DataFrame(), pd.Series(dtype=float)
    x = pd.DataFrame(index=model_df.index)
    x["predictor_z"] = (pred - pred.mean()) / sd
    if predictor_col != "current_age":
        x["age"] = model_df["current_age"].astype(float) if "current_age" in model_df.columns else np.nan
    x["male"] = model_df["gender"].astype("string").eq("男").astype(float)
    x["followup_years"] = model_df["followup_days"].astype(float) / 365.25
    years = pd.get_dummies(model_df["current_year"].astype("Int64").astype("string"), prefix="year", drop_first=True)
    x = pd.concat([x, years], axis=1)
    x = sm.add_constant(x, has_constant="add").astype(float)
    y = model_df["transition_outcome"].astype(float)
    return x, y


def fit_transition_predictor(data: pd.DataFrame, predictor_col: str) -> dict:
    cols = ["transition_outcome", "person_id", "current_year", "gender", "followup_days", predictor_col]
    if predictor_col != "current_age":
        cols.append("current_age")
    model_df = data[cols].dropna()
    if len(model_df) < 200 or model_df["transition_outcome"].nunique() < 2:
        return {}
    pred = pd.to_numeric(model_df[predictor_col], errors="coerce")
    sd = pred.std()
    if not sd or pd.isna(sd):
        return {}
    x = pd.DataFrame(index=model_df.index)
    x["predictor_z"] = (pred - pred.mean()) / sd
    if predictor_col != "current_age":
        x["age"] = model_df["current_age"].astype(float)
    x["male"] = model_df["gender"].astype("string").eq("男").astype(float)
    x["followup_years"] = model_df["followup_days"].astype(float) / 365.25
    years = pd.get_dummies(model_df["current_year"].astype("Int64").astype("string"), prefix="year", drop_first=True)
    x = pd.concat([x, years], axis=1)
    x = sm.add_constant(x, has_constant="add").astype(float)
    y = model_df["transition_outcome"].astype(float)
    fit = sm.GLM(y, x, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": model_df["person_id"]}, maxiter=100
    )
    log_or = fit.params["predictor_z"]
    se = fit.bse["predictor_z"]
    return {
        "estimate": log_or,
        "ci_low": log_or - 1.96 * se,
        "ci_high": log_or + 1.96 * se,
        "p_value": fit.pvalues["predictor_z"],
        "or_per_sd": math.exp(log_or),
        "or_ci_low": math.exp(log_or - 1.96 * se),
        "or_ci_high": math.exp(log_or + 1.96 * se),
        "n": int(len(model_df)),
        "events": int(y.sum()),
    }


def run_transition_predictor_models(lagged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if lagged.empty:
        return pd.DataFrame()
    tasks = [
        (
            "Balanced to biased",
            lagged[lagged["current_group"].eq("平和质")].copy(),
            lambda x: x["next_group"].ne("平和质").astype(float),
        ),
        (
            "Phlegm-dampness to balanced",
            lagged[lagged["current_group"].eq("痰湿质")].copy(),
            lambda x: x["next_group"].eq("平和质").astype(float),
        ),
    ]
    for task_name, sub, outcome_fn in tasks:
        if sub.empty:
            continue
        sub["transition_outcome"] = outcome_fn(sub)
        for predictor_label, predictor_col in TRANSITION_PREDICTORS.items():
            current_col = f"current_{predictor_col}"
            if current_col not in sub.columns:
                continue
            try:
                result = fit_transition_predictor(sub, current_col)
            except Exception as exc:
                result = {"error": str(exc)}
            if result:
                rows.append(
                    {
                        "transition_task": task_name,
                        "predictor": predictor_label,
                        "predictor_col": predictor_col,
                        **result,
                    }
                )
    results = pd.DataFrame(rows)
    if not results.empty:
        results["q_value"] = results.groupby("transition_task", group_keys=False)["p_value"].apply(p_adjust_bh)
        results.to_csv(TABLE_DIR / "transition_predictor_models.csv", index=False, encoding="utf-8-sig")
        plot_transition_predictors(results)
    return results


def plot_transition_predictors(results: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for ax, task in zip(axes, ["Balanced to biased", "Phlegm-dampness to balanced"]):
        sub = results[results["transition_task"].eq(task)].copy()
        if sub.empty:
            ax.axis("off")
            continue
        sub = sub.reindex(sub["estimate"].abs().sort_values(ascending=False).index).head(10)
        sub = sub.sort_values("estimate")
        colors = np.where(sub["estimate"].ge(0), "#C44E52", "#4C72B0")
        ax.barh(sub["predictor"], sub["estimate"], color=colors)
        ax.axvline(0, color="#333333", lw=0.8)
        ax.set_title(task)
        ax.set_xlabel("log(OR) per SD")
        ax.set_ylabel("")
    fig.savefig(FIG_DIR / "figure_8_transition_predictors.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "figure_8_transition_predictors.pdf", bbox_inches="tight")
    plt.close(fig)


def design_matrix_for_adjustment(data: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=data.index)
    x["const_target"] = data["const_target"].astype(float)
    x["age"] = data["age"].astype(float)
    x["male"] = data["gender"].astype("string").eq("男").astype(float)
    years = pd.get_dummies(data["year"].astype("Int64").astype("string"), prefix="year", drop_first=True)
    x = pd.concat([x, years], axis=1)
    x = sm.add_constant(x, has_constant="add")
    return x.astype(float)


def fit_cluster_ols(data: pd.DataFrame, outcome_col: str) -> dict:
    model_df = data[[outcome_col, "const_target", "age", "gender", "year", "person_id"]].dropna()
    if len(model_df) < 200 or model_df["const_target"].nunique() < 2:
        return {}
    y_raw = model_df[outcome_col].astype(float)
    if y_raw.std() == 0 or pd.isna(y_raw.std()):
        return {}
    y = (y_raw - y_raw.mean()) / y_raw.std()
    x = design_matrix_for_adjustment(model_df)
    fit = sm.OLS(y, x).fit(cov_type="cluster", cov_kwds={"groups": model_df["person_id"]})
    est = fit.params["const_target"]
    se = fit.bse["const_target"]
    return {
        "estimate": est,
        "ci_low": est - 1.96 * se,
        "ci_high": est + 1.96 * se,
        "p_value": fit.pvalues["const_target"],
        "n": int(len(model_df)),
    }


def fit_cluster_glm(data: pd.DataFrame, outcome_col: str) -> dict:
    model_df = data[[outcome_col, "const_target", "age", "gender", "year", "person_id"]].dropna()
    if len(model_df) < 200 or model_df["const_target"].nunique() < 2 or model_df[outcome_col].nunique() < 2:
        return {}
    y = model_df[outcome_col].astype(float)
    x = design_matrix_for_adjustment(model_df)
    fit = sm.GLM(y, x, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": model_df["person_id"]}, maxiter=100
    )
    log_or = fit.params["const_target"]
    se = fit.bse["const_target"]
    return {
        "estimate": log_or,
        "ci_low": log_or - 1.96 * se,
        "ci_high": log_or + 1.96 * se,
        "p_value": fit.pvalues["const_target"],
        "or": math.exp(log_or),
        "or_ci_low": math.exp(log_or - 1.96 * se),
        "or_ci_high": math.exp(log_or + 1.96 * se),
        "n": int(len(model_df)),
    }


def run_phenome_association(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in COMPARISON_TARGETS:
        sub = df[df["primary_constitution"].isin(["平和质", target])].copy()
        sub["const_target"] = sub["primary_constitution"].eq(target).astype(float)
        for outcome_label, outcome_col in CONTINUOUS_OUTCOMES.items():
            if outcome_col not in sub.columns:
                continue
            try:
                result = fit_cluster_ols(sub, outcome_col)
            except Exception as exc:
                result = {"error": str(exc)}
            if result:
                rows.append(
                    {
                        "target_constitution": target,
                        "reference": "平和质",
                        "outcome": outcome_label,
                        "outcome_col": outcome_col,
                        "outcome_type": "continuous_z",
                        **result,
                    }
                )
        for outcome_label, outcome_col in BINARY_OUTCOMES.items():
            if outcome_col not in sub.columns:
                continue
            try:
                result = fit_cluster_glm(sub, outcome_col)
            except Exception as exc:
                result = {"error": str(exc)}
            if result:
                rows.append(
                    {
                        "target_constitution": target,
                        "reference": "平和质",
                        "outcome": outcome_label,
                        "outcome_col": outcome_col,
                        "outcome_type": "binary_log_or",
                        **result,
                    }
                )
    assoc = pd.DataFrame(rows)
    if not assoc.empty:
        assoc["q_value_all"] = p_adjust_bh(assoc["p_value"])
        assoc["q_value_by_type"] = assoc.groupby("outcome_type", group_keys=False)["p_value"].apply(p_adjust_bh)
        assoc.to_csv(TABLE_DIR / "constitution_phenome_associations.csv", index=False, encoding="utf-8-sig")
        plot_phenome_heatmap(assoc)
    return assoc


def plot_phenome_heatmap(assoc: pd.DataFrame) -> None:
    selected = assoc.copy()
    selected["effect_for_heatmap"] = selected["estimate"]
    order = [
        "BMI",
        "Waist circumference",
        "Systolic BP",
        "Diastolic BP",
        "Fasting glucose",
        "Triglycerides",
        "HDL-C",
        "LDL-C",
        "Total cholesterol",
        "eGFR",
        "Creatinine",
        "Overweight (BMI>=24)",
        "Obesity (BMI>=28)",
        "High BP (>=140/90)",
        "FPG >=6.1 mmol/L",
        "FPG >=7.0 mmol/L",
        "High TG",
        "High LDL-C",
        "Low HDL-C",
        "eGFR <60",
        "Any dyslipidemia",
    ]
    selected = selected[selected["outcome"].isin(order)]
    heat = selected.pivot_table(
        index="outcome", columns="target_constitution", values="effect_for_heatmap", aggfunc="first"
    ).reindex(index=order, columns=COMPARISON_TARGETS)
    q = selected.pivot_table(index="outcome", columns="target_constitution", values="q_value_all", aggfunc="first").reindex(
        index=order, columns=COMPARISON_TARGETS
    )
    annot = heat.copy().astype("object")
    for i in heat.index:
        for j in heat.columns:
            if pd.isna(heat.loc[i, j]):
                annot.loc[i, j] = ""
            else:
                star = "*" if q.loc[i, j] < 0.05 else ""
                annot.loc[i, j] = f"{heat.loc[i, j]:.2f}{star}"

    fig, ax = plt.subplots(figsize=(7.5, 8.2), constrained_layout=True)
    sns.heatmap(
        heat,
        cmap="vlag",
        center=0,
        vmin=-0.8,
        vmax=0.8,
        annot=annot,
        fmt="",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Std. beta or log(OR), adjusted for age/sex/year"},
        ax=ax,
    )
    ax.set_title("Constitution-phenome association atlas")
    ax.set_xlabel("Target constitution vs 平和质")
    ax.set_ylabel("")
    fig.savefig(FIG_DIR / "figure_3_constitution_phenome_heatmap.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "figure_3_constitution_phenome_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def make_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=20)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, MODEL_NUMERIC_FEATURES),
            ("cat", cat_pipe, MODEL_CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


@dataclass
class ModelResult:
    task_name: str
    model_name: str
    pipeline: Pipeline
    threshold: float
    metrics: dict
    test_prob: np.ndarray
    test_y: np.ndarray
    val_prob: np.ndarray
    val_y: np.ndarray


def best_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    thresholds = np.linspace(0.05, 0.95, 181)
    scores = [balanced_accuracy_score(y_true, prob >= t) for t in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def evaluate_binary(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "event_rate": float(np.mean(y_true)),
        "threshold": float(threshold),
        "auc": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) == 2 else np.nan,
        "pr_auc": float(average_precision_score(y_true, prob)) if len(np.unique(y_true)) == 2 else np.nan,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "brier": float(brier_score_loss(y_true, prob)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def get_model_data(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    cols = ["person_id", "year", target_col] + MODEL_NUMERIC_FEATURES + MODEL_CATEGORICAL_FEATURES
    data = df[cols].copy()
    data = data.dropna(subset=["year", target_col])
    for col in MODEL_NUMERIC_FEATURES:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    for col in MODEL_CATEGORICAL_FEATURES:
        data[col] = data[col].astype("object")
        data.loc[pd.isna(data[col]), col] = np.nan
    y = data[target_col].astype(int)
    x = data.drop(columns=[target_col])
    return x, y


def run_single_task_models(df: pd.DataFrame, task_name: str, target_col: str) -> list[ModelResult]:
    x, y = get_model_data(df, target_col)
    train_mask = x["year"].le(2023)
    val_mask = x["year"].eq(2024)
    test_mask = x["year"].ge(2025)

    x_train = x.loc[train_mask].drop(columns=["person_id", "year"])
    y_train = y.loc[train_mask]
    x_val = x.loc[val_mask].drop(columns=["person_id", "year"])
    y_val = y.loc[val_mask]
    x_test = x.loc[test_mask].drop(columns=["person_id", "year"])
    y_test = y.loc[test_mask]

    results: list[ModelResult] = []
    model_specs = [
        (
            "Logistic regression",
            Pipeline(
                [
                    ("preprocess", make_preprocessor(scale_numeric=True)),
                    (
                        "clf",
                        LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"),
                    ),
                ]
            ),
        ),
        (
            "XGBoost",
            Pipeline(
                [
                    ("preprocess", make_preprocessor(scale_numeric=False)),
                    (
                        "clf",
                        XGBClassifier(
                            n_estimators=350,
                            max_depth=3,
                            learning_rate=0.045,
                            subsample=0.88,
                            colsample_bytree=0.88,
                            reg_lambda=1.5,
                            objective="binary:logistic",
                            eval_metric="logloss",
                            tree_method="hist",
                            random_state=20260426,
                            n_jobs=4,
                            scale_pos_weight=float((y_train == 0).sum() / max((y_train == 1).sum(), 1)),
                        ),
                    ),
                ]
            ),
        ),
    ]

    for model_name, pipe in model_specs:
        pipe.fit(x_train, y_train)
        val_prob = pipe.predict_proba(x_val)[:, 1]
        threshold = best_threshold(y_val.to_numpy(), val_prob)
        test_prob = pipe.predict_proba(x_test)[:, 1]
        metrics = evaluate_binary(y_test.to_numpy(), test_prob, threshold)
        metrics.update(
            {
                "task": task_name,
                "model": model_name,
                "train_n": int(len(y_train)),
                "val_n": int(len(y_val)),
                "test_n": int(len(y_test)),
                "train_event_rate": float(y_train.mean()),
                "val_event_rate": float(y_val.mean()),
                "test_event_rate": float(y_test.mean()),
            }
        )
        results.append(
            ModelResult(
                task_name=task_name,
                model_name=model_name,
                pipeline=pipe,
                threshold=threshold,
                metrics=metrics,
                test_prob=test_prob,
                test_y=y_test.to_numpy(),
                val_prob=val_prob,
                val_y=y_val.to_numpy(),
            )
        )
    return results


def conformal_summary(y_cal: np.ndarray, p_cal: np.ndarray, y_test: np.ndarray, p_test: np.ndarray, alpha: float = 0.10) -> dict:
    p_true_cal = np.where(y_cal == 1, p_cal, 1 - p_cal)
    scores = 1 - p_true_cal
    q_level = min(1.0, math.ceil((len(scores) + 1) * (1 - alpha)) / len(scores))
    qhat = float(np.quantile(scores, q_level, method="higher"))
    threshold = 1 - qhat
    include_zero = (1 - p_test) >= threshold
    include_one = p_test >= threshold
    set_size = include_zero.astype(int) + include_one.astype(int)
    covered = np.where(y_test == 1, include_one, include_zero)
    return {
        "alpha": alpha,
        "confidence": 1 - alpha,
        "qhat": qhat,
        "probability_threshold_for_inclusion": threshold,
        "empirical_coverage": float(np.mean(covered)),
        "average_set_size": float(np.mean(set_size)),
        "singleton_rate": float(np.mean(set_size == 1)),
        "ambiguous_rate": float(np.mean(set_size == 2)),
        "empty_rate": float(np.mean(set_size == 0)),
    }


def run_prediction_models(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_results: list[ModelResult] = []
    tasks = [
        ("Biased vs balanced constitution", "is_biased"),
        ("Phlegm-dampness vs others", "phlegm_damp_any"),
    ]
    for task_name, target_col in tasks:
        all_results.extend(run_single_task_models(df, task_name, target_col))

    metrics = pd.DataFrame([r.metrics for r in all_results])
    metrics.to_csv(TABLE_DIR / "temporal_validation_model_metrics.csv", index=False, encoding="utf-8-sig")

    plot_roc_curves(all_results)
    plot_decision_curves(all_results)

    conformal_rows = []
    for result in all_results:
        if result.model_name != "XGBoost":
            continue
        row = conformal_summary(result.val_y, result.val_prob, result.test_y, result.test_prob)
        row.update({"task": result.task_name, "model": result.model_name})
        conformal_rows.append(row)
        try:
            write_shap_outputs(result)
        except Exception as exc:
            (OUT_DIR / f"shap_error_{slugify(result.task_name)}.txt").write_text(str(exc), encoding="utf-8")
    conformal = pd.DataFrame(conformal_rows)
    conformal.to_csv(TABLE_DIR / "conformal_prediction_summary.csv", index=False, encoding="utf-8-sig")

    # Sensitivity analysis: remove anthropometric variables that may overlap with
    # the symptom-derived definition of phlegm-dampness constitution.
    no_anthro = df.copy()
    for col in ["height", "weight", "waist", "bmi"]:
        no_anthro[col] = np.nan
    sensitivity_results: list[ModelResult] = []
    for task_name, target_col in tasks:
        sensitivity_results.extend(run_single_task_models(no_anthro, task_name, target_col))
    sensitivity_metrics = pd.DataFrame([r.metrics for r in sensitivity_results])
    sensitivity_metrics["feature_set"] = "No height/weight/BMI/waist"
    sensitivity_metrics.to_csv(
        TABLE_DIR / "temporal_validation_model_metrics_sensitivity_no_anthro.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return metrics, conformal


def plot_roc_curves(results: list[ModelResult]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for ax, task_name in zip(axes, sorted({r.task_name for r in results})):
        subset = [r for r in results if r.task_name == task_name]
        for r in subset:
            fpr, tpr, _ = roc_curve(r.test_y, r.test_prob)
            auc = roc_auc_score(r.test_y, r.test_prob)
            ax.plot(fpr, tpr, lw=1.8, label=f"{r.model_name} AUC={auc:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="#777777", lw=1)
        ax.set_title(task_name)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(frameon=False, loc="lower right")
    fig.savefig(FIG_DIR / "figure_4_temporal_validation_roc.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "figure_4_temporal_validation_roc.pdf", bbox_inches="tight")
    plt.close(fig)


def net_benefit(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> float:
    pred = prob >= threshold
    tp = np.sum((pred == 1) & (y_true == 1))
    fp = np.sum((pred == 1) & (y_true == 0))
    n = len(y_true)
    return tp / n - fp / n * (threshold / (1 - threshold))


def plot_decision_curves(results: list[ModelResult]) -> None:
    thresholds = np.linspace(0.05, 0.60, 56)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for ax, task_name in zip(axes, sorted({r.task_name for r in results})):
        subset = [r for r in results if r.task_name == task_name and r.model_name == "XGBoost"]
        if not subset:
            continue
        r = subset[0]
        prevalence = np.mean(r.test_y)
        treat_all = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))
        ax.plot(thresholds, [net_benefit(r.test_y, r.test_prob, t) for t in thresholds], label="XGBoost")
        ax.plot(thresholds, treat_all, "--", color="#777777", label="Treat all")
        ax.plot(thresholds, np.zeros_like(thresholds), ":", color="#333333", label="Treat none")
        ax.set_title(task_name)
        ax.set_xlabel("Threshold probability")
        ax.set_ylabel("Net benefit")
        ax.legend(frameon=False)
    fig.savefig(FIG_DIR / "figure_5_decision_curve.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "figure_5_decision_curve.pdf", bbox_inches="tight")
    plt.close(fig)


def slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()


def readable_feature_name(name: str) -> str:
    name = name.replace("num__", "").replace("cat__", "")
    replacements = {
        "age": "Age",
        "height": "Height",
        "weight": "Weight",
        "waist": "Waist circumference",
        "bmi": "BMI",
        "sbp_mean": "Systolic BP",
        "dbp_mean": "Diastolic BP",
        "hemoglobin": "Hemoglobin",
        "wbc": "White blood cells",
        "platelet": "Platelets",
        "fpg": "Fasting glucose",
        "alt": "ALT",
        "ast": "AST",
        "tbil": "Total bilirubin",
        "creatinine": "Creatinine",
        "urea": "Urea",
        "egfr": "eGFR",
        "tc": "Total cholesterol",
        "tg": "Triglycerides",
        "ldl": "LDL-C",
        "hdl": "HDL-C",
    }
    for key, value in replacements.items():
        if name == key:
            return value
    return name.replace("_", " = ")


def write_shap_outputs(result: ModelResult) -> None:
    task_slug = slugify(result.task_name)
    # Recreate the exact temporal test input for this task.
    target_col = "is_biased" if "Biased" in result.task_name else "phlegm_damp_any"
    raw_df = pd.read_pickle(OUT_DIR / "analysis_model_frame.pkl")
    x, y = get_model_data(raw_df, target_col)
    test_mask = x["year"].ge(2025)
    x_test = x.loc[test_mask].drop(columns=["person_id", "year"])

    sample_n = min(1500, len(x_test))
    sample = x_test.sample(sample_n, random_state=20260426) if len(x_test) > sample_n else x_test

    preprocess = result.pipeline.named_steps["preprocess"]
    clf = result.pipeline.named_steps["clf"]
    x_trans = preprocess.transform(sample)
    feature_names = preprocess.get_feature_names_out()

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(x_trans)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    mean_abs = np.abs(shap_values).mean(axis=0)
    imp = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .head(30)
    )
    imp["feature_readable"] = imp["feature"].map(readable_feature_name)
    imp.to_csv(TABLE_DIR / f"shap_top_features_{task_slug}.csv", index=False, encoding="utf-8-sig")

    top = imp.head(20).sort_values("mean_abs_shap", ascending=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.8), constrained_layout=True)
    ax.barh(top["feature_readable"], top["mean_abs_shap"], color="#4E79A7")
    ax.set_title(f"SHAP feature importance: {result.task_name}")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_ylabel("")
    fig.savefig(FIG_DIR / f"figure_6_shap_{task_slug}.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"figure_6_shap_{task_slug}.pdf", bbox_inches="tight")
    plt.close(fig)


def write_summary_markdown(
    overview: dict,
    pair_df: pd.DataFrame,
    assoc: pd.DataFrame,
    metrics: pd.DataFrame,
    conformal: pd.DataFrame,
    table1: pd.DataFrame | None = None,
    lagged_risk: pd.DataFrame | None = None,
    transition_models: pd.DataFrame | None = None,
    disease_distribution: pd.DataFrame | None = None,
    disease_associations: pd.DataFrame | None = None,
    lagged_disease_risk: pd.DataFrame | None = None,
) -> None:
    constitution_counts = pd.read_csv(TABLE_DIR / "constitution_group_distribution.csv")
    top_counts = constitution_counts.head(6)
    transition_change_rate = np.nan
    if not pair_df.empty:
        transition_change_rate = float((pair_df["from_group"] != pair_df["to_group"]).mean())

    strong_assoc = pd.DataFrame()
    if not assoc.empty:
        strong_assoc = assoc.sort_values(["q_value_all", "p_value"]).head(12)

    lines = [
        "# DigiConstitution initial analysis summary",
        "",
        "## Dataset",
        f"- Records: {overview['n_rows']:,}",
        f"- Unique persons: {overview['n_unique_persons']:,}",
        f"- Examination dates: {overview['date_min']} to {overview['date_max']}",
        f"- Persons with >=2 records: {overview['persons_with_2plus_records']:,}",
        f"- Persons with >=3 records: {overview['persons_with_3plus_records']:,}",
        f"- Constitution-missing records: {overview['n_constitution_missing']:,}",
        "",
        "## Primary constitution groups",
    ]
    for _, row in top_counts.iterrows():
        lines.append(f"- {row['constitution_group']}: {int(row['records']):,} records ({row['percent']:.1f}%)")

    lines.extend(
        [
            "",
            "## Longitudinal state transitions",
            f"- Adjacent record pairs: {len(pair_df):,}",
            f"- Crude adjacent-group change rate: {transition_change_rate:.1%}" if not np.isnan(transition_change_rate) else "- No transition pairs found.",
        ]
    )

    if table1 is not None and not table1.empty:
        first_count = table1.loc[table1["variable"].eq("N"), "Overall"].iloc[0]
        lines.extend(
            [
                "",
                "## Baseline Table 1",
                f"- Baseline table uses the first available record per person: n={first_count}.",
                "- Continuous variables are summarized as mean (SD); binary variables as n (%).",
            ]
        )

    if lagged_risk is not None and not lagged_risk.empty:
        lagged_top = (
            lagged_risk[lagged_risk["model"].eq("next_status_adjusted_for_current_status")]
            .sort_values(["q_value_all", "p_value"])
            .head(10)
        )
        lines.extend(["", "## Lagged next-visit risk signals"])
        for _, row in lagged_top.iterrows():
            lines.append(
                "- "
                f"Current {row['target_constitution']} vs 平和质 -> next {row['outcome']}: "
                f"OR={row['or']:.2f} ({row['or_ci_low']:.2f}-{row['or_ci_high']:.2f}), "
                f"q={format_p(row['q_value_all'])}, adjusted for current status"
            )

    if disease_distribution is not None and not disease_distribution.empty:
        main_disease_labels = {
            "Recorded cerebrovascular disease",
            "Recorded heart disease",
            "Recorded eye disease",
            "Recorded neurological disease",
            "Abnormal ECG",
            "Abnormal abdominal ultrasound",
            "Urine protein trace/positive",
            "Urine glucose trace/positive",
            "Urine occult blood trace/positive",
            "Liver enzyme elevation",
            "Anemia",
            "Kidney impairment or proteinuria",
            "Diabetes-related marker",
            "Cardiometabolic risk cluster",
        }
        disease_top = (
            disease_distribution[disease_distribution["disease_or_marker"].isin(main_disease_labels)]
            .sort_values("phlegm_damp_percent_among_cases", ascending=False)
            .head(8)
        )
        lines.extend(["", "## Disease-to-constitution profile"])
        for _, row in disease_top.iterrows():
            lines.append(
                "- "
                f"{row['disease_or_marker']}: among cases, "
                f"phlegm-dampness={row['phlegm_damp_percent_among_cases']:.1f}%, "
                f"biased constitution={row['biased_percent_among_cases']:.1f}%, "
                f"top biased group={row['top_biased_constitution_among_cases']}"
            )

    if disease_associations is not None and not disease_associations.empty:
        disease_assoc_top = (
            disease_associations[disease_associations["disease_or_marker"].isin(main_disease_labels)]
            .sort_values(["q_value_all", "p_value"])
            .head(10)
        )
        lines.extend(["", "## Constitution-to-disease baseline signals"])
        for _, row in disease_assoc_top.iterrows():
            lines.append(
                "- "
                f"{row['target_constitution']} vs 平和质 -> {row['disease_or_marker']}: "
                f"OR={row['or']:.2f} ({row['or_ci_low']:.2f}-{row['or_ci_high']:.2f}), "
                f"q={format_p(row['q_value_all'])}, adjusted for age/sex"
            )

    if lagged_disease_risk is not None and not lagged_disease_risk.empty:
        lagged_disease_top = lagged_disease_risk.sort_values(["q_value_all", "p_value"]).head(10)
        lines.extend(["", "## Constitution-to-future disease/marker signals"])
        for _, row in lagged_disease_top.iterrows():
            lines.append(
                "- "
                f"Current {row['target_constitution']} vs 平和质 -> next {row['disease_or_marker']}: "
                f"OR={row['or']:.2f} ({row['or_ci_low']:.2f}-{row['or_ci_high']:.2f}), "
                f"q={format_p(row['q_value_all'])}, adjusted for current status"
            )

    if transition_models is not None and not transition_models.empty:
        transition_top = transition_models.sort_values(["q_value", "p_value"]).head(10)
        lines.extend(["", "## Transition predictor signals"])
        for _, row in transition_top.iterrows():
            lines.append(
                "- "
                f"{row['transition_task']}, {row['predictor']}: "
                f"OR per SD={row['or_per_sd']:.2f} ({row['or_ci_low']:.2f}-{row['or_ci_high']:.2f}), "
                f"q={format_p(row['q_value'])}"
            )

    lines.extend(["", "## Temporal prediction performance"])
    for _, row in metrics.sort_values(["task", "model"]).iterrows():
        lines.append(
            "- "
            f"{row['task']} | {row['model']}: "
            f"AUC={row['auc']:.3f}, PR-AUC={row['pr_auc']:.3f}, "
            f"balanced accuracy={row['balanced_accuracy']:.3f}, "
            f"F1={row['f1']:.3f}, Brier={row['brier']:.3f}"
        )

    sensitivity_path = TABLE_DIR / "temporal_validation_model_metrics_sensitivity_no_anthro.csv"
    if sensitivity_path.exists():
        sensitivity = pd.read_csv(sensitivity_path)
        lines.extend(["", "## Sensitivity: no height/weight/BMI/waist"])
        for _, row in sensitivity.sort_values(["task", "model"]).iterrows():
            lines.append(
                "- "
                f"{row['task']} | {row['model']}: "
                f"AUC={row['auc']:.3f}, PR-AUC={row['pr_auc']:.3f}, "
                f"balanced accuracy={row['balanced_accuracy']:.3f}, "
                f"F1={row['f1']:.3f}"
            )

    lines.extend(["", "## Conformal prediction, XGBoost, 90% target coverage"])
    for _, row in conformal.iterrows():
        lines.append(
            "- "
            f"{row['task']}: empirical coverage={row['empirical_coverage']:.3f}, "
            f"ambiguous-set rate={row['ambiguous_rate']:.3f}, "
            f"average set size={row['average_set_size']:.3f}"
        )

    if not strong_assoc.empty:
        lines.extend(["", "## Strongest adjusted constitution-phenome signals"])
        for _, row in strong_assoc.iterrows():
            if row["outcome_type"] == "binary_log_or":
                effect = f"OR={row['or']:.2f} ({row['or_ci_low']:.2f}-{row['or_ci_high']:.2f})"
            else:
                effect = f"std beta={row['estimate']:.2f} ({row['ci_low']:.2f}-{row['ci_high']:.2f})"
            lines.append(
                f"- {row['target_constitution']} vs 平和质, {row['outcome']}: {effect}, q={format_p(row['q_value_all'])}"
            )

    lines.extend(
        [
            "",
            "## Notes for manuscript interpretation",
            "- These are feasibility-stage results. Threshold definitions and final covariate sets should be locked before manuscript tables are finalized.",
            "- The main predictive models intentionally exclude the 33 TCM questionnaire items to avoid circular prediction.",
            "- Disease tables separate recorded disease history from disease-related examination markers; marker outcomes should not be interpreted as physician diagnoses.",
            "- A no-anthropometrics sensitivity model is included because phlegm-dampness scoring can overlap conceptually with body size/adiposity items.",
            "- The split-conformal summaries are temporal stress tests, not guaranteed-coverage conformal inference, because 2025-2026 shows visible distribution shift from 2017-2024.",
            "- Temporal validation uses 2017-2023 for training, 2024 for threshold selection/calibration, and 2025-2026 for testing.",
        ]
    )
    (OUT_DIR / "analysis_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_notebook() -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"]["kernelspec"] = {
        "display_name": "Python (.venv)",
        "language": "python",
        "name": "python3",
    }
    nb["metadata"]["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    cells = [
        nbf.v4.new_markdown_cell(
            "# DigiConstitution Initial Analysis\n\n"
            "This notebook documents the first reproducible analysis pass for translating TCM constitution into digital health states. "
            "The heavy computations are implemented in `scripts/initial_constitution_analysis.py`; this notebook is a compact, rerunnable analysis map."
        ),
        nbf.v4.new_markdown_cell(
            "## Analysis objectives\n\n"
            "1. Clean constitution labels and routine examination variables.\n"
            "2. Describe longitudinal follow-up structure and constitution-state transitions.\n"
            "3. Build a constitution-phenome association atlas.\n"
            "4. Train temporally validated prediction models without using the 33 TCM questionnaire items.\n"
            "5. Add uncertainty-aware screening summaries through conformal prediction."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "ROOT = Path.cwd()\n"
            "OUT = ROOT / 'outputs' / 'initial_analysis'\n"
            "TABLES = OUT / 'tables'\n"
            "FIGS = OUT / 'figures'\n"
            "print(OUT)"
        ),
        nbf.v4.new_markdown_cell("## Re-run the full analysis"),
        nbf.v4.new_code_cell(
            "# Uncomment to regenerate all outputs from the source Excel workbook.\n"
            "# %run scripts/initial_constitution_analysis.py"
        ),
        nbf.v4.new_markdown_cell("## Dataset overview"),
        nbf.v4.new_code_cell(
            "import json\n"
            "overview = json.loads((OUT / 'data_overview.json').read_text(encoding='utf-8'))\n"
            "overview"
        ),
        nbf.v4.new_markdown_cell("## Constitution distribution"),
        nbf.v4.new_code_cell(
            "pd.read_csv(TABLES / 'constitution_group_distribution.csv').head(10)"
        ),
        nbf.v4.new_markdown_cell("## Baseline Table 1"),
        nbf.v4.new_code_cell(
            "table1 = pd.read_csv(TABLES / 'table1_baseline_by_constitution_first_record.csv')\n"
            "table1.head(25)"
        ),
        nbf.v4.new_markdown_cell("## Disease-to-constitution mapping"),
        nbf.v4.new_code_cell(
            "d2c = pd.read_csv(TABLES / 'disease_to_constitution_distribution_baseline.csv')\n"
            "d2c[['disease_or_marker','n_cases','biased_percent_among_cases','phlegm_damp_percent_among_cases','top_biased_constitution_among_cases']].head(20)"
        ),
        nbf.v4.new_markdown_cell("![Disease to constitution](../../outputs/initial_analysis/figures/figure_9_disease_to_constitution_heatmap.png)"),
        nbf.v4.new_markdown_cell("## Constitution-to-disease mapping"),
        nbf.v4.new_code_cell(
            "c2d = pd.read_csv(TABLES / 'constitution_to_disease_associations_baseline.csv')\n"
            "c2d.sort_values(['q_value_all','p_value']).head(20)"
        ),
        nbf.v4.new_markdown_cell("![Constitution to disease](../../outputs/initial_analysis/figures/figure_10_constitution_to_disease_heatmap.png)"),
        nbf.v4.new_markdown_cell("![Yearly distribution](../../outputs/initial_analysis/figures/figure_1_constitution_year_trend.png)"),
        nbf.v4.new_markdown_cell("## Longitudinal transitions"),
        nbf.v4.new_code_cell(
            "pd.read_csv(TABLES / 'adjacent_constitution_transition_matrix_percent.csv', index_col=0)"
        ),
        nbf.v4.new_markdown_cell("![Transition matrix](../../outputs/initial_analysis/figures/figure_2_transition_matrix.png)"),
        nbf.v4.new_markdown_cell("## Lagged next-visit risk"),
        nbf.v4.new_code_cell(
            "lagged = pd.read_csv(TABLES / 'lagged_constitution_future_risk_models.csv')\n"
            "lagged[lagged['model'].eq('next_status_adjusted_for_current_status')].sort_values(['q_value_all','p_value']).head(20)"
        ),
        nbf.v4.new_markdown_cell("![Lagged future risk](../../outputs/initial_analysis/figures/figure_7_lagged_future_risk_heatmap.png)"),
        nbf.v4.new_markdown_cell("## Lagged disease/marker risk"),
        nbf.v4.new_code_cell(
            "lagged_disease = pd.read_csv(TABLES / 'lagged_constitution_future_disease_risk_models.csv')\n"
            "lagged_disease.sort_values(['q_value_all','p_value']).head(20)"
        ),
        nbf.v4.new_markdown_cell("![Lagged future disease risk](../../outputs/initial_analysis/figures/figure_11_lagged_future_disease_risk_heatmap.png)"),
        nbf.v4.new_markdown_cell("## Transition predictor models"),
        nbf.v4.new_code_cell(
            "transition = pd.read_csv(TABLES / 'transition_predictor_models.csv')\n"
            "transition.sort_values(['q_value','p_value']).head(20)"
        ),
        nbf.v4.new_markdown_cell("![Transition predictors](../../outputs/initial_analysis/figures/figure_8_transition_predictors.png)"),
        nbf.v4.new_markdown_cell("## Constitution-phenome atlas"),
        nbf.v4.new_code_cell(
            "assoc = pd.read_csv(TABLES / 'constitution_phenome_associations.csv')\n"
            "assoc.sort_values(['q_value_all', 'p_value']).head(20)"
        ),
        nbf.v4.new_markdown_cell("![Phenome atlas](../../outputs/initial_analysis/figures/figure_3_constitution_phenome_heatmap.png)"),
        nbf.v4.new_markdown_cell("## Temporal prediction models"),
        nbf.v4.new_code_cell(
            "metrics = pd.read_csv(TABLES / 'temporal_validation_model_metrics.csv')\n"
            "metrics[['task','model','test_n','test_event_rate','auc','pr_auc','balanced_accuracy','f1','brier']]"
        ),
        nbf.v4.new_markdown_cell("## Sensitivity model without anthropometrics"),
        nbf.v4.new_code_cell(
            "sens = pd.read_csv(TABLES / 'temporal_validation_model_metrics_sensitivity_no_anthro.csv')\n"
            "sens[['task','model','test_n','test_event_rate','auc','pr_auc','balanced_accuracy','f1','brier']]"
        ),
        nbf.v4.new_markdown_cell("![ROC curves](../../outputs/initial_analysis/figures/figure_4_temporal_validation_roc.png)"),
        nbf.v4.new_markdown_cell("## Uncertainty-aware screening"),
        nbf.v4.new_code_cell(
            "pd.read_csv(TABLES / 'conformal_prediction_summary.csv')"
        ),
        nbf.v4.new_markdown_cell(
            "## Interpretation note\n\n"
            "These are first-pass feasibility results. Before manuscript submission, the clinical thresholds, covariate adjustment set, "
            "and model validation protocol should be frozen in a statistical analysis plan."
        ),
    ]
    nb["cells"] = cells
    nbf.write(nb, NB_PATH)


def main() -> None:
    print("Reading and cleaning workbook...")
    raw, df = read_and_clean()
    print(f"Cleaned frame: {df.shape[0]:,} records x {df.shape[1]:,} columns")

    # Save a de-identified analysis frame for internal reproducible model explanations.
    pkl_cols = ["person_id", "year", "is_biased", "phlegm_damp_any"] + MODEL_NUMERIC_FEATURES + MODEL_CATEGORICAL_FEATURES
    df[pkl_cols].to_pickle(OUT_DIR / "analysis_model_frame.pkl")

    overview = write_overview_tables(raw, df)
    print("Writing baseline Table 1...")
    table1 = write_table1(df)
    print("Mapping diseases and constitution groups...")
    disease_distribution, disease_associations = run_disease_constitution_mapping(df)

    print("Plotting distribution and transition figures...")
    plot_constitution_trend(df)
    pair_df = build_transition_matrix(df)

    print("Building lagged annual follow-up models...")
    lagged = build_lagged_frame(df)
    lagged_risk = run_lagged_risk_models(lagged)
    lagged_disease_risk = run_lagged_disease_risk_models(lagged)
    transition_models = run_transition_predictor_models(lagged)

    print("Running constitution-phenome association atlas...")
    assoc = run_phenome_association(df)

    print("Training temporally validated prediction models...")
    metrics, conformal = run_prediction_models(df)

    print("Writing summary and notebook...")
    write_summary_markdown(
        overview,
        pair_df,
        assoc,
        metrics,
        conformal,
        table1,
        lagged_risk,
        transition_models,
        disease_distribution,
        disease_associations,
        lagged_disease_risk,
    )
    write_notebook()

    print("Done.")
    print(f"Summary: {OUT_DIR / 'analysis_summary.md'}")
    print(f"Notebook: {NB_PATH}")


if __name__ == "__main__":
    main()
