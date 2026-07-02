"""Inference helpers for the portable acidizing-response Streamlit app.

This module is intentionally self-contained. It carries the inference-time
feature engineering used by the saved sklearn pipeline so the
``streamlit-inference`` folder can be copied and run without the training code.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models" / "rf_50q"
SAMPLES_DIR = APP_DIR / "samples"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
REFERENCE_FRAME_PATH = MODEL_DIR / "reference_frame.csv"
SAMPLE_TEMPLATE_PATH = SAMPLES_DIR / "batch_template.csv"

BEFORE_COL = "IPR Vogel sebelum pengasaman (BFPD)"
INTERVAL_COL = "Interval(ft)"
ACID_COL = "Tipe Acid"
WELL_COL = "Well"
LITHOLOGY_COLS = ["Sandstone", "Siltstone", "Claystone", "Shale", "Coal", "Limestone"]

CLASS_ORDER = ["low_gain", "high_gain"]
PROBABILITY_COLUMNS = ["probability_low_gain", "probability_high_gain"]
ACTIVE_MODEL_KEY = "rf_50q"
ALLOWED_ACID_OPTIONS = [
    "Mud Acidizing 12% HCl : 3% HF",
    "Matrix Acidizing 15% HCl",
]
REQUIRED_INPUT_COLUMNS = [
    WELL_COL,
    ACID_COL,
    *LITHOLOGY_COLS,
    BEFORE_COL,
    "Batas Reservoar",
]
NUMERIC_INPUT_COLUMNS = [*LITHOLOGY_COLS, BEFORE_COL]
DERIVED_NUMERIC_COLUMNS = [INTERVAL_COL]

BASE_NUMERIC_FEATURES = [
    *[f"clr_{column.lower()}" for column in LITHOLOGY_COLS],
    BEFORE_COL,
    INTERVAL_COL,
    "before_bfpd_per_ft",
    "log1p_before_bfpd",
    "log1p_interval_ft",
]
BASE_CATEGORICAL_FEATURES = [ACID_COL]
DOMAIN_NUMERIC_FEATURES = [
    "reservoir_top",
    "reservoir_bottom",
    "reservoir_midpoint",
    "reservoir_gross_span",
    "reservoir_net_span",
    "reservoir_gap_span",
    "reservoir_segment_count",
    "reservoir_continuity_ratio",
    "interval_net_discrepancy",
    "sand_fraction",
    "fines_fraction",
    "carbonate_fraction",
    "coal_fraction",
    "sand_to_fines_ratio",
    "carbonate_to_clastic_ratio",
    "clean_sand_index",
    "lithology_entropy",
    "lithology_nonzero_count",
    "before_bfpd_per_net_span",
    "before_bfpd_per_gross_span",
    "hcl_concentration_pct",
    "hf_concentration_pct",
    "mud_acid_flag",
    "mud_acid_x_sand",
    "mud_acid_x_fines",
    "hcl_x_carbonate",
    "hf_x_silicate",
]
DOMAIN_CATEGORICAL_FEATURES = ["well_area", "layer_family", "dominant_lithology"]
MODEL_FEATURE_COLUMNS = [
    *BASE_NUMERIC_FEATURES,
    *DOMAIN_NUMERIC_FEATURES,
    *BASE_CATEGORICAL_FEATURES,
    *DOMAIN_CATEGORICAL_FEATURES,
]


@dataclass(frozen=True)
class ModelOption:
    key: str
    label: str
    short_label: str
    artifact_path: Path
    class_boundary_gain: float
    low_gain_rule: str
    high_gain_rule: str
    boundary_note: str
    decision_note: str


def _load_metadata() -> dict[str, Any]:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Missing model metadata: {METADATA_PATH}")
    with METADATA_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_model_options() -> dict[str, ModelOption]:
    metadata = _load_metadata()
    model_key = str(metadata["key"])
    artifact_path = MODEL_DIR / str(metadata["artifact_filename"])
    return {
        model_key: ModelOption(
            key=model_key,
            label=str(metadata["label"]),
            short_label=str(metadata["short_label"]),
            artifact_path=artifact_path,
            class_boundary_gain=float(metadata["class_boundary_gain"]),
            low_gain_rule=str(metadata["low_gain_rule"]),
            high_gain_rule=str(metadata["high_gain_rule"]),
            boundary_note=str(metadata["boundary_note"]),
            decision_note=str(metadata["decision_note"]),
        )
    }


MODEL_OPTIONS = _build_model_options()


def normalize_well(value: Any) -> str:
    text = str(value).strip().upper()
    return re.sub(r"\s+", "", text)


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.astype(float)
    return numerator.astype(float) / denominator.where(denominator.abs() > 1e-6)


def parse_interval_geometry(value: Any) -> dict[str, float]:
    text = str(value)
    pairs = re.findall(r"(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)", text)
    intervals = [(min(float(a), float(b)), max(float(a), float(b))) for a, b in pairs]
    if not intervals:
        return {
            "reservoir_top": np.nan,
            "reservoir_bottom": np.nan,
            "reservoir_midpoint": np.nan,
            "reservoir_gross_span": np.nan,
            "reservoir_net_span": np.nan,
            "reservoir_gap_span": np.nan,
            "reservoir_segment_count": 0.0,
            "reservoir_continuity_ratio": np.nan,
        }

    top = min(a for a, _ in intervals)
    bottom = max(b for _, b in intervals)
    gross = bottom - top
    net = sum(b - a for a, b in intervals)
    return {
        "reservoir_top": top,
        "reservoir_bottom": bottom,
        "reservoir_midpoint": (top + bottom) / 2.0,
        "reservoir_gross_span": gross,
        "reservoir_net_span": net,
        "reservoir_gap_span": max(gross - net, 0.0),
        "reservoir_segment_count": float(len(intervals)),
        "reservoir_continuity_ratio": net / gross if gross > 1e-6 else 1.0,
    }


def parse_layer_family(value: Any) -> str:
    text = str(value).upper()
    tokens = re.findall(r"\b([A-Z]+)\d*\b", text)
    stop = {"LAY", "LAP", "INT", "M", "MMD", "DAN"}
    families = [token for token in tokens if token not in stop and len(token) <= 3]
    unique = list(dict.fromkeys(families))
    if not unique:
        return "UNKNOWN"
    if len(unique) > 1:
        return "MULTI"
    return unique[0]


def parse_acid_chemistry(value: Any) -> tuple[float, float, float]:
    text = str(value).upper()
    hcl_match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*HCL", text)
    hf_match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*HF", text)
    hcl = float(hcl_match.group(1)) if hcl_match else 0.0
    hf = float(hf_match.group(1)) if hf_match else 0.0
    return hcl, hf, float("HF" in text or "MUD ACID" in text)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df.copy()
    lith = features[LITHOLOGY_COLS].astype(float).clip(lower=0.0)
    closed = (lith + 0.5).div((lith + 0.5).sum(axis=1), axis=0)
    logs = np.log(closed)
    clr = logs.sub(logs.mean(axis=1), axis=0)
    for column in LITHOLOGY_COLS:
        features[f"clr_{column.lower()}"] = clr[column]

    interval = features[INTERVAL_COL].replace(0, np.nan)
    features["before_bfpd_per_ft"] = features[BEFORE_COL] / interval
    features["log1p_before_bfpd"] = np.log1p(features[BEFORE_COL].clip(lower=0))
    features["log1p_interval_ft"] = np.log1p(features[INTERVAL_COL].clip(lower=0))
    return features


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df.copy()
    geometry = pd.DataFrame(
        [parse_interval_geometry(value) for value in features["Batas Reservoar"]],
        index=features.index,
    )
    for column in geometry:
        features[column] = geometry[column]

    features["interval_net_discrepancy"] = (
        features[INTERVAL_COL] - features["reservoir_net_span"]
    ).abs()
    lithology = features[LITHOLOGY_COLS].astype(float).clip(lower=0.0)
    lith_sum = lithology.sum(axis=1).replace(0, np.nan)
    fractions = lithology.div(lith_sum, axis=0).fillna(0.0)
    fines = fractions["Siltstone"] + fractions["Claystone"] + fractions["Shale"]
    clastic = fractions["Sandstone"] + fines
    features["sand_fraction"] = fractions["Sandstone"]
    features["fines_fraction"] = fines
    features["carbonate_fraction"] = fractions["Limestone"]
    features["coal_fraction"] = fractions["Coal"]
    features["sand_to_fines_ratio"] = (fractions["Sandstone"] + 0.01) / (fines + 0.01)
    features["carbonate_to_clastic_ratio"] = (fractions["Limestone"] + 0.01) / (
        clastic + 0.01
    )
    features["clean_sand_index"] = fractions["Sandstone"] - fines
    positive_fractions = fractions.where(fractions > 0.0, 1.0)
    features["lithology_entropy"] = -(positive_fractions * np.log(positive_fractions)).sum(axis=1)
    features["lithology_nonzero_count"] = (lithology > 0).sum(axis=1).astype(float)
    features["dominant_lithology"] = fractions.idxmax(axis=1).astype(str)
    features["before_bfpd_per_net_span"] = safe_divide(
        features[BEFORE_COL], features["reservoir_net_span"]
    )
    features["before_bfpd_per_gross_span"] = safe_divide(
        features[BEFORE_COL], features["reservoir_gross_span"]
    )
    features["well_area"] = (
        features[WELL_COL].astype(str).str.upper().str.extract(r"^([A-Z0-9]+?)-", expand=False)
    ).fillna("UNKNOWN")
    features["layer_family"] = features["Layer Target"].map(parse_layer_family)

    chemistry = features[ACID_COL].map(parse_acid_chemistry)
    features["hcl_concentration_pct"] = chemistry.map(lambda item: item[0])
    features["hf_concentration_pct"] = chemistry.map(lambda item: item[1])
    features["mud_acid_flag"] = chemistry.map(lambda item: item[2])
    features["mud_acid_x_sand"] = features["mud_acid_flag"] * features["sand_fraction"]
    features["mud_acid_x_fines"] = features["mud_acid_flag"] * features["fines_fraction"]
    features["hcl_x_carbonate"] = (
        features["hcl_concentration_pct"] * features["carbonate_fraction"]
    )
    features["hf_x_silicate"] = features["hf_concentration_pct"] * (
        features["sand_fraction"] + features["fines_fraction"]
    )
    return features


def read_csv_upload(upload: Any) -> pd.DataFrame:
    """Read an uploaded or local CSV while tolerating UTF-8 BOM headers."""
    return pd.read_csv(upload, encoding="utf-8-sig")


def template_frame() -> pd.DataFrame:
    """Return a one-row portable batch input template."""
    if not SAMPLE_TEMPLATE_PATH.exists():
        columns = REQUIRED_INPUT_COLUMNS
        return pd.DataFrame([{column: "" for column in columns}])
    return pd.read_csv(SAMPLE_TEMPLATE_PATH, encoding="utf-8-sig")


def load_model(option_key: str) -> Any:
    option = MODEL_OPTIONS[option_key]
    if not option.artifact_path.exists():
        raise FileNotFoundError(f"Missing model artifact: {option.artifact_path}")
    return joblib.load(option.artifact_path)


def load_reference_frame() -> pd.DataFrame:
    if not REFERENCE_FRAME_PATH.exists():
        raise FileNotFoundError(f"Missing reference frame: {REFERENCE_FRAME_PATH}")
    return pd.read_csv(REFERENCE_FRAME_PATH, encoding="utf-8-sig")


def canonicalize_columns(raw: pd.DataFrame) -> pd.DataFrame:
    result = raw.copy()
    result.columns = [str(column).lstrip("\ufeff").strip() for column in result.columns]
    return result


def missing_required_columns(raw: pd.DataFrame) -> list[str]:
    return [column for column in REQUIRED_INPUT_COLUMNS if column not in raw.columns]


def prepare_features(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create model-ready domain features from raw deployment inputs."""
    normalized = canonicalize_columns(raw)
    missing = missing_required_columns(normalized)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    frame = normalized.copy()
    frame.insert(0, "job_index", np.arange(1, len(frame) + 1))
    frame["well_norm"] = frame[WELL_COL].map(normalize_well)

    for column in NUMERIC_INPUT_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    interval_geometry = frame["Batas Reservoar"].map(parse_interval_geometry)
    frame[INTERVAL_COL] = interval_geometry.map(lambda item: item["reservoir_net_span"])
    frame["Layer Target"] = ""
    frame["lithology_sum"] = frame[LITHOLOGY_COLS].sum(axis=1)
    frame = add_engineered_features(frame)
    frame = add_domain_features(frame)
    return normalized, frame


def _class_probability_map(model: Any, features: pd.DataFrame) -> pd.DataFrame:
    if not hasattr(model, "predict_proba"):
        predictions = pd.Series(model.predict(features), index=features.index)
        data = pd.DataFrame(index=features.index)
        for class_name in CLASS_ORDER:
            data[f"probability_{class_name}"] = (predictions == class_name).astype(float)
        return data

    probabilities = model.predict_proba(features)
    classes = [str(item) for item in model.classes_]
    data = pd.DataFrame(index=features.index)
    for class_name in CLASS_ORDER:
        column = f"probability_{class_name}"
        if class_name in classes:
            data[column] = probabilities[:, classes.index(class_name)]
        else:
            data[column] = np.nan
    return data


def confidence_bucket(max_probability: float) -> str:
    if pd.isna(max_probability):
        return "tidak diketahui"
    if max_probability >= 0.75:
        return "tinggi"
    if max_probability >= 0.60:
        return "sedang"
    return "rendah"


def row_quality_reasons(
    raw_row: pd.Series,
    feature_row: pd.Series,
    reference: pd.DataFrame,
) -> list[str]:
    reasons: list[str] = []

    missing_inputs = [
        column
        for column in REQUIRED_INPUT_COLUMNS
        if pd.isna(raw_row.get(column)) or str(raw_row.get(column)).strip() == ""
    ]
    if missing_inputs:
        reasons.append(f"input kosong: {', '.join(missing_inputs[:4])}")

    acid = raw_row.get(ACID_COL)
    if pd.notna(acid) and str(acid).strip() not in ALLOWED_ACID_OPTIONS:
        reasons.append("tipe acid di luar dua opsi deployment")

    invalid_numeric = [
        column
        for column in NUMERIC_INPUT_COLUMNS
        if pd.isna(raw_row.get(column))
    ]
    if invalid_numeric:
        reasons.append(f"input numerik tidak valid: {', '.join(invalid_numeric[:4])}")

    lithology_sum = feature_row.get("lithology_sum", np.nan)
    if pd.notna(lithology_sum) and not (95.0 <= float(lithology_sum) <= 105.0):
        reasons.append(f"jumlah litologi {float(lithology_sum):.1f}, sebaiknya mendekati 100")

    if pd.notna(feature_row.get(INTERVAL_COL)) and float(feature_row[INTERVAL_COL]) <= 0:
        reasons.append("interval reservoir hasil parsing harus positif")
    if pd.isna(feature_row.get(INTERVAL_COL)):
        reasons.append("Batas Reservoar tidak dapat dibaca menjadi interval")
    if pd.notna(feature_row.get(BEFORE_COL)) and float(feature_row[BEFORE_COL]) <= 0:
        reasons.append("BFPD sebelum treatment tidak positif")

    outside = []
    for column in [*NUMERIC_INPUT_COLUMNS, *DERIVED_NUMERIC_COLUMNS, *DOMAIN_NUMERIC_FEATURES]:
        if column not in reference.columns or column not in feature_row.index:
            continue
        value = feature_row[column]
        if pd.isna(value):
            continue
        ref = pd.to_numeric(reference[column], errors="coerce").dropna()
        if ref.empty:
            continue
        lower = float(ref.min())
        upper = float(ref.max())
        if float(value) < lower or float(value) > upper:
            outside.append(f"{column}={float(value):.3g} di luar [{lower:.3g}, {upper:.3g}]")
    if outside:
        reasons.append("di luar rentang training: " + "; ".join(outside[:4]))

    return reasons


def predict(
    raw: pd.DataFrame,
    option_key: str,
    low_confidence_threshold: float = 0.60,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score raw deployment rows with a selected model option."""
    normalized, engineered = prepare_features(raw)
    model = load_model(option_key)
    reference = load_reference_frame()
    features = engineered[MODEL_FEATURE_COLUMNS].copy()

    predictions = pd.Series(model.predict(features), index=features.index, name="prediction")
    probability_frame = _class_probability_map(model, features)
    max_probability = probability_frame[PROBABILITY_COLUMNS].max(axis=1)

    result = normalized.copy()
    result.insert(0, "input_row", np.arange(1, len(result) + 1))
    result["selected_model"] = MODEL_OPTIONS[option_key].short_label
    result["prediction"] = predictions.values
    result[PROBABILITY_COLUMNS] = probability_frame[PROBABILITY_COLUMNS].values
    result["confidence_score"] = max_probability.round(3).values
    result["confidence"] = [confidence_bucket(value) for value in max_probability]

    flags = []
    reasons = []
    for idx, raw_row in normalized.iterrows():
        row_reasons = row_quality_reasons(raw_row, engineered.loc[idx], reference)
        if pd.isna(max_probability.loc[idx]) or float(max_probability.loc[idx]) < low_confidence_threshold:
            row_reasons.append("confidence model rendah")
        flags.append(bool(row_reasons))
        reasons.append("; ".join(row_reasons) if row_reasons else "lolos pemeriksaan dasar deployment")

    result["review_flag"] = flags
    result["reason"] = reasons
    return result, engineered


def predict_active_model(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return predict(raw, ACTIVE_MODEL_KEY)
