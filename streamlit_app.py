"""Aplikasi Streamlit untuk screening respons acidizing rendah/tinggi."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import streamlit as st

from model_service import (
    ACID_COL,
    ACTIVE_MODEL_KEY,
    APP_DIR,
    ALLOWED_ACID_OPTIONS,
    BEFORE_COL,
    LITHOLOGY_COLS,
    MODEL_OPTIONS,
    REQUIRED_INPUT_COLUMNS,
    WELL_COL,
    predict_active_model,
    read_csv_upload,
    template_frame,
)


st.set_page_config(
    page_title="Screening Respons Acidizing",
    page_icon=None,
    layout="wide",
)


st.markdown(
    """
    <style>
    .main .block-container { padding-top: 1.4rem; max-width: 1180px; }
    .app-title { font-size: 1.9rem; font-weight: 720; letter-spacing: 0; }
    .subtle { color: #5b6472; font-size: 0.92rem; }
    .model-note {
        border-left: 4px solid #2f6f73;
        background: #f2f7f6;
        padding: 0.75rem 0.9rem;
        margin: 0.5rem 0 1rem 0;
    }
    .result-band {
        border: 1px solid #d8dee4;
        border-radius: 8px;
        padding: 0.95rem 1rem;
        background: #ffffff;
    }
    .flagged { color: #9a3412; font-weight: 650; }
    .clear { color: #166534; font-weight: 650; }
    div[data-testid="stMetric"] {
        border: 1px solid #d8dee4;
        border-radius: 8px;
        padding: 0.7rem 0.8rem;
        background: #fbfcfd;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


LABEL_PREDIKSI = {
    "low_gain": "Gain rendah",
    "high_gain": "Gain tinggi",
}


@st.cache_data
def cached_template_csv() -> str:
    buffer = StringIO()
    template_frame().to_csv(buffer, index=False)
    return buffer.getvalue()


def label_prediksi(value: object) -> str:
    return LABEL_PREDIKSI.get(str(value), str(value))


def tampilan_hasil(scored: pd.DataFrame) -> pd.DataFrame:
    view = scored.copy()
    view["prediksi"] = view["prediction"].map(label_prediksi)
    view = view.rename(
        columns={
            "input_row": "baris_input",
            WELL_COL: "well",
            "selected_model": "model",
            "probability_low_gain": "probabilitas_gain_rendah",
            "probability_high_gain": "probabilitas_gain_tinggi",
            "confidence_score": "skor_keyakinan",
            "confidence": "keyakinan",
            "review_flag": "perlu_review",
            "reason": "alasan",
        }
    )
    columns = [
        "baris_input",
        "well",
        "model",
        "prediksi",
        "probabilitas_gain_rendah",
        "probabilitas_gain_tinggi",
        "skor_keyakinan",
        "keyakinan",
        "perlu_review",
        "alasan",
    ]
    return view[[column for column in columns if column in view.columns]]


def render_model_note() -> None:
    option = MODEL_OPTIONS[ACTIVE_MODEL_KEY]
    st.markdown(
        f"""
        <div class="model-note">
          <strong>{option.short_label}</strong><br>
          {option.boundary_note}<br>
          Garis batas kelas gain: <strong>{option.class_boundary_gain:.9f}</strong><br>
          <span class="subtle">{option.decision_note}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_batch() -> None:
    left, right = st.columns([2, 1])
    with left:
        uploaded = st.file_uploader("Unggah CSV", type=["csv"])
    with right:
        st.download_button(
            "Unduh template",
            data=cached_template_csv(),
            file_name="template_input_respons_acidizing.csv",
            mime="text/csv",
        )

    if uploaded is None:
        return

    try:
        raw = read_csv_upload(uploaded)
        scored, _ = predict_active_model(raw)
    except Exception as exc:
        st.error(str(exc))
        return

    summary_cols = st.columns(4)
    summary_cols[0].metric("Baris input", int(scored["input_row"].nunique()))
    summary_cols[1].metric("Gain tinggi", int((scored["prediction"] == "high_gain").sum()))
    summary_cols[2].metric("Gain rendah", int((scored["prediction"] == "low_gain").sum()))
    summary_cols[3].metric("Perlu review", int(scored["review_flag"].sum()))

    st.dataframe(tampilan_hasil(scored), hide_index=True, width="stretch")
    st.download_button(
        "Unduh hasil CSV",
        data=scored.to_csv(index=False),
        file_name="prediksi_respons_acidizing_rf_50_median.csv",
        mime="text/csv",
    )


def render_model_notes() -> None:
    st.subheader("Model yang digunakan")
    # render_model_note()

    option = MODEL_OPTIONS[ACTIVE_MODEL_KEY]
    artifact_display = Path(option.artifact_path).relative_to(APP_DIR)
    notes = pd.DataFrame(
        [
            {
                "model": option.short_label,
                "artifact": str(artifact_display),
                "batas": option.boundary_note,
                "dasar_pemilihan": option.decision_note,
            }
        ]
    )
    st.dataframe(notes, hide_index=True, width="stretch")

    st.info(
        "Prediksi ini adalah sinyal pendukung keputusan untuk engineer. "
        "Hasilnya bukan akurasi deployment yang sudah tervalidasi penuh dan "
        "tetap perlu ditinjau bersama konteks reservoir, produksi, petrofisika, "
        "dan operasi."
    )


def render_input_contract() -> None:
    st.subheader("Bentuk CSV")
    contract = pd.DataFrame(
        {
            "kolom": REQUIRED_INPUT_COLUMNS,
        }
    )
    st.dataframe(contract, hide_index=True, width="stretch")
    st.download_button(
        "Unduh template CSV",
        data=cached_template_csv(),
        file_name="template_input_respons_acidizing.csv",
        mime="text/csv",
    )


def main() -> None:
    st.markdown('<div class="app-title">Screening Respons Acidizing</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtle">Prediksi gain rendah/tinggi dari input mudlog dan desain treatment sebelum acidizing.</div>',
        unsafe_allow_html=True,
    )

    # st.sidebar.info("Model aktif: RF 50% median.")
    render_model_note()

    batch_tab, notes_tab, contract_tab = st.tabs(
        ["Input CSV", "Catatan model", "Contoh CSV"]
    )
    with batch_tab:
        render_batch()

    with notes_tab:
        render_model_notes()

    with contract_tab:
        render_input_contract()


if __name__ == "__main__":
    main()
