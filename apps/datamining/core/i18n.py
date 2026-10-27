"""Dukungan dwibahasa (ID/EN) / Bilingual support."""

LANGS = {"id": "Bahasa Indonesia", "en": "English"}
DEFAULT_LANG = "id"

_fallback = {"lang": DEFAULT_LANG}


def get_lang() -> str:
    try:
        import streamlit as st

        return st.session_state.get("lang", DEFAULT_LANG)
    except Exception:
        return _fallback["lang"]


def set_lang(code: str) -> None:
    _fallback["lang"] = code
    try:
        import streamlit as st

        st.session_state["lang"] = code
    except Exception:
        pass


def L(id_text: str, en_text: str) -> str:
    """Pilih teks sesuai bahasa aktif / pick text for the active language."""
    return id_text if get_lang() == "id" else en_text


# ---------------------------------------------------------------- proses
PROCESSES = {
    "classification": ("Klasifikasi", "Classification"),
    "regression": ("Regresi", "Regression"),
    "clustering": ("Klastering", "Clustering"),
    "association": ("Association Rule Mining", "Association Rule Mining"),
    "timeseries": ("Time Series (ARIMA/SARIMA)", "Time Series (ARIMA/SARIMA)"),
}

PROCESS_DESC = {
    "classification": (
        "Memprediksi label/kelas kategorik dari sekumpulan fitur.",
        "Predicts a categorical label/class from a set of features.",
    ),
    "regression": (
        "Memprediksi nilai numerik kontinu dari sekumpulan fitur.",
        "Predicts a continuous numeric value from a set of features.",
    ),
    "clustering": (
        "Mengelompokkan data tanpa label menjadi segmen-segmen homogen.",
        "Groups unlabeled data into homogeneous segments.",
    ),
    "association": (
        "Menemukan pola 'jika-maka' antar item dalam data transaksi.",
        "Finds 'if-then' patterns between items in transaction data.",
    ),
    "timeseries": (
        "Memodelkan dan meramalkan data berurut waktu dengan ARIMA/SARIMA.",
        "Models and forecasts time-ordered data with ARIMA/SARIMA.",
    ),
}


def process_label(key: str) -> str:
    return L(*PROCESSES[key])


def process_desc(key: str) -> str:
    return L(*PROCESS_DESC[key])
