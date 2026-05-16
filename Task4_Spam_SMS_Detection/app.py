"""Streamlit dashboard for SMS spam detection."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predict import find_message_column  # noqa: E402
from train_model import clean_text  # noqa: E402


DATA_PATH = PROJECT_ROOT / "data" / "spam.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "best_spam_model.joblib"
COMPARISON_PATH = PROJECT_ROOT / "reports" / "model_comparison.csv"


st.set_page_config(page_title="Spam SMS Detection", layout="wide")


def inject_style() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #08121f;
            --panel: #101d2d;
            --panel-soft: #17263a;
            --line: rgba(226, 232, 240, 0.18);
            --text: #f4f8ff;
            --muted: #b8c7d9;
            --accent: #2ef2b3;
            --accent-2: #45caff;
            --accent-3: #ff8a5b;
            --danger: #ff4d78;
            --warning: #ffd166;
        }

        .stApp {
            background:
                linear-gradient(115deg, rgba(46, 242, 179, 0.16), transparent 26%),
                linear-gradient(245deg, rgba(255, 138, 91, 0.18), transparent 32%),
                linear-gradient(180deg, #08121f 0%, #0b1c2f 48%, #07111d 100%);
            color: var(--text);
            background-size: 140% 140%, 160% 160%, 100% 100%;
            animation: appFlow 14s ease-in-out infinite;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1220px;
        }

        @keyframes glowShift {
            0% { transform: translate3d(0, 0, 0) scale(1); opacity: 0.55; }
            50% { transform: translate3d(-22px, 18px, 0) scale(1.08); opacity: 0.86; }
            100% { transform: translate3d(0, 0, 0) scale(1); opacity: 0.55; }
        }

        @keyframes appFlow {
            0%, 100% { background-position: 0% 30%, 100% 10%, 0 0; }
            50% { background-position: 100% 60%, 0% 80%, 0 0; }
        }

        @keyframes slideUp {
            from { transform: translateY(18px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }

        @keyframes scanLine {
            0% { transform: translateY(-120%); opacity: 0; }
            12% { opacity: 0.9; }
            88% { opacity: 0.9; }
            100% { transform: translateY(520%); opacity: 0; }
        }

        @keyframes pulseRing {
            0% { box-shadow: 0 0 0 0 rgba(32, 214, 163, 0.45); }
            70% { box-shadow: 0 0 0 12px rgba(32, 214, 163, 0); }
            100% { box-shadow: 0 0 0 0 rgba(32, 214, 163, 0); }
        }

        @keyframes floatPhone {
            0%, 100% { transform: translateY(0) rotate(-1deg); }
            50% { transform: translateY(-10px) rotate(1deg); }
        }

        @keyframes spinRadar {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }

        @keyframes floatSignalA {
            0%, 100% { transform: translate3d(0, 0, 0); }
            50% { transform: translate3d(16px, -12px, 0); }
        }

        @keyframes floatSignalB {
            0%, 100% { transform: translate3d(0, 0, 0); }
            50% { transform: translate3d(-12px, 14px, 0); }
        }

        @keyframes gridMove {
            from { background-position: 0 0; }
            to { background-position: 46px 46px; }
        }

        @keyframes shimmer {
            0% { left: -70%; }
            100% { left: 120%; }
        }

        @keyframes barLoad {
            from { width: 0; }
        }

        [data-testid="stHeader"] {
            background: rgba(7, 17, 22, 0);
        }

        .hero {
            border: 1px solid var(--line);
            background:
                linear-gradient(135deg, rgba(46, 242, 179, 0.14), rgba(69, 202, 255, 0.10), rgba(255, 138, 91, 0.10)),
                rgba(16, 29, 45, 0.92);
            border-radius: 24px;
            padding: 30px 32px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.28);
            position: relative;
            overflow: hidden;
            animation: slideUp 650ms ease-out both;
        }

        .hero:before {
            content: "";
            position: absolute;
            inset: 0;
            background-image:
                linear-gradient(rgba(69, 202, 255, 0.10) 1px, transparent 1px),
                linear-gradient(90deg, rgba(69, 202, 255, 0.10) 1px, transparent 1px);
            background-size: 46px 46px;
            mask-image: linear-gradient(90deg, rgba(0,0,0,0.55), transparent 76%);
            animation: gridMove 9s linear infinite;
        }

        .hero:after {
            content: "";
            position: absolute;
            width: 420px;
            height: 420px;
            border: 1px solid rgba(69, 202, 255, 0.20);
            border-radius: 999px;
            right: -190px;
            top: -230px;
            animation: glowShift 6s ease-in-out infinite;
        }

        .hero-grid {
            display: grid;
            grid-template-columns: minmax(0, 1.45fr) minmax(260px, 0.55fr);
            gap: 22px;
            align-items: center;
            position: relative;
            z-index: 1;
        }

        .phone-demo {
            border: 1px solid rgba(69, 202, 255, 0.28);
            background:
                linear-gradient(180deg, rgba(19, 39, 61, 0.94), rgba(8, 18, 31, 0.96));
            border-radius: 26px;
            padding: 16px;
            box-shadow: inset 0 0 0 1px rgba(255,255,255,0.03), 0 18px 42px rgba(0,0,0,0.28);
            animation: floatPhone 4.5s ease-in-out infinite;
        }

        .shield-stage {
            position: relative;
            min-height: 245px;
            display: grid;
            place-items: center;
            overflow: hidden;
        }

        .radar-ring {
            position: absolute;
            width: 220px;
            height: 220px;
            border-radius: 999px;
            border: 1px solid rgba(69, 202, 255, 0.20);
            background:
                conic-gradient(from 90deg, rgba(46, 242, 179, 0.36), transparent 34%, rgba(255, 138, 91, 0.24), transparent 68%, rgba(69, 202, 255, 0.28));
            mask: radial-gradient(circle, transparent 54%, black 56%);
            animation: spinRadar 6.5s linear infinite;
        }

        .shield-core {
            width: 120px;
            height: 142px;
            clip-path: polygon(50% 0%, 90% 15%, 84% 70%, 50% 100%, 16% 70%, 10% 15%);
            background: linear-gradient(145deg, var(--accent), var(--accent-2) 58%, var(--accent-3));
            box-shadow: 0 0 35px rgba(46, 242, 179, 0.35), inset 0 0 24px rgba(255,255,255,0.22);
            display: grid;
            place-items: center;
            color: #07111d;
            font-size: 2.9rem;
            font-weight: 900;
            animation: pulseRing 2.4s infinite;
            position: relative;
            z-index: 2;
        }

        .signal-card {
            position: absolute;
            border: 1px solid rgba(226, 232, 240, 0.16);
            background: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(12px);
            border-radius: 14px;
            padding: 10px 12px;
            font-size: 0.82rem;
            color: #eaf6ff;
            box-shadow: 0 16px 32px rgba(0,0,0,0.18);
        }

        .signal-card.one {
            top: 22px;
            left: 8px;
            animation: floatSignalA 4s ease-in-out infinite;
        }

        .signal-card.two {
            right: 8px;
            bottom: 30px;
            animation: floatSignalB 4.7s ease-in-out infinite;
        }

        .phone-top {
            width: 72px;
            height: 6px;
            background: rgba(148, 163, 184, 0.28);
            border-radius: 999px;
            margin: 0 auto 16px;
        }

        .sms-bubble {
            position: relative;
            border: 1px solid rgba(255, 138, 91, 0.35);
            background: rgba(255, 138, 91, 0.12);
            border-radius: 16px;
            padding: 13px;
            color: #fff1f4;
            font-weight: 650;
            line-height: 1.35;
            overflow: hidden;
        }

        .sms-bubble:before {
            content: "";
            position: absolute;
            left: 0;
            right: 0;
            top: 0;
            height: 34px;
            background: linear-gradient(180deg, rgba(69, 202, 255, 0.0), rgba(69, 202, 255, 0.34), rgba(69, 202, 255, 0.0));
            animation: scanLine 3.2s ease-in-out infinite;
        }

        .phone-score {
            margin-top: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            color: var(--muted);
            font-size: 0.86rem;
        }

        .live-dot {
            width: 10px;
            height: 10px;
            border-radius: 999px;
            display: inline-block;
            background: var(--accent);
            margin-right: 7px;
            animation: pulseRing 1.8s infinite;
        }

        .eyebrow {
            color: var(--warning);
            font-size: 0.84rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }

        .hero h1 {
            margin: 0;
            color: #f7fbff;
            font-size: 2.7rem;
            line-height: 1.05;
            letter-spacing: 0;
        }

        .hero p {
            color: var(--muted);
            max-width: 760px;
            font-size: 1.04rem;
            margin: 14px 0 0 0;
        }

        .status-row {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-top: 20px;
        }

        .pill {
            border: 1px solid rgba(69, 202, 255, 0.28);
            background: linear-gradient(135deg, rgba(46, 242, 179, 0.12), rgba(69, 202, 255, 0.10));
            color: #e7fbff;
            border-radius: 999px;
            padding: 7px 12px;
            font-size: 0.86rem;
            font-weight: 650;
            animation: slideUp 700ms ease-out both;
        }

        .card {
            border: 1px solid var(--line);
            background: rgba(13, 27, 34, 0.88);
            border-radius: 14px;
            padding: 18px;
            box-shadow: 0 16px 40px rgba(0, 0, 0, 0.18);
            min-height: 116px;
            animation: slideUp 600ms ease-out both;
            transition: transform 180ms ease, border-color 180ms ease, background 180ms ease;
        }

        .card:hover {
            transform: translateY(-4px);
            border-color: rgba(255, 138, 91, 0.48);
            background: rgba(23, 38, 58, 0.94);
        }

        .metric-label {
            color: var(--muted);
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            font-weight: 700;
        }

        .metric-value {
            color: #f8fbff;
            font-size: 2rem;
            font-weight: 800;
            margin-top: 8px;
        }

        .metric-note {
            color: var(--muted);
            font-size: 0.88rem;
            margin-top: 6px;
        }

        .section-title {
            color: #f8fbff;
            font-size: 1.15rem;
            font-weight: 800;
            margin: 20px 0 10px;
        }

        .result-card {
            border-radius: 16px;
            padding: 20px;
            border: 1px solid var(--line);
            background: rgba(13, 27, 34, 0.92);
        }

        .result-spam {
            border-color: rgba(255, 85, 115, 0.55);
            background: linear-gradient(135deg, rgba(255, 85, 115, 0.18), rgba(13, 27, 34, 0.92));
        }

        .result-ham {
            border-color: rgba(32, 214, 163, 0.55);
            background: linear-gradient(135deg, rgba(32, 214, 163, 0.18), rgba(13, 27, 34, 0.92));
        }

        .risk-number {
            font-size: 2.4rem;
            font-weight: 900;
            color: #f8fbff;
            margin: 4px 0;
        }

        .risk-bar {
            height: 12px;
            border-radius: 999px;
            background: rgba(148, 163, 184, 0.18);
            overflow: hidden;
            margin-top: 14px;
        }

        .risk-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, var(--accent), var(--warning), var(--danger));
            animation: barLoad 900ms ease-out both;
        }

        .result-card {
            animation: slideUp 520ms ease-out both;
            position: relative;
            overflow: hidden;
        }

        .result-card:after {
            content: "";
            position: absolute;
            top: 0;
            bottom: 0;
            width: 120px;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
            animation: shimmer 2.8s ease-in-out infinite;
        }

        .small-muted {
            color: var(--muted);
            font-size: 0.92rem;
        }

        div[data-testid="stTabs"] button {
            color: #dbeafe;
            font-weight: 700;
        }

        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: var(--accent);
        }

        div[data-testid="stTextArea"] textarea,
        div[data-testid="stFileUploader"] section {
            border-radius: 14px;
            border-color: rgba(110, 231, 249, 0.28);
            background-color: rgba(13, 27, 34, 0.7);
        }

        .stButton > button,
        .stDownloadButton > button {
            border-radius: 12px;
            border: 1px solid rgba(255, 209, 102, 0.42);
            background: linear-gradient(135deg, #2ef2b3, #45caff 55%, #ffd166);
            color: #041014;
            font-weight: 800;
            min-height: 44px;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            border-color: rgba(247, 251, 255, 0.7);
            color: #041014;
        }

        @media (max-width: 800px) {
            .hero-grid {
                grid-template-columns: 1fr;
            }

            .hero h1 {
                font-size: 2.05rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_dataset() -> pd.DataFrame | None:
    if not DATA_PATH.exists():
        return None
    df = pd.read_csv(DATA_PATH, encoding="latin-1")
    df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "message"})
    df["spam"] = df["label"].map({"ham": 0, "spam": 1})
    df["message_length"] = df["message"].str.len()
    df["clean_message"] = df["message"].map(clean_text)
    return df


@st.cache_data
def load_comparison() -> pd.DataFrame | None:
    if not COMPARISON_PATH.exists():
        return None
    return pd.read_csv(COMPARISON_PATH)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def metric_card(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def hero(model_name: str | None) -> None:
    model_text = model_name if model_name else "train model first"
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-grid">
                <div>
                    <div class="eyebrow">AI MESSAGE GUARD | REAL TIME SMS FILTER</div>
                    <h1>Message Shield for Spam SMS Detection</h1>
                    <p>
                        A colorful live scanner that reads an SMS, turns the text into TF-IDF signals,
                        and highlights whether the message looks safe or suspicious.
                    </p>
                    <div class="status-row">
                        <span class="pill"><span class="live-dot"></span>Live scanner</span>
                        <span class="pill">TF-IDF text features</span>
                        <span class="pill">Naive Bayes</span>
                        <span class="pill">Logistic Regression</span>
                        <span class="pill">Linear SVM</span>
                        <span class="pill">Active model: {model_text}</span>
                    </div>
                </div>
                <div class="phone-demo">
                    <div class="shield-stage">
                        <div class="radar-ring"></div>
                        <div class="signal-card one">TF-IDF signal locked</div>
                        <div class="shield-core">S</div>
                        <div class="signal-card two">Spam risk scanning</div>
                    </div>
                    <div class="sms-bubble">
                        URGENT! You have won a reward. Click now to claim before midnight.
                    </div>
                    <div class="phone-score">
                        <span><span class="live-dot"></span>Animated scanner active</span>
                        <strong>Shield online</strong>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def predict_messages(model, messages: pd.Series) -> pd.DataFrame:
    clean_messages = messages.map(clean_text)
    probabilities = model.predict_proba(clean_messages)[:, 1]
    predictions = model.predict(clean_messages)
    return pd.DataFrame(
        {
            "message": messages,
            "spam_probability": probabilities,
            "predicted_label": ["spam" if item == 1 else "ham" for item in predictions],
        }
    )


def top_words(messages: pd.Series, limit: int = 15) -> pd.DataFrame:
    words: list[str] = []
    for message in messages:
        words.extend(clean_text(message).split())

    blocked = {"the", "and", "you", "for", "that", "this", "with", "have", "your"}
    common_words = Counter(
        word for word in words if len(word) > 2 and word not in blocked
    ).most_common(limit)
    return pd.DataFrame(common_words, columns=["word", "count"])


def show_result(probability: float, label: str) -> None:
    width = max(4, min(100, int(probability * 100)))
    label_text = "Spam detected" if label == "spam" else "Legitimate message"
    card_class = "result-spam" if label == "spam" else "result-ham"
    note = (
        "This message has promotional or suspicious language patterns."
        if label == "spam"
        else "The message pattern is closer to normal personal communication."
    )
    st.markdown(
        f"""
        <div class="result-card {card_class}">
            <div class="metric-label">Prediction result</div>
            <div class="risk-number">{label_text}</div>
            <div class="small-muted">Spam probability: {probability:.2%}</div>
            <div class="risk-bar"><div class="risk-fill" style="width:{width}%"></div></div>
            <p class="small-muted">{note}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


inject_style()

df = load_dataset()
comparison = load_comparison()
model_bundle = load_model()
model_name = model_bundle["model_name"] if model_bundle is not None else None

hero(model_name)

if model_bundle is None:
    st.warning("Model not found. Run `python src/train_model.py` before using predictions.")

overview_tab, prediction_tab, insights_tab = st.tabs(
    ["Security Overview", "SMS Analyzer", "Message Intelligence"]
)

with overview_tab:
    if df is not None:
        spam_count = int(df["spam"].sum())
        ham_count = int((df["spam"] == 0).sum())
        spam_rate = df["spam"].mean() * 100
        avg_length = df["message_length"].mean()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Total Messages", f"{len(df):,}", "SMS records loaded")
        with c2:
            metric_card("Spam Found", f"{spam_count:,}", "messages marked as spam")
        with c3:
            metric_card("Spam Rate", f"{spam_rate:.1f}%", "dataset imbalance")
        with c4:
            metric_card("Average Length", f"{avg_length:.0f}", "characters per SMS")

        st.markdown('<div class="section-title">Traffic Breakdown</div>', unsafe_allow_html=True)
        left, right = st.columns([1.1, 1])
        with left:
            st.bar_chart(pd.Series({"Legitimate": ham_count, "Spam": spam_count}))
        with right:
            st.dataframe(
                pd.DataFrame(
                    [
                        {"label": "Legitimate", "messages": ham_count, "share": f"{100 - spam_rate:.1f}%"},
                        {"label": "Spam", "messages": spam_count, "share": f"{spam_rate:.1f}%"},
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info("Dataset not found. Extract `spam.csv` into the data folder.")

    if comparison is not None:
        st.markdown('<div class="section-title">Model Leaderboard</div>', unsafe_allow_html=True)
        rounded = comparison.copy()
        numeric_cols = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "average_precision"]
        rounded[numeric_cols] = rounded[numeric_cols].round(4)
        st.dataframe(rounded, use_container_width=True, hide_index=True)
        st.bar_chart(comparison.set_index("model")[["precision", "recall", "f1_score", "roc_auc"]])
    else:
        st.info("Model comparison report is not available yet. Train the model first.")

with prediction_tab:
    if model_bundle is not None:
        left, right = st.columns([1.15, 0.85])
        with left:
            st.markdown('<div class="section-title">Live SMS Analyzer</div>', unsafe_allow_html=True)
            sample_sms = st.text_area(
                "Paste SMS message",
                value="Congratulations! You won a free prize. Call now to claim.",
                height=170,
            )
            analyze = st.button("Analyze Message", type="primary", use_container_width=True)

        with right:
            st.markdown('<div class="section-title">Result Panel</div>', unsafe_allow_html=True)
            if analyze:
                result = predict_messages(model_bundle["model"], pd.Series([sample_sms]))
                probability = float(result.loc[0, "spam_probability"])
                label = result.loc[0, "predicted_label"]
                show_result(probability, label)
            else:
                st.markdown(
                    """
                    <div class="result-card">
                        <div class="metric-label">Waiting for message</div>
                        <div class="risk-number">Ready</div>
                        <p class="small-muted">Paste an SMS and run the analyzer to see the spam score.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown('<div class="section-title">Batch Prediction</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload SMS CSV with a message, sms, text, or v2 column", type=["csv"])

        if uploaded_file is not None:
            uploaded_df = pd.read_csv(uploaded_file, encoding="latin-1")
            message_column = find_message_column(uploaded_df)
            predictions = predict_messages(model_bundle["model"], uploaded_df[message_column])
            st.dataframe(predictions.head(100), use_container_width=True, hide_index=True)
            st.download_button(
                "Download predictions",
                predictions.to_csv(index=False).encode("utf-8"),
                file_name="spam_sms_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )
    else:
        st.info("Train the model to enable predictions.")

with insights_tab:
    if df is not None:
        st.markdown('<div class="section-title">Message Length Pattern</div>', unsafe_allow_html=True)
        length_summary = df.groupby("label")["message_length"].mean().sort_values(ascending=False)
        st.bar_chart(length_summary)

        spam_words = top_words(df.loc[df["label"] == "spam", "message"])
        ham_words = top_words(df.loc[df["label"] == "ham", "message"])

        left, right = st.columns(2)
        with left:
            st.markdown('<div class="section-title">Common Spam Words</div>', unsafe_allow_html=True)
            st.dataframe(spam_words, use_container_width=True, hide_index=True)
            st.bar_chart(spam_words.set_index("word"))
        with right:
            st.markdown('<div class="section-title">Common Legitimate Words</div>', unsafe_allow_html=True)
            st.dataframe(ham_words, use_container_width=True, hide_index=True)
            st.bar_chart(ham_words.set_index("word"))

        st.markdown('<div class="section-title">Dataset Preview</div>', unsafe_allow_html=True)
        st.dataframe(df[["label", "message", "message_length"]].head(100), use_container_width=True)
    else:
        st.info("Dataset not found.")
