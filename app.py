import streamlit as st
import pandas as pd
import pickle
import re
import time
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="DEEN Spam Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# PASTEL PINK & PURPLE THEME — CSS
# ─────────────────────────────────────────
# Colour palette:
#   Deep purple  : #7C3D8F   (headings, borders, sidebar text)
#   Soft purple  : #B39DDB   (sidebar gradient mid)
#   Pastel purple: #E8D5F5   (card backgrounds, pills)
#   Pastel pink  : #FFD6E7   (accent fills, chips)
#   Deep pink    : #D63A8A   (highlights, metric numbers)
#   Light pink   : #FFF0F7   (page background)
#   White        : #FFFFFF   (card surfaces)
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;600;700;800&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif;
}

/* ── Page background ── */
.main, .block-container {
    background-color: #FFF0F7 !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #6A1B8A 0%, #9C4DB8 50%, #CE93D8 100%);
}
section[data-testid="stSidebar"] * {
    color: white !important;
}
section[data-testid="stSidebar"] .stSelectbox label {
    color: #FFD6E7 !important;
    font-weight: 700;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.3) !important;
}

/* ── Metric cards ── */
.metric-card {
    background: white;
    border-radius: 18px;
    padding: 22px 20px;
    box-shadow: 0 4px 18px rgba(124,61,143,0.12);
    text-align: center;
    border-left: 5px solid #B39DDB;
    margin-bottom: 12px;
    transition: transform 0.2s;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(124,61,143,0.18);
}
.metric-card h2 {
    font-size: 2rem;
    font-weight: 800;
    color: #7C3D8F;
    margin: 0;
}
.metric-card p {
    color: #999;
    font-size: 0.83rem;
    margin: 5px 0 0 0;
    font-weight: 600;
}

/* ── Section headers ── */
.section-header {
    background: linear-gradient(90deg, #9C4DB8, #D63A8A);
    color: white !important;
    border-radius: 14px;
    padding: 13px 22px;
    margin-bottom: 20px;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.3px;
}

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #6A1B8A 0%, #9C4DB8 50%, #D63A8A 100%);
    border-radius: 22px;
    padding: 44px 40px;
    color: white;
    margin-bottom: 30px;
    box-shadow: 0 8px 32px rgba(124,61,143,0.25);
}

/* ── Result cards ── */
.spam-card {
    background: linear-gradient(135deg, #D63A8A, #F06292);
    color: white;
    border-radius: 18px;
    padding: 30px;
    text-align: center;
    font-size: 1.7rem;
    font-weight: 800;
    box-shadow: 0 8px 28px rgba(214,58,138,0.4);
    animation: pinkpulse 1.6s ease-in-out infinite;
}
.ham-card {
    background: linear-gradient(135deg, #9C4DB8, #CE93D8);
    color: white;
    border-radius: 18px;
    padding: 30px;
    text-align: center;
    font-size: 1.7rem;
    font-weight: 800;
    box-shadow: 0 8px 28px rgba(156,77,184,0.35);
}
@keyframes pinkpulse {
    0%   { box-shadow: 0 8px 28px rgba(214,58,138,0.4); }
    50%  { box-shadow: 0 12px 42px rgba(214,58,138,0.65); }
    100% { box-shadow: 0 8px 28px rgba(214,58,138,0.4); }
}

/* ── Info chips (pink) ── */
.chip {
    display: inline-block;
    background: #FFD6E7;
    color: #7C3D8F;
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.8rem;
    font-weight: 700;
    margin: 3px;
    border: 1px solid #F48FB1;
}

/* ── Stat pills (purple) ── */
.stat-pill {
    background: #E8D5F5;
    border-radius: 30px;
    padding: 6px 18px;
    font-size: 0.85rem;
    color: #6A1B8A;
    font-weight: 700;
    display: inline-block;
    margin: 4px;
    border: 1px solid #CE93D8;
}

/* ── Team cards ── */
.team-card {
    background: white;
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(124,61,143,0.1);
    border-top: 4px solid #CE93D8;
    margin-bottom: 10px;
    transition: transform 0.2s;
}
.team-card:hover { transform: translateY(-3px); }
.team-card h4 {
    color: #7C3D8F;
    margin: 10px 0 4px 0;
    font-size: 0.95rem;
    font-weight: 700;
}
.team-card p { color: #B39DDB; font-size: 0.78rem; margin: 0; font-weight: 600; }

/* ── Tips box ── */
.tips-box {
    background: white;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 4px 15px rgba(124,61,143,0.1);
    border-left: 4px solid #F48FB1;
    height: 100%;
}

/* ── Conf label ── */
.conf-label {
    font-size: 0.85rem;
    color: #9C4DB8;
    font-weight: 700;
    margin-bottom: 4px;
}

/* ── Best model banner ── */
.best-model-banner {
    background: linear-gradient(135deg, #9C4DB8, #D63A8A);
    border-radius: 18px;
    padding: 26px;
    color: white;
    margin-bottom: 24px;
    box-shadow: 0 6px 22px rgba(124,61,143,0.3);
}

/* ── Streamlit native overrides ── */
div[data-testid="stMetric"] {
    background: white;
    border-radius: 14px;
    padding: 14px;
    box-shadow: 0 3px 12px rgba(124,61,143,0.1);
    border-left: 4px solid #CE93D8;
}
div[data-testid="stMetric"] label {
    color: #9C4DB8 !important;
    font-weight: 700;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #6A1B8A !important;
    font-weight: 800;
}

/* ── Buttons ── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #9C4DB8, #D63A8A) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    padding: 10px 20px !important;
    box-shadow: 0 4px 14px rgba(124,61,143,0.3) !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(124,61,143,0.45) !important;
}

/* ── Expander ── */
details {
    background: white !important;
    border-radius: 12px !important;
    border: 1px solid #E8D5F5 !important;
    padding: 4px 8px;
}
summary {
    color: #7C3D8F !important;
    font-weight: 700 !important;
}

/* ── Dataframe ── */
div[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 3px 12px rgba(124,61,143,0.08);
}

/* ── Text input / Text area ── */
div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea {
    border: 2px solid #CE93D8 !important;
    border-radius: 12px !important;
    background: white !important;
}
div[data-testid="stTextInput"] input:focus,
div[data-testid="stTextArea"] textarea:focus {
    border-color: #9C4DB8 !important;
    box-shadow: 0 0 0 3px rgba(156,77,184,0.15) !important;
}

/* ── Selectbox ── */
div[data-testid="stSelectbox"] > div > div {
    border: 2px solid #CE93D8 !important;
    border-radius: 12px !important;
}

/* ── Progress bar ── */
div[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #9C4DB8, #D63A8A) !important;
    border-radius: 10px !important;
}
div[data-testid="stProgress"] > div {
    background: #E8D5F5 !important;
    border-radius: 10px !important;
}

/* ── Success / warning / info boxes ── */
div[data-testid="stAlert"] {
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [ps.stem(w) for w in words if w not in stop_words]
    return ' '.join(words)


# ─────────────────────────────────────────
# LOAD DATA & MODEL
# ─────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/spam.csv", encoding="latin-1")
    if "v1" in df.columns:
        df = df[["v1", "v2"]]
        df.columns = ["label", "message"]
    return df

@st.cache_resource
def load_model():
    model      = pickle.load(open("models/best_model.pkl", "rb"))
    vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))
    return model, vectorizer

df                = load_data()
model, vectorizer = load_model()

# Session state
if "history" not in st.session_state:
    st.session_state.history = []


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 14px 0 22px 0;'>
        <div style='font-size:3.2rem;'>🌸</div>
        <div style='font-size:1.5rem; font-weight:800; letter-spacing:2px;'>DEEN</div>
        <div style='font-size:0.72rem; color:#FFD6E7; margin-top:4px; font-weight:600;'>
            Detection Engine for Email Nuisance
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.selectbox(
        "🗂️ Navigate",
        ["🏠 Home", "🔍 Text Analyzer", "📊 Data Explorer",
         "📈 Visualizations", "🤖 Model Info", "📋 History"]
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.73rem; color:#FFD6E7; text-align:center; line-height:1.7;'>
        Built with 💜 by Team DEEN<br/>
        NLP Assignment · 2026
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════
# HOME PAGE
# ══════════════════════════════════════════
if page == "🏠 Home":

    st.markdown("""
    <div class='hero-banner'>
        <h1 style='margin:0; font-size:2.6rem; font-weight:800;'>🌸 DEEN Spam Detector</h1>
        <p style='margin:10px 0 0 0; color:#FFD6E7; font-size:1.05rem; font-weight:600;'>
            Detection Engine for Email Nuisance &nbsp;·&nbsp;
            Powered by NLP &amp; Machine Learning
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Quick stats
    total  = len(df)
    n_spam = len(df[df["label"] == "spam"]) if "spam" in df["label"].values else len(df[df["label"] == 1])
    n_ham  = total - n_spam

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='metric-card'>
            <h2>{total:,}</h2><p>Total Messages</p></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-card'>
            <h2 style='color:#D63A8A;'>{n_spam:,}</h2><p>Spam Messages</p></div>""",
            unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='metric-card'>
            <h2 style='color:#9C4DB8;'>{n_ham:,}</h2><p>Ham Messages</p></div>""",
            unsafe_allow_html=True)
    with c4:
        st.markdown("""<div class='metric-card'>
            <h2 style='color:#D63A8A;'>98.1%</h2><p>Best Accuracy</p></div>""",
            unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("<div class='section-header'>💜 About This Project</div>", unsafe_allow_html=True)
        st.write("""
        This project uses **Natural Language Processing (NLP)** and **Machine Learning**
        to classify SMS/email messages as **Spam** or **Ham (Not Spam)**.

        We compare four classifier configurations:
        """)
        for m in ["BoW + Naive Bayes ⭐ Best", "BoW + KNN",
                  "TF-IDF + Naive Bayes", "TF-IDF + KNN"]:
            st.markdown(f"<span class='chip'>{m}</span>", unsafe_allow_html=True)

    with col_b:
        st.markdown("<div class='section-header'>🚀 How to Use</div>", unsafe_allow_html=True)
        steps = [
            ("1️⃣", "Go to **Text Analyzer** in the sidebar"),
            ("2️⃣", "Type or paste your SMS/email message"),
            ("3️⃣", "Click the **Analyze** button"),
            ("4️⃣", "View the prediction and confidence score"),
            ("5️⃣", "Check **History** to review past predictions"),
        ]
        for icon, step in steps:
            st.markdown(f"{icon} &nbsp; {step}")

    st.markdown("<br/>", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>👥 Team Members</div>", unsafe_allow_html=True)
    members = [
        ("Nur Adina",        "🌸", "Group Leader"),
        ("Laila Khadija",    "💜", "Data Analysis"),
        ("Kesshi Akshainie", "🌷", "Model Training"),
        ("Nurin Afiqah",     "✨", "UI & Reporting"),
    ]
    cols = st.columns(4)
    for col, (name, emoji, role) in zip(cols, members):
        with col:
            st.markdown(f"""
            <div class='team-card'>
                <div style='font-size:2.2rem;'>{emoji}</div>
                <h4>{name}</h4>
                <p>{role}</p>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════
# TEXT ANALYZER
# ══════════════════════════════════════════
elif page == "🔍 Text Analyzer":

    st.markdown("<div class='section-header'>🔍 Text Analyzer</div>", unsafe_allow_html=True)
    st.write("Enter an SMS or email message below to check whether it is **Spam** or **Ham**.")

    col1, col2 = st.columns([2, 1])

    with col1:
        message = st.text_area(
            "✉️ Your Message",
            placeholder="e.g. Congratulations! You've won a FREE prize. Call now to claim...",
            height=160
        )
        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            analyze = st.button("🔍 Analyze", use_container_width=True)
        with col_btn2:
            clear = st.button("🗑️ Clear")
        if clear:
            st.rerun()

    with col2:
        st.markdown("""
        <div class='tips-box'>
            <b style='color:#7C3D8F;'>💡 Tips</b><br/><br/>
            <small style='color:#555; line-height:2;'>
            • Spam often uses <b style='color:#D63A8A;'>FREE</b>,
              <b style='color:#D63A8A;'>WIN</b>,
              <b style='color:#D63A8A;'>URGENT</b><br/>
            • Ham is typically personal &amp; conversational<br/>
            • Excessive <b>!</b> marks signal spam<br/>
            • Model trained on SMS Spam Collection Dataset
            </small>
        </div>
        """, unsafe_allow_html=True)

    if analyze:
        if not message.strip():
            st.warning("⚠️ Please enter a message before analyzing.")
        else:
            with st.spinner("✨ Analyzing your message..."):
                time.sleep(0.6)

                processed   = preprocess(message)
                transformed = vectorizer.transform([processed])
                prediction  = model.predict(transformed)[0]
                probability = model.predict_proba(transformed)[0]
                confidence  = max(probability) * 100
                spam_prob   = probability[1] * 100
                ham_prob    = probability[0] * 100

            st.markdown("<br/>", unsafe_allow_html=True)

            if prediction == 1:
                st.markdown(f"""
                <div class='spam-card'>
                    🚨 SPAM DETECTED<br/>
                    <span style='font-size:1rem; font-weight:600; opacity:0.9;'>
                        Confidence: {confidence:.1f}%
                    </span>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='ham-card'>
                    ✅ HAM — Safe Message<br/>
                    <span style='font-size:1rem; font-weight:600; opacity:0.9;'>
                        Confidence: {confidence:.1f}%
                    </span>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br/>", unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("<p class='conf-label'>💜 Ham Probability</p>", unsafe_allow_html=True)
                st.progress(ham_prob / 100)
                st.caption(f"{ham_prob:.1f}%")
            with c2:
                st.markdown("<p class='conf-label'>🌸 Spam Probability</p>", unsafe_allow_html=True)
                st.progress(spam_prob / 100)
                st.caption(f"{spam_prob:.1f}%")

            st.markdown("<br/>", unsafe_allow_html=True)
            st.markdown("<div class='section-header'>📊 Message Statistics</div>", unsafe_allow_html=True)

            words        = message.split()
            unique_words = set(w.lower() for w in words)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Characters", len(message))
            c2.metric("Total Words",      len(words))
            c3.metric("Unique Words",     len(unique_words))
            c4.metric("Sentences",
                      message.count('.') + message.count('!') + message.count('?') or 1)

            with st.expander("🔬 View Preprocessed Tokens"):
                tokens = processed.split()
                if tokens:
                    st.markdown(
                        " ".join([f"<span class='chip'>{t}</span>" for t in tokens]),
                        unsafe_allow_html=True
                    )
                else:
                    st.write("No tokens remaining after preprocessing.")

            st.session_state.history.append({
                "Message"   : message[:80] + ("..." if len(message) > 80 else ""),
                "Result"    : "🚨 SPAM" if prediction == 1 else "✅ HAM",
                "Confidence": f"{confidence:.1f}%",
                "Words"     : len(words),
            })
            st.success("✔️ Result saved to History.")


# ══════════════════════════════════════════
# DATA EXPLORER
# ══════════════════════════════════════════
elif page == "📊 Data Explorer":

    st.markdown("<div class='section-header'>📊 Data Explorer</div>", unsafe_allow_html=True)

    total  = len(df)
    n_spam = int((df["label"] == "spam").sum()) if "spam" in df["label"].values \
             else int((df["label"] == 1).sum())
    n_ham  = total - n_spam

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class='metric-card'>
            <h2>{total:,}</h2><p>Total Messages</p></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-card'>
            <h2 style='color:#D63A8A;'>{n_spam:,}</h2><p>Spam Messages</p></div>""",
            unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='metric-card'>
            <h2 style='color:#9C4DB8;'>{n_ham:,}</h2><p>Ham Messages</p></div>""",
            unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>🔎 Filter & Search</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        label_filter = st.selectbox("Filter by Label", ["All", "Ham", "Spam"])
    with col2:
        search_term = st.text_input("🔍 Search messages", placeholder="Type a keyword...")

    filtered = df.copy()
    if label_filter == "Spam":
        filtered = filtered[filtered["label"].isin(["spam", 1])]
    elif label_filter == "Ham":
        filtered = filtered[filtered["label"].isin(["ham", 0])]
    if search_term:
        filtered = filtered[filtered["message"].str.contains(search_term, case=False, na=False)]

    st.markdown(f"Showing **{len(filtered):,}** messages")
    st.dataframe(filtered.head(50), use_container_width=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>📏 Average Message Length by Class</div>",
                unsafe_allow_html=True)

    df["msg_length"] = df["message"].apply(len)
    length_stats = df.groupby("label")["msg_length"].mean().reset_index()
    length_stats.columns = ["Label", "Average Length (chars)"]
    st.dataframe(length_stats, use_container_width=True)
    st.bar_chart(df.groupby("label")["msg_length"].mean())


# ══════════════════════════════════════════
# VISUALIZATIONS
# ══════════════════════════════════════════
elif page == "📈 Visualizations":

    st.markdown("<div class='section-header'>📈 Visualizations</div>", unsafe_allow_html=True)

    viz_list = [
        ("Visualization 1 - Spam vs Ham Distribution.png",
         "Figure 1: Class Distribution",
         "Shows the imbalance between ham (~87%) and spam (~13%) in the dataset."),
        ("Visualization 2 - Spam Word Cloud.png",
         "Figure 2: Spam Word Cloud",
         "Most frequent words in spam messages — 'free', 'call', 'win', 'urgent' dominate."),
        ("Visualization 3 - Ham Word Cloud.png",
         "Figure 3: Ham Word Cloud",
         "Most frequent words in legitimate messages — casual, conversational vocabulary."),
        ("Visualization 4 - Top 20 Words.png",
         "Figure 4: Top 20 Most Frequent Words",
         "Overall word frequency across both classes after preprocessing."),
        ("Visualization 5 - Model Accuracy Comparison.png",
         "Figure 5: Model Accuracy Comparison",
         "BoW + Naive Bayes achieves the highest accuracy at 98.12%."),
        ("Visualization 6 - Best Model Confusion Matrix.png",
         "Figure 6: Confusion Matrix — BoW + Naive Bayes",
         "Only 21 total misclassifications out of 1,115 test samples."),
        ("Visualization 7 - Performance Metrics Comparison.png",
         "Figure 7: Performance Metrics Comparison",
         "Side-by-side comparison of accuracy, precision, recall and F1 across all models."),
    ]

    for fname, title, caption in viz_list:
        st.markdown(f"<div class='section-header'>{title}</div>", unsafe_allow_html=True)
        st.image(f"visualizations/{fname}", use_container_width=True)
        st.markdown(
            f"<p style='color:#9C4DB8; font-size:0.85rem; font-style:italic; "
            f"text-align:center; font-weight:600;'>{caption}</p>",
            unsafe_allow_html=True
        )
        st.markdown("<br/>", unsafe_allow_html=True)


# ══════════════════════════════════════════
# MODEL INFO
# ══════════════════════════════════════════
elif page == "🤖 Model Info":

    st.markdown("<div class='section-header'>🤖 Model Information</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='best-model-banner'>
        <h3 style='margin:0; font-weight:800;'>⭐ Best Model: BoW + Naive Bayes</h3>
        <p style='margin:8px 0 0 0; font-size:0.95rem; color:#FFD6E7;'>
            Highest accuracy (98.12%) &nbsp;·&nbsp; Best F1 Score (0.9307)
            &nbsp;·&nbsp; Lowest misclassification rate
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>📊 Performance Comparison</div>", unsafe_allow_html=True)

    results = pd.DataFrame({
        "Model"    : ["BoW + Naive Bayes ⭐", "BoW + KNN",
                      "TF-IDF + Naive Bayes",  "TF-IDF + KNN"],
        "Accuracy" : [0.9812, 0.9444, 0.9740, 0.9444],
        "Precision": [0.9156, 1.0000, 0.9918, 1.0000],
        "Recall"   : [0.9463, 0.5839, 0.8121, 0.5839],
        "F1 Score" : [0.9307, 0.7373, 0.8930, 0.7373],
    })
    st.dataframe(
        results.style
            .highlight_max(subset=["Accuracy", "Recall", "F1 Score"], color="#E8D5F5")
            .highlight_max(subset=["Precision"], color="#FFD6E7"),
        use_container_width=True
    )

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>⚙️ Pipeline Details</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📁 Dataset**")
        for tag in ["SMS Spam Collection", "5,572 messages", "87% Ham · 13% Spam"]:
            st.markdown(f"<span class='stat-pill'>{tag}</span>", unsafe_allow_html=True)

        st.markdown("<br/>**🧹 Preprocessing Steps**", unsafe_allow_html=True)
        for step in ["Lowercasing", "Punctuation removal",
                     "Stop word removal", "Porter Stemming"]:
            st.markdown(f"<span class='chip'>✔ {step}</span>", unsafe_allow_html=True)

    with col2:
        st.markdown("**🔢 Feature Extraction**")
        for tag in ["Bag of Words (BoW)", "TF-IDF"]:
            st.markdown(f"<span class='stat-pill'>{tag}</span>", unsafe_allow_html=True)

        st.markdown("<br/>**🤖 Classifiers**", unsafe_allow_html=True)
        for tag in ["Naive Bayes", "K-Nearest Neighbours"]:
            st.markdown(f"<span class='stat-pill'>{tag}</span>", unsafe_allow_html=True)

        st.markdown("<br/>**✂️ Train-Test Split**", unsafe_allow_html=True)
        for tag in ["80% Training", "20% Testing"]:
            st.markdown(f"<span class='stat-pill'>{tag}</span>", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    with st.expander("📖 Why does BoW + Naive Bayes perform best?"):
        st.write("""
        **Naive Bayes** works extremely well with text because:
        - It calculates the probability of each word belonging to spam or ham
        - **Bag of Words** raw word counts align naturally with this probabilistic framework
        - It handles high-dimensional sparse text features efficiently
        - Even though it assumes word independence, it performs surprisingly well in practice

        **KNN** underperforms here because:
        - High-dimensional text features cause the "curse of dimensionality"
        - Distance metrics become unreliable in sparse feature spaces
        - The class imbalance (87:13) pushes KNN to predict ham by default
        """)


# ══════════════════════════════════════════
# HISTORY
# ══════════════════════════════════════════
elif page == "📋 History":

    st.markdown("<div class='section-header'>📋 Prediction History</div>", unsafe_allow_html=True)

    if not st.session_state.history:
        st.info("📭 No predictions yet. Go to **Text Analyzer** to analyze a message.")
    else:
        history_df = pd.DataFrame(st.session_state.history)

        n_total = len(history_df)
        n_spam  = int((history_df["Result"] == "🚨 SPAM").sum())
        n_ham   = n_total - n_spam

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class='metric-card'>
                <h2>{n_total}</h2><p>Total Analyzed</p></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class='metric-card'>
                <h2 style='color:#D63A8A;'>{n_spam}</h2>
                <p>Spam Detected</p></div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class='metric-card'>
                <h2 style='color:#9C4DB8;'>{n_ham}</h2>
                <p>Ham Messages</p></div>""", unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)
        st.dataframe(history_df, use_container_width=True)

        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
