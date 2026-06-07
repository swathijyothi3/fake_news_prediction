# ╔══════════════════════════════════════════════════════════════════╗
# ║          FakeScope AI  ·  Fake News Detection System            ║
# ║  Dataset   : ISOT Fake News (Fake.csv + True.csv)               ║
# ║  Models    : KNN + Bag-of-Words  /  KNN + TF-IDF                ║
# ║  Interface : Streamlit with full custom dark-sci-fi UI           ║
# ╚══════════════════════════════════════════════════════════════════╝

import warnings
warnings.filterwarnings("ignore")

import os, re, random
import streamlit as st
import joblib
import nltk
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.graph_objects as go
from collections import Counter

# ──────────────────────────────────────────────────────────────────
#  PAGE CONFIG  (must be the very first Streamlit call)
# ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FakeScope AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────
#  GLOBAL CSS  —  dark sci-fi / neon-glass theme
# ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Rajdhani:wght@500;600;700&family=Exo+2:wght@300;400;600&family=Space+Mono:wght@400;700&display=swap');

/* ── CSS Variables ── */
:root {
  --bg:        #04040f;
  --bg2:       #08082a;
  --cyan:      #00d4ff;
  --purple:    #8b5cf6;
  --green:     #10b981;
  --red:       #ef4444;
  --amber:     #f59e0b;
  --txt:       #e2e8f0;
  --muted:     #64748b;
  --glass:     rgba(255,255,255,0.035);
  --gborder:   rgba(255,255,255,0.08);
  --shadow:    0 24px 64px rgba(0,0,0,0.7),0 8px 24px rgba(0,0,0,0.5);
  --glow-c:    0 0 24px rgba(0,212,255,0.45),0 0 80px rgba(0,212,255,0.12);
  --glow-p:    0 0 24px rgba(139,92,246,0.45),0 0 80px rgba(139,92,246,0.12);
}

/* ── App Shell ── */
[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(ellipse 60% 45% at 10% 90%, rgba(0,212,255,.10) 0%, transparent 70%),
    radial-gradient(ellipse 55% 45% at 90% 10%, rgba(139,92,246,.10) 0%, transparent 70%),
    radial-gradient(ellipse 40% 35% at 50% 50%, rgba(16,185,129,.04) 0%, transparent 70%),
    linear-gradient(155deg, #04040f 0%, #07082a 55%, #0c0420 100%);
  min-height: 100vh;
}
/* scanline overlay */
[data-testid="stAppViewContainer"]::after{
  content:'';position:fixed;inset:0;pointer-events:none;z-index:9998;
  background:repeating-linear-gradient(0deg,transparent 0,transparent 3px,rgba(0,0,0,.07) 3px,rgba(0,0,0,.07) 4px);
}
[data-testid="stHeader"]      { background:transparent !important; }
[data-testid="stToolbar"]     { display:none !important; }
[data-testid="stSidebar"]     { display:none !important; }
section[data-testid="stSidebar"]+div {margin-left:0 !important;}
.main .block-container        { max-width:1320px; padding:1.5rem 2rem 4rem; }

/* ── HERO ── */
.hero-wrap{ text-align:center; padding:2.5rem 0 1.5rem; }
.hero-badge{
  display:inline-block;font-family:'Space Mono',monospace;font-size:.68rem;
  letter-spacing:.25em;color:var(--cyan);text-transform:uppercase;
  background:rgba(0,212,255,.07);border:1px solid rgba(0,212,255,.22);
  border-radius:100px;padding:.28rem 1.1rem;margin-bottom:1.1rem;
}
.hero-title{
  font-family:'Orbitron',sans-serif;font-size:clamp(2.4rem,5vw,4.4rem);
  font-weight:900;line-height:1.05;letter-spacing:.04em;
  background:linear-gradient(130deg,#ffffff 0%,var(--cyan) 42%,var(--purple) 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  background-clip:text;filter:drop-shadow(0 0 28px rgba(0,212,255,.28));
  margin-bottom:.6rem;
}
.hero-sub{
  font-family:'Exo 2',sans-serif;font-size:1rem;color:var(--muted);
  letter-spacing:.04em;max-width:580px;margin:0 auto 1rem;line-height:1.6;
}
.hero-line{
  height:1px;
  background:linear-gradient(90deg,transparent,var(--cyan),var(--purple),transparent);
  max-width:480px;margin:1.5rem auto 0;
}

/* ── GLASS PANEL ── */
.gpanel{
  background:var(--glass);backdrop-filter:blur(18px);
  -webkit-backdrop-filter:blur(18px);
  border:1px solid var(--gborder);border-radius:18px;padding:1.6rem 1.8rem;
  box-shadow:var(--shadow),inset 0 1px 0 rgba(255,255,255,.07);
  transition:transform .25s,box-shadow .25s;
}
.gpanel:hover{
  transform:translateY(-3px);
  box-shadow:0 32px 80px rgba(0,0,0,.8),0 12px 32px rgba(0,0,0,.6),
             inset 0 1px 0 rgba(255,255,255,.1);
}

/* ── SECTION LABEL ── */
.slabel{
  font-family:'Space Mono',monospace;font-size:.62rem;letter-spacing:.28em;
  color:var(--cyan);text-transform:uppercase;margin-bottom:.55rem;
  display:flex;align-items:center;gap:.5rem;
}
.slabel::before{content:'';display:inline-block;width:16px;height:1px;background:var(--cyan);}

/* ── VECTORIZER BUTTON AREA ── */
.vbtn-row{display:flex;gap:.9rem;margin:.4rem 0 1.1rem;}
.vbtn{
  flex:1;padding:1rem 1.2rem;border-radius:14px;text-align:center;
  border:1px solid rgba(255,255,255,.09);background:rgba(255,255,255,.03);
  transition:all .22s;cursor:pointer;
}
.vbtn.active{
  border-color:var(--cyan);background:rgba(0,212,255,.07);
  box-shadow:0 0 22px rgba(0,212,255,.22),inset 0 0 18px rgba(0,212,255,.05);
}
.vbtn .vnum{
  font-family:'Orbitron',sans-serif;font-size:.7rem;letter-spacing:.2em;
  color:var(--muted);margin-bottom:.3rem;
}
.vbtn.active .vnum{color:var(--cyan);}
.vbtn .vtitle{
  font-family:'Rajdhani',sans-serif;font-weight:700;font-size:1.05rem;
  color:var(--txt);letter-spacing:.06em;
}
.vbtn .vdesc{
  font-family:'Exo 2',sans-serif;font-size:.75rem;color:var(--muted);
  margin-top:.25rem;line-height:1.4;
}

/* ── TIP BOX ── */
.tipbox{
  background:rgba(0,212,255,.04);border-left:3px solid var(--cyan);
  border-radius:0 12px 12px 0;padding:.75rem 1rem;
  font-family:'Exo 2',sans-serif;font-size:.82rem;
  color:var(--muted);line-height:1.5;margin:.5rem 0 1rem;
}
.tipbox strong{color:var(--cyan);}

/* ── STREAMLIT WIDGET OVERRIDES ── */
[data-testid="stTextArea"] textarea,
[data-testid="stTextInput"] input{
  background:#0f172a !important;
  border:1px solid rgba(255,255,255,.1) !important;
  border-radius:11px !important;color:#ffffff !important;
  font-family:'Exo 2',sans-serif !important;font-size:.93rem !important;
  transition:border-color .2s,box-shadow .2s !important;
  caret-color:var(--cyan) !important;
}
[data-testid="stTextArea"] textarea:focus,
[data-testid="stTextInput"] input:focus{
  border-color:var(--cyan) !important;
  box-shadow:0 0 0 3px rgba(0,212,255,.1) !important;outline:none !important;
}
label[data-testid="stWidgetLabel"] p{
  font-family:'Rajdhani',sans-serif !important;font-weight:600 !important;
  font-size:.78rem !important;letter-spacing:.14em !important;
  color:var(--muted) !important;text-transform:uppercase !important;
}
/* Placeholder text */
[data-testid="stTextArea"] textarea::placeholder,
[data-testid="stTextInput"] input::placeholder{ color:#334155 !important; }

/* ── PREDICT BUTTON ── */
.stButton>button{
  background:linear-gradient(125deg,#00c6f7 0%,#7c3aed 100%) !important;
  border:none !important;border-radius:13px !important;
  color:#fff !important;font-family:'Orbitron',sans-serif !important;
  font-weight:700 !important;font-size:.85rem !important;
  letter-spacing:.12em !important;padding:.85rem 2rem !important;
  width:100% !important;text-transform:uppercase !important;
  box-shadow:0 4px 22px rgba(0,212,255,.28),0 2px 8px rgba(0,0,0,.35) !important;
  transition:all .22s !important;
}
.stButton>button:hover{
  transform:translateY(-2px) !important;
  box-shadow:0 10px 36px rgba(0,212,255,.42),0 4px 14px rgba(0,0,0,.45) !important;
}
.stButton>button:active{ transform:translateY(0) !important; }

/* ── RESULT CARD ── */
.rcard{
  border-radius:18px;padding:2.2rem 2rem;text-align:center;
  margin:.8rem 0;position:relative;overflow:hidden;
  box-shadow:var(--shadow);
}
.rcard.real{ background:linear-gradient(135deg,rgba(16,185,129,.11) 0%,rgba(5,150,105,.04) 100%);
             border:1px solid rgba(16,185,129,.3); }
.rcard.fake{ background:linear-gradient(135deg,rgba(239,68,68,.11) 0%,rgba(185,28,28,.04) 100%);
             border:1px solid rgba(239,68,68,.3); }

/* Rotating corner shimmer */
.rcard::before{
  content:'';position:absolute;inset:-50%;
  background:conic-gradient(transparent 270deg,rgba(255,255,255,.04) 360deg);
  animation:spin 5s linear infinite;pointer-events:none;
}
@keyframes spin{to{transform:rotate(360deg);}}

.r-icon{font-size:2.2rem;display:block;animation:pulse 2.2s ease-in-out infinite;margin-bottom:.3rem;}
@keyframes pulse{0%,100%{transform:scale(1);opacity:1;}50%{transform:scale(1.12);opacity:.85;}}
.r-verdict{
  font-family:'Orbitron',sans-serif;font-size:clamp(2.2rem,5vw,4.2rem);
  font-weight:900;letter-spacing:.14em;line-height:1;margin:.3rem 0;
}
.rcard.real .r-verdict{ color:#10b981;text-shadow:0 0 40px rgba(16,185,129,.55); }
.rcard.fake .r-verdict{ color:#ef4444;text-shadow:0 0 40px rgba(239,68,68,.55); }
.r-sub{ font-family:'Exo 2',sans-serif;font-size:.9rem;color:var(--muted);margin-top:.4rem;}
.r-chip{
  display:inline-block;font-family:'Space Mono',monospace;font-size:.75rem;
  padding:.28rem .95rem;border-radius:100px;margin-top:.8rem;
}
.rcard.real .r-chip{background:rgba(16,185,129,.13);color:#10b981;border:1px solid rgba(16,185,129,.28);}
.rcard.fake .r-chip{background:rgba(239,68,68,.13);color:#ef4444;border:1px solid rgba(239,68,68,.28);}

/* ── STATS ROW ── */
.srow{ display:flex;gap:.75rem;margin:.9rem 0; }
.schip{
  flex:1;background:var(--glass);border:1px solid var(--gborder);
  border-radius:12px;padding:.85rem .6rem;text-align:center;
}
.schip .sv{ font-family:'Orbitron',sans-serif;font-size:1.2rem;font-weight:700;color:var(--cyan);display:block; }
.schip .sl{ font-family:'Space Mono',monospace;font-size:.58rem;color:var(--muted);
            letter-spacing:.1em;text-transform:uppercase;display:block;margin-top:.18rem; }

/* ── SECTION HEADING ── */
.sechead{
  font-family:'Rajdhani',sans-serif;font-weight:700;font-size:.75rem;
  letter-spacing:.22em;color:var(--muted);text-transform:uppercase;
  margin:1.2rem 0 .65rem;display:flex;align-items:center;gap:.55rem;
}
.sechead::after{content:'';flex:1;height:1px;background:var(--gborder);}

/* ── PLOTLY / MATPLOTLIB ── */
[data-testid="stPlotlyChart"]{ background:transparent !important; }
.js-plotly-plot .plotly .bg{ fill:transparent !important; }
[data-testid="stImage"] img{ border-radius:13px; }
figure{ background:transparent !important; }

/* ── INFO / WARNING BOX ── */
[data-testid="stInfo"]{
  background:rgba(0,212,255,.05) !important;
  border:1px solid rgba(0,212,255,.18) !important;
  border-radius:11px !important;color:var(--txt) !important;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar{ width:4px;height:4px; }
::-webkit-scrollbar-track{ background:var(--bg); }
::-webkit-scrollbar-thumb{ background:var(--cyan);border-radius:4px; }

/* ── PLACEHOLDER PANEL ── */
.placeholder{
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  min-height:340px;border-radius:18px;
  background:var(--glass);border:1px solid var(--gborder);
  box-shadow:var(--shadow);gap:.7rem;
}
.ph-icon{ font-size:3rem;opacity:.18; }
.ph-title{ font-family:'Orbitron',sans-serif;font-size:.75rem;
           letter-spacing:.2em;color:#1e293b;text-transform:uppercase; }
.ph-sub{ font-family:'Exo 2',sans-serif;font-size:.78rem;color:#1e293b; }

/* ── FOOTER ── */
.footer{
  text-align:center;font-family:'Space Mono',monospace;font-size:.6rem;
  color:var(--muted);letter-spacing:.16em;padding:2.5rem 0 .5rem;
  border-top:1px solid var(--gborder);margin-top:3.5rem;opacity:.6;
}

/* ── WC NOTE ── */
.wcnote{
  font-family:'Space Mono',monospace;font-size:.6rem;color:var(--muted);
  letter-spacing:.1em;text-align:center;margin-top:.4rem;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
#  NLTK SETUP  (cached so it only runs once)
# ──────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def setup_nltk():
    for pkg, path in [("stopwords","corpora/stopwords"),
                      ("wordnet","corpora/wordnet"),
                      ("omw-1.4","corpora/omw-1.4")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)
    return True

setup_nltk()
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

STOP_WORDS = set(stopwords.words("english"))
_lemma     = WordNetLemmatizer()


# ──────────────────────────────────────────────────────────────────
#  TEXT PREPROCESSING  (identical to training pipeline)
# ──────────────────────────────────────────────────────────────────
def preprocess(text: str) -> str:
    """Lowercase → strip URLs/HTML/non-alpha → lemmatize → remove stopwords."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = [_lemma.lemmatize(w)
              for w in text.split()
              if w not in STOP_WORDS and len(w) > 2]
    return " ".join(tokens)


# ──────────────────────────────────────────────────────────────────
#  MODEL LOADING
# ──────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    """Load BOW and TF-IDF model bundles from disk."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bow_bundle   = joblib.load("model_bow.pkl")
        tfidf_bundle = joblib.load("model_tfidf.pkl")
    return bow_bundle, tfidf_bundle


# ──────────────────────────────────────────────────────────────────
#  VISUALISATION HELPERS
# ──────────────────────────────────────────────────────────────────
_WC_PALETTE = ["#00d4ff","#8b5cf6","#10b981","#f59e0b","#06b6d4",
               "#a78bfa","#34d399","#fbbf24","#38bdf8","#c084fc"]

def _wc_color(*args, **kwargs):
    return random.choice(_WC_PALETTE)

def make_wordcloud(clean_text: str):
    """Generate a transparent-background word cloud figure."""
    if not clean_text.strip() or len(clean_text.split()) < 5:
        return None
    wc = WordCloud(
        width=900, height=400,
        background_color=None, mode="RGBA",
        color_func=_wc_color,
        max_words=80,
        prefer_horizontal=0.85,
        relative_scaling=0.45,
        min_font_size=11,
        collocations=False,
        max_font_size=120,
    ).generate(clean_text)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    return fig


def make_gauge(prob_real: float, label: str) -> go.Figure:
    """Plotly gauge: left = Fake (red), right = Real (green)."""
    is_real = label == "REAL"
    bar_col = "#10b981" if is_real else "#ef4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(prob_real * 100, 1),
        number={"suffix": "%",
                "font": {"family": "Orbitron", "size": 26, "color": "#e2e8f0"}},
        delta={"reference": 50,
               "increasing": {"color": "#10b981"},
               "decreasing": {"color": "#ef4444"},
               "font": {"size": 12}},
        title={"text": "REAL NEWS PROBABILITY",
               "font": {"family": "Space Mono", "size": 9, "color": "#64748b"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1,
                     "tickfont": {"family": "Space Mono", "size": 8,
                                  "color": "#64748b"}},
            "bar": {"color": bar_col, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  30], "color": "rgba(239,68,68,.14)"},
                {"range": [30, 50], "color": "rgba(245,158,11,.08)"},
                {"range": [50, 70], "color": "rgba(245,158,11,.08)"},
                {"range": [70,100], "color": "rgba(16,185,129,.14)"},
            ],
            "threshold": {
                "line": {"color": "#00d4ff", "width": 3},
                "thickness": 0.82,
                "value": 50
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        height=230,
        margin=dict(l=20, r=20, t=44, b=5),
    )
    return fig


def make_top_words(clean_text: str, n: int = 15) -> go.Figure | None:
    """Horizontal bar chart of top-N word frequencies."""
    words  = clean_text.split()
    counts = Counter(words).most_common(n)
    if len(counts) < 3:
        return None
    labels, values = zip(*reversed(counts))

    # Gradient colours from cyan → purple
    colors = []
    for i, _ in enumerate(labels):
        t = i / max(len(labels) - 1, 1)
        r = int(0   + t * 139)
        g = int(212 - t * 120)
        b = int(255 - t * 9)
        colors.append(f"rgba({r},{g},{b},.88)")

    fig = go.Figure(go.Bar(
        x=list(values),
        y=list(labels),
        orientation="h",
        marker=dict(color=colors,
                    line=dict(color="rgba(255,255,255,.06)", width=.5)),
        text=[str(v) for v in values],
        textposition="outside",
        textfont=dict(family="Space Mono", size=9, color="#64748b"),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Exo 2", color="#e2e8f0"),
        xaxis=dict(visible=False, showgrid=False),
        yaxis=dict(showgrid=False,
                   tickfont=dict(family="Exo 2", size=11, color="#94a3b8")),
        margin=dict(l=10, r=55, t=6, b=6),
        height=max(280, len(labels) * 25),
        bargap=0.38,
    )
    return fig


# ──────────────────────────────────────────────────────────────────
#  LOAD MODELS
# ──────────────────────────────────────────────────────────────────
try:
    bow_bundle, tfidf_bundle = load_models()
    _models_ok = True
except FileNotFoundError as _e:
    _models_ok = False
    st.error(
        f"Model files not found: {_e}.  "
        "Make sure **model_bow.pkl** and **model_tfidf.pkl** are in the same "
        "folder as app.py."
    )


# ══════════════════════════════════════════════════════════════════
#  LAYOUT
# ══════════════════════════════════════════════════════════════════

# ── Hero ─────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <div class="hero-badge">NATURAL LANGUAGE PROCESSING · KNN CLASSIFIER · ISOT DATASET</div>
  <h1 class="hero-title">FakeScope AI</h1>
  <p class="hero-sub">
    Detect fake news with machine learning. Trained on 44,000+ articles using
    Bag-of-Words and TF-IDF vectorisation with K-Nearest Neighbours classification.
  </p>
  <div class="hero-line"></div>
</div>
""", unsafe_allow_html=True)

if not _models_ok:
    st.stop()

# ── Session State ─────────────────────────────────────────────────
if "choice" not in st.session_state:
    st.session_state.choice = "BOW"
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ── Two-Column Layout ─────────────────────────────────────────────
left, right = st.columns([1.08, 1], gap="large")

# ╔════════════════════════════ LEFT COLUMN ══════════════════════╗
with left:

    # ── Step 1: Vectorizer Selection ─────────────────────────────
    st.markdown('<p class="slabel">Step 1 — Choose a Vectorizer</p>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="small")
    with c1:
        bow_label = (
            "✦ BOW  (Active)" if st.session_state.choice == "BOW"
            else "⬡ BOW"
        )
        if st.button(bow_label, key="btn_bow", use_container_width=True):
            st.session_state.choice = "BOW"
            st.session_state.last_result = None
            st.rerun()
    with c2:
        tfidf_label = (
            "✦ TF-IDF  (Active)" if st.session_state.choice == "TFIDF"
            else "◈ TF-IDF"
        )
        if st.button(tfidf_label, key="btn_tfidf", use_container_width=True):
            st.session_state.choice = "TFIDF"
            st.session_state.last_result = None
            st.rerun()

    # Description of active choice
    if st.session_state.choice == "BOW":
        tip_html = (
            "<strong>Option 1 — Bag of Words (BOW)</strong><br>"
            "Counts how many times each word appears in the article. Simple, fast, "
            "and effective. Best for catching obvious patterns and repeated keywords "
            "often seen in fake news. <em>Accuracy on test set: ~80%</em>"
        )
    else:
        tip_html = (
            "<strong>Option 2 — TF-IDF</strong><br>"
            "Weights words by how unique they are across the entire dataset. "
            "Common words get lower scores; rare, meaningful words get higher scores. "
            "Captures writing style and domain-specific language. <em>Accuracy on test set: ~70%</em>"
        )
    st.markdown(f'<div class="tipbox">{tip_html}</div>', unsafe_allow_html=True)

    # ── Step 2: Article Input ─────────────────────────────────────
    st.markdown('<p class="slabel">Step 2 — Enter the News Article</p>',
                unsafe_allow_html=True)

    title_in = st.text_input(
        "News Headline  (optional — helps accuracy)",
        placeholder="e.g.  Senate approves bipartisan infrastructure bill ...",
        key="title_input",
    )

    article_in = st.text_area(
        "Article Body  *  (required)",
        placeholder=(
            "Paste the full article text here — at least 80 to 100 words.\n\n"
            "Headlines alone are not enough; the classifier needs body text.\n"
            "This model is optimised for English-language political news articles."
        ),
        height=215,
        key="article_input",
    )

    st.markdown("")
    predict_btn = st.button("Analyze Article", use_container_width=True)

# ╔════════════════════════════ RIGHT COLUMN ═════════════════════╗
with right:

    # ── No prediction yet ─────────────────────────────────────────
    if not predict_btn and st.session_state.last_result is None:
        st.markdown("""
        <div class="placeholder">
          <div class="ph-icon">🔬</div>
          <div class="ph-title">Awaiting Input</div>
          <div class="ph-sub">Enter an article on the left and click Analyze</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── Validate input ────────────────────────────────────────
        if predict_btn:
            if not article_in.strip():
                st.warning("Please enter some article text before analysing.")
                st.stop()

            raw_combined = (title_in + " " + article_in).strip()
            clean        = preprocess(raw_combined)

            if len(clean.split()) < 20:
                st.warning(
                    "The article is too short for reliable classification. "
                    "KNN needs enough vocabulary overlap with its training data. "
                    "Please paste at least 80-100 words of article content "
                    "(headlines alone are not sufficient)."
                )
                st.stop()

            # Select model bundle
            bundle = (bow_bundle if st.session_state.choice == "BOW"
                      else tfidf_bundle)
            vec    = bundle["vectorizer"]
            model  = bundle["model"]

            with st.spinner("Analysing ..."):
                X_vec  = vec.transform([clean])
                pred   = model.predict(X_vec)[0]
                proba  = model.predict_proba(X_vec)[0]   # [p_fake, p_real]

            st.session_state.last_result = {
                "pred":      pred,
                "proba":     proba,
                "clean":     clean,
                "raw_len":   len(raw_combined.split()),
                "clean_len": len(clean.split()),
                "choice":    st.session_state.choice,
            }

        # ── Draw result from session state ────────────────────────
        r = st.session_state.last_result
        if r is None:
            st.stop()

        pred       = r["pred"]
        proba      = r["proba"]
        clean      = r["clean"]
        label      = "REAL" if pred == 1 else "FAKE"
        css_cls    = "real" if pred == 1 else "fake"
        icon       = "✅" if pred == 1 else "🚨"
        confidence = max(proba[0], proba[1])
        prob_real  = proba[1]

        # Result card
        st.markdown(f"""
        <div class="rcard {css_cls}">
          <span class="r-icon">{icon}</span>
          <div class="r-verdict">{label}</div>
          <p class="r-sub">
            This article is classified as <strong>{label.lower()} news</strong>
            using the <strong>{r['choice']}</strong> vectorizer
          </p>
          <span class="r-chip">Confidence: {confidence*100:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)

        # Stats row
        st.markdown(f"""
        <div class="srow">
          <div class="schip">
            <span class="sv">{r['raw_len']:,}</span>
            <span class="sl">Input Words</span>
          </div>
          <div class="schip">
            <span class="sv">{r['clean_len']:,}</span>
            <span class="sl">After Cleaning</span>
          </div>
          <div class="schip">
            <span class="sv">{proba[0]*100:.0f}%</span>
            <span class="sl">Fake Score</span>
          </div>
          <div class="schip">
            <span class="sv">{prob_real*100:.0f}%</span>
            <span class="sl">Real Score</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Confidence gauge
        st.plotly_chart(
            make_gauge(prob_real, label),
            use_container_width=True,
            config={"displayModeBar": False},
        )


# ══════════════════════════════════════════════════════════════════
#  VISUALISATION SECTION  (full-width, appears after prediction)
# ══════════════════════════════════════════════════════════════════
r = st.session_state.last_result
if r is not None:
    clean = r["clean"]

    st.markdown(
        '<div class="hero-line" style="margin:2rem auto;"></div>',
        unsafe_allow_html=True
    )

    col_wc, col_bar = st.columns(2, gap="large")

    # ── Word Cloud ────────────────────────────────────────────────
    with col_wc:
        st.markdown('<p class="sechead">Word Cloud — Key Terms in Your Article</p>',
                    unsafe_allow_html=True)
        fig_wc = make_wordcloud(clean)
        if fig_wc:
            st.pyplot(fig_wc, use_container_width=True)
            plt.close(fig_wc)
            st.markdown(
                '<p class="wcnote">'
                'Larger word = appears more often · stopwords removed · colours are decorative'
                '</p>',
                unsafe_allow_html=True
            )
        else:
            st.info("Not enough unique words to generate a word cloud.")

    # ── Top-15 Words ──────────────────────────────────────────────
    with col_bar:
        st.markdown('<p class="sechead">Top 15 Words — Frequency After Preprocessing</p>',
                    unsafe_allow_html=True)
        fig_bar = make_top_words(clean, n=15)
        if fig_bar:
            st.plotly_chart(
                fig_bar,
                use_container_width=True,
                config={"displayModeBar": False},
            )
        else:
            st.info("Not enough words to chart after preprocessing.")

    # ── How it works ─────────────────────────────────────────────
    st.markdown('<p class="sechead">How It Works — Pipeline</p>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="tipbox" style="font-size:.83rem;line-height:1.7;">
      <strong>1. Text Cleaning:</strong> Lowercase → strip URLs, HTML tags, punctuation → keep only letters.<br>
      <strong>2. Stopword Removal:</strong> Common words ("the", "is", "at") are removed — they carry no signal.<br>
      <strong>3. Lemmatisation:</strong> Words are reduced to their base form: "running" → "run", "says" → "say".<br>
      <strong>4. Vectorisation:</strong>
        <em>BOW</em> counts word occurrences (top 5,000 words from training data) |
        <em>TF-IDF</em> weights each word by rarity across 39,000+ training articles.<br>
      <strong>5. KNN Classification:</strong> The 5 nearest training articles (by vocabulary overlap) vote on the label. Because the model is distance-based on sparse vectors, it performs best on articles of 80+ words that match political news vocabulary from the training data.
    </div>
    """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  FAKESCOPE AI  ·  ISOT FAKE NEWS DATASET  ·  KNN + BOW / TF-IDF  ·  BUILT WITH STREAMLIT
</div>
""", unsafe_allow_html=True)
