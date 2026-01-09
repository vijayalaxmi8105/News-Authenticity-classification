import streamlit as st
import pickle

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="News Authenticity Analyzer",
    page_icon="📰",
    layout="centered"
)

# ---------- LOAD MODEL ----------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ---------- CUSTOM CSS (AESTHETIC MAGIC) ----------
st.markdown("""
<style>
.main {
    background-color: #fafafa;
}
.title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
}
.subtitle {
    text-align: center;
    color: #6c757d;
    margin-bottom: 30px;
}
.card {
    background-color: white;
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
}
.footer {
    text-align: center;
    color: gray;
    font-size: 13px;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown('<div class="title">📰 News Authenticity Analyzer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Intelligent News Authenticity Classification Using Machine Learning</div>',
    unsafe_allow_html=True
)

# ---------- MAIN CARD ----------
st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("🔍 Analyze News Article")
news_text = st.text_area(
    "Paste the news content below",
    height=180,
    placeholder="Enter a news article here..."
)

analyze = st.button(" Analyze News")

if analyze:
    if news_text.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        data = vectorizer.transform([news_text])
        prediction = model.predict(data)

        if prediction[0] == 0:
            st.error("🚨 **Fake News Detected**")
            st.caption("The text shows patterns commonly associated with fake or misleading news.")
        else:
            st.success("✅ **This News Appears to be Real**")
            st.caption("The text matches patterns typically found in authentic news articles.")

st.markdown('</div>', unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown(
    '<div class="footer">Built with Machine Learning & NLP | Streamlit UI</div>',
    unsafe_allow_html=True
)
