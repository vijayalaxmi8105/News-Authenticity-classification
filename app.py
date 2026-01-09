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

# ---------- SIDEBAR ----------
st.sidebar.markdown("## 📌 Project Overview")
st.sidebar.write("**Project Title:**")
st.sidebar.write("News Authenticity Analyzer")

st.sidebar.write("**Domain:**")
st.sidebar.write("Artificial Intelligence / Machine Learning")

st.sidebar.write("**Technique:**")
st.sidebar.write("Natural Language Processing (TF-IDF)")

st.sidebar.write("**Algorithm:**")
st.sidebar.write("Logistic Regression")

st.sidebar.write("**Model Accuracy:**")
st.sidebar.write("~98.5%")

st.sidebar.write("**Deployment:**")
st.sidebar.write("Streamlit (Localhost)")

st.sidebar.markdown("---")
 


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
analyze = st.button("Analyze News")

if analyze:
    if news_text.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing news content..."):
            data = vectorizer.transform([news_text])
            prediction = model.predict(data)
            confidence = max(model.predict_proba(data)[0]) * 100
            word_count = len(news_text.split())

        st.write("---")
        st.info(f"📝 **Word Count:** {word_count}")
        st.info(f"📊 **Confidence Score:** {confidence:.2f}%")

        if prediction[0] == 0:
            st.error("🚨 **Fake News Detected**")
            st.caption("The content shows linguistic patterns commonly found in fake or misleading news.")
        else:
            st.success("✅ **This News Appears to be Real**")
            st.caption("The content aligns with patterns observed in authentic news articles.")


st.markdown('</div>', unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown(
    '<div class="footer">Built with Machine Learning & NLP | Streamlit UI</div>',
    unsafe_allow_html=True
)
