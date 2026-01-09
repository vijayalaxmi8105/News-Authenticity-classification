import streamlit as st
import pickle

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="News Authenticity Analyzer",
    page_icon="📰",
    layout="wide"
)

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_models():
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_models()

# ---------- SESSION STATE ----------
if "history" not in st.session_state:
    st.session_state.history = []

 #----------SIDEBAR ----------
with st.sidebar:
    st.markdown(" Analysis History")

    if "history" not in st.session_state:
        st.session_state.history = []

    if len(st.session_state.history) == 0:
        st.caption("No analyses yet.")
    else:
        for i, item in enumerate(reversed(st.session_state.history), 1):
            result = item.get("result", "N/A")
            confidence = item.get("confidence", "N/A")
            words = item.get("words", "N/A")

            st.markdown(
                f"**{i}. {result}**  \n"
                f"Confidence: {confidence}%  \n"
                f"Words: {words}"
            )



# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
    .main {
        max-width: 1000px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    .title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #6c757d;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .card {
        background: white;
        border-radius: 10px;
        padding: 2rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 2rem;
    }
    
    .stButton > button {
        width: 100%;
        background-color: #ff4b4b !important;
        color: white !important;
        font-weight: 500;
        border: none !important;
        padding: 0.75rem 1.5rem;
        border-radius: 5px;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #ff6b6b !important;
        transform: translateY(-1px);
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .result-metric {
        background: white;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        display: flex;
        align-items: center;
    }
    
    .result-icon {
        font-size: 1.5rem;
        margin-right: 1rem;
    }
    
    .result-content h4 {
        margin: 0 0 0.25rem 0;
        font-size: 1rem;
        color: #6c757d;
    }
    
    .result-content p {
        margin: 0;
        font-size: 1.25rem;
        font-weight: 600;
        color: #2c3e50;
    }
    
    .fake-news {
        border-left: 4px solid #ff4b4b;
    }
    
    .real-news {
        border-left: 4px solid #4CAF50;
    }
    
    .analysis-box {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1.5rem;
        margin-top: 1.5rem;
    }
    
    .analysis-box h3 {
        margin-top: 0;
        color: #2c3e50;
    }
    
    .analysis-box p {
        color: #6c757d;
        line-height: 1.6;
    }
    
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #e9ecef;
        color: #6c757d;
        font-size: 0.875rem;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
</style>
""", unsafe_allow_html=True)

# ---------- MAIN CONTENT ----------
st.markdown("# 📰 News Authenticity Analyzer")
st.markdown("""
<div class="subtitle">
    Intelligent News Authenticity Classification Using Machine Learning
</div>
""", unsafe_allow_html=True)

# Input Section
st.markdown("### 🔍 Analyze News Article")
st.markdown("Paste the news content below")

news_text = st.text_area(
    label="News Article Input",
    placeholder="Paste the news article here...",
    height=220,
    label_visibility="collapsed"
)


analyze = st.button("Analyze News")

if analyze and news_text.strip():
    with st.spinner("� Analyzing news content..."):
        # Preprocess and predict
        data = vectorizer.transform([news_text])
        prediction = model.predict(data)
        confidence = max(model.predict_proba(data)[0]) * 100
        
        # Save to history
        st.session_state.history.append({
            "text": news_text[:50],  # short preview
            "result": "Real" if prediction[0] == 1 else "Fake",
            "confidence": confidence
        })
        
        word_count = len(news_text.split())
    
    # Display results
    st.markdown("### Analysis Results")
    
    # Metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="result-metric">
            <div class="result-icon">📝</div>
            <div class="result-content">
                <h4>Word Count</h4>
                <p>{:,}</p>
            </div>
        </div>
        """.format(word_count), unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="result-metric">
            <div class="result-icon">📊</div>
            <div class="result-content">
                <h4>Confidence</h4>
                <p>{confidence:.1f}%</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        # Prediction Result
        is_fake = prediction[0] == 0

        result_class = "fake-news" if is_fake else "real-news"
        result_icon = "🚨" if is_fake else "✅"
        result_text = "Fake News Detected" if is_fake else "News Appears Authentic"

        analysis_text = (
            "The content analysis indicates characteristics that are statistically more common "
            "in potentially misleading or unverified news sources."
            if is_fake
            else
            "The content demonstrates linguistic patterns consistent with verified news sources."
        )

        note_text = (
            "Please verify information through trusted sources."
            if is_fake
            else
            "Always verify information through multiple reliable sources."
        )

        st.markdown(f"""
        <div class="result-metric {result_class}">
            <div class="result-icon">{result_icon}</div>
            <div class="result-content">
                <h4>Analysis Result</h4>
                <p>{result_text}</p>
            </div>
        </div>

        <div class="analysis-box">
            <h3>Analysis</h3>
            <p>{analysis_text}</p>
            <p><em>Note: This is an automated analysis. {note_text}</em></p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    Built with Machine Learning & NLP | Streamlit UI | Deployed Locally
</div>
""", unsafe_allow_html=True)
