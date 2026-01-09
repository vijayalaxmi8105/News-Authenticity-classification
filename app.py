import streamlit as st
import pickle

# Load model & vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📰 Fake News Detection System")
st.write("Enter a news article to check whether it is Fake or Real")

news_text = st.text_area("Paste the news content here:")

if st.button("Check News"):
    if news_text.strip() == "":
        st.warning("Please enter some text")
    else:
        data = vectorizer.transform([news_text])
        prediction = model.predict(data)

        if prediction[0] == 0:
            st.error("🚨 Fake News Detected")
        else:
            st.success("✅ This News is Real")
