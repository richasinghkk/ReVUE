import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st
import joblib
import requests
from bs4 import BeautifulSoup
#import speech_recognition as sr

# Import scraper functions
from src.scrapers.movie_scraper import search_movies, analyze_movie
from src.preprocessing.text_cleaner import clean_text


# ----------------------------------
# Streamlit Page Settings
# ----------------------------------
st.set_page_config(page_title="ReVUE (Movie Review Analyzer)", layout="centered")

# ----------------------------------
# UI Styling
# ----------------------------------
STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

[data-testid="stAppViewContainer"] > .main {
  background: linear-gradient(135deg, #f4e8ff, #e6f7ff, #f8ffe8);
  min-height: 100vh;
  padding: 3rem 1rem;
}

.glass {
  background: rgba(255,255,255,0.55);
  padding: 28px;
  border-radius: 18px;
  backdrop-filter: blur(12px);
  border: 1px solid rgba(255,255,255,0.5);
  max-width: 950px;
  margin: auto;
  margin-top: 2.5rem;
}

.title {
  font-size: 46px;
  font-weight: 800;
  background: linear-gradient(90deg, #6366f1, #ec4899, #22c55e);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  text-align: center;
}

.result {
  background: rgba(255,255,255,0.75);
  padding: 18px;
  border-radius: 12px;
  margin-top: 12px;
  border: 1px solid rgba(255,255,255,0.5);
  text-align: center;
}

.sentiment-label {
  font-size: 24px;
  font-weight: 700;
}
</style>
"""

st.markdown(STYLE, unsafe_allow_html=True)

# ----------------------------------
# Header
# ----------------------------------
st.markdown("<div class='title'>🎬 ReVUE</div>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; color:#475569; margin-bottom:8px;'>Movie Review Analyzer</div>", unsafe_allow_html=True)

# ----------------------------------
# Load ML Model
# ----------------------------------
MODEL_PATH = "saved_models/tfidf_model.pkl"
pipeline = joblib.load(MODEL_PATH)
vectorizer = pipeline["vectorizer"]
model = pipeline["model"]


# ----------------------------------
# Sentiment Analysis Function
# ----------------------------------
def analyze_sentiment(text):
    cleaned = clean_text(text)
    prob = model.predict_proba(vectorizer.transform([cleaned]))[:, 1][0]

    if prob > 0.60:
        label = "Positive"
        advice = "🎉 Recommended — You should watch this movie!"
    elif prob > 0.40:
        label = "Mixed"
        advice = "🤔 Mixed — audience response is average."
    else:
        label = "Negative"
        advice = "❌ Not Recommended — audience didn’t like it."

    stars = 5 if prob > 0.85 else 4 if prob > 0.70 else 3 if prob > 0.55 else 2 if prob > 0.40 else 1

    return label, float(prob), stars, advice


# ----------------------------------
# Main UI Tabs
# ----------------------------------
st.markdown("<div class='glass'>", unsafe_allow_html=True)
tab1, tab_imdb, tab2, tab3, tab4 = st.tabs(
    ["✏️ Type Review", "🎞 IMDb Movie Analyzer", "📄 File Upload", "🔗 Paste URL", "🎤 Voice Input"]
)


# =====================================================================================
# ✏️ TYPED REVIEW TAB
# =====================================================================================
with tab1:
    txt = st.text_area("Write your review here...", height=160)
    if st.button("Analyze Typed Review"):
        if txt.strip():
            label, prob, stars, advice = analyze_sentiment(txt)
            st.markdown(f"""
                <div class="result">
                  <div class="sentiment-label">{label}</div>
                  <p><b>Positive Probability:</b> {prob:.2f}</p>
                  <div style="font-size:20px;">{"★"*stars + "☆"*(5-stars)}</div>
                  <p>{advice}</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Please enter a review.")


# =====================================================================================
# 🎞 IMDb MOVIE ANALYZER TAB (NOW WITH RECOMMENDATIONS)
# =====================================================================================
with tab_imdb:

    if "search_results" not in st.session_state:
        st.session_state.search_results = []

    st.subheader("Search IMDb by movie name")

    movie_name = st.text_input("Enter movie name:")
    num_reviews = st.slider("Reviews to scrape", 10, 100, 50)

    # SEARCH BUTTON
    if st.button("Search Movie"):
        st.session_state.search_results = search_movies(movie_name)

        if len(st.session_state.search_results) == 0:
            st.error("No movies found on TMDb.")

    # SHOW DROPDOWN WHEN RESULTS EXIST
    if len(st.session_state.search_results) > 0:

        selected_label = st.selectbox(
            "Select the correct movie",
            [f"{m['title']} ({m['year']})" for m in st.session_state.search_results]
        )

        selected_movie = next(
            m for m in st.session_state.search_results
            if f"{m['title']} ({m['year']})" == selected_label
        )

        # Movie Poster + Basic Details
        st.markdown(f"""
            <div style="text-align:center;">
                <img src="{selected_movie['poster']}" width="230" style="border-radius:12px;"><br><br>
                <h2>{selected_movie['title']} ({selected_movie['year']})</h2>
            </div>
        """, unsafe_allow_html=True)

        # ANALYZE BUTTON
        if st.button("Analyze Selected Movie"):

            movie_data, reviews, similar, recommended = analyze_movie(
                selected_movie, max_reviews=num_reviews
            )

            # Extra details
            st.markdown(f"""
                <div style="text-align:center;">
                    <b>Genres:</b> {", ".join(movie_data['genres']) if movie_data['genres'] else "N/A"}<br>
                    <b>IMDb ID:</b> {movie_data['imdb_id'] or "N/A"}<br><br>
                </div>
            """, unsafe_allow_html=True)

            # Reviews Sentiment Summary
            if len(reviews) == 0:
                st.warning("No reviews found on IMDb.")
            else:
                sentiments, probs = [], []
                for r in reviews:
                    lbl, pr, _, _ = analyze_sentiment(r)
                    sentiments.append(lbl)
                    probs.append(pr)

                avg_prob = sum(probs) / len(probs)

                st.write("### 🎭 Sentiment Summary")
                st.write(f"👍 Positive: {sentiments.count('Positive')}")
                st.write(f"😐 Mixed: {sentiments.count('Mixed')}")
                st.write(f"👎 Negative: {sentiments.count('Negative')}")

                # Final Recommendation
                final_label = "Positive" if avg_prob > 0.60 else "Mixed" if avg_prob > 0.40 else "Negative"
                stars = 5 if avg_prob > 0.85 else 4 if avg_prob > 0.70 else 3 if avg_prob > 0.55 else 2 if avg_prob > 0.40 else 1

                st.markdown(f"""
                    <div class="result">
                        <div class="sentiment-label">{final_label}</div>
                        <div style="font-size:22px;">{"★"*stars + "☆"*(5-stars)}</div>
                        <p>Audience response is generally: {final_label}</p>
                    </div>
                """, unsafe_allow_html=True)

            # ============================================================
            # 🎯 SIMILAR MOVIES RECOMMENDATION
            # ============================================================
            st.write("---")
            st.write("## 🎬 Similar Movies (Content-Based Recommendation)")

            if len(similar) == 0:
                st.write("No similar movies found.")
            else:
                cols = st.columns(3)
                idx = 0
                for movie in similar:
                    with cols[idx % 3]:
                        st.image(movie['poster'], width=150)
                        st.markdown(f"**{movie['title']} ({movie['year']})**")
                    idx += 1

            # ============================================================
            # 🌟 TMDB Recommended Movies
            # ============================================================
            st.write("---")
            st.write("## 🌟 Recommended Movies (TMDB AI Suggestions)")

            if len(recommended) == 0:
                st.write("No recommendations available.")
            else:
                cols = st.columns(3)
                idx = 0
                for movie in recommended:
                    with cols[idx % 3]:
                        st.image(movie['poster'], width=150)
                        st.markdown(f"**{movie['title']} ({movie['year']})**")
                    idx += 1


# =====================================================================================
# 📄 FILE UPLOAD TAB
# =====================================================================================
with tab2:
    uploaded = st.file_uploader("Upload TXT / PDF / DOCX", type=["txt", "pdf", "docx"])
    if uploaded and st.button("Analyze File Review"):
        try:
            if uploaded.name.endswith(".txt"):
                content = uploaded.read().decode()
            elif uploaded.name.endswith(".pdf"):
                import PyPDF2
                reader = PyPDF2.PdfReader(uploaded)
                content = " ".join([p.extract_text() for p in reader.pages])
            else:
                import docx
                doc = docx.Document(uploaded)
                content = " ".join([p.text for p in doc.paragraphs])

            label, prob, stars, advice = analyze_sentiment(content)
            st.write(f"**{label}** — {advice} (Score: {prob:.2f})")

        except Exception as e:
            st.error(f"Error reading file: {e}")


# =====================================================================================
# 🔗 URL REVIEW TAB
# =====================================================================================
with tab3:
    url = st.text_input("Paste URL")
    if st.button("Fetch & Analyze"):
        try:
            html = requests.get(url).text
            soup = BeautifulSoup(html, "html.parser")
            text = " ".join([p.get_text() for p in soup.find_all("p")])

            label, prob, stars, advice = analyze_sentiment(text)
            st.write(f"**{label}** — {advice} (Score: {prob:.2f})")

        except Exception as e:
            st.error(f"URL fetch failed: {e}")


# =====================================================================================
# 🎤 VOICE INPUT TAB
# =====================================================================================
with tab4:
    if st.button("Record Voice Review"):
        try:
            r = sr.Recognizer()
            with sr.Microphone() as mic:
                st.info("🎙️ Listening...")
                audio = r.listen(mic)

            text = r.recognize_google(audio)
            st.success(f"You said: {text}")

            label, prob, stars, advice = analyze_sentiment(text)
            st.write(f"**{label}** — {advice} (Score: {prob:.2f})")

        except Exception:
            st.error("Could not process voice input.")


st.markdown("</div>", unsafe_allow_html=True)
