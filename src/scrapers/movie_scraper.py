import requests
from bs4 import BeautifulSoup

TMDB_API_KEY = "94e5e5f176b3fa9456b489de5d6ab9b9"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0 Safari/537.36"
}

TMDB_BASE = "https://api.themoviedb.org/3"


# ------------------------------------------------------
# 1) SEARCH MOVIES (FAST + ACCURATE)
# ------------------------------------------------------
def search_movies(movie_name):
    """Return list of movies with SAFE dictionary keys."""

    search_url = f"{TMDB_BASE}/search/movie?api_key={TMDB_API_KEY}&query={movie_name}"
    r = requests.get(search_url).json()

    if not r.get("results"):
        return []

    movies = []
    for m in r["results"]:
        movies.append({
            "title": m.get("title"),
            "year": m.get("release_date", "")[:4] if m.get("release_date") else "N/A",
            "tmdb_id": m.get("id"),
            "poster": (
                f"https://image.tmdb.org/t/p/w500{m['poster_path']}"
                if m.get("poster_path") else
                "https://via.placeholder.com/300x450?text=No+Image"
            ),

            # Keys added to avoid KeyError
            "genres": [],
            "imdb_id": None
        })

    return movies


# ------------------------------------------------------
# 2) GET FULL DETAILS (IMDB ID + GENRES)
# ------------------------------------------------------
def get_movie_details(tmdb_id):
    """Fetch IMDb ID + Genres only when the user selects the movie."""

    url = f"{TMDB_BASE}/movie/{tmdb_id}?api_key={TMDB_API_KEY}&append_to_response=external_ids"
    data = requests.get(url).json()

    imdb_id = data.get("external_ids", {}).get("imdb_id")
    genres = [g["name"] for g in data.get("genres", [])]

    return imdb_id, genres


# ------------------------------------------------------
# 3) UNIVERSAL IMDb REVIEW SCRAPER
# ------------------------------------------------------
def fetch_imdb_reviews(imdb_id, max_reviews=50):
    """Scrape reviews from ALL IMDb layouts."""

    if not imdb_id:
        return []

    url = f"https://www.imdb.com/title/{imdb_id}/reviews"
    reviews = []

    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")

        selectors = [
            ".text.show-more__control",
            ".review-container .text",
            ".ipc-html-content-inner-div",
            ".review-content .text",
        ]

        for sel in selectors:
            blocks = soup.select(sel)
            for b in blocks:
                txt = b.get_text(" ", strip=True)
                if txt and txt not in reviews:
                    reviews.append(txt)
                if len(reviews) >= max_reviews:
                    return reviews

    except Exception as e:
        print("IMDb scraping error:", e)

    return reviews


# ------------------------------------------------------
# 4) TMDB SIMILAR MOVIES RECOMMENDER
# ------------------------------------------------------
def get_similar_movies(tmdb_id, limit=10):
    """Return similar movies from TMDB."""
    url = f"{TMDB_BASE}/movie/{tmdb_id}/similar?api_key={TMDB_API_KEY}"
    data = requests.get(url).json()

    results = data.get("results", [])

    movies = []
    for m in results[:limit]:
        movies.append({
            "title": m.get("title"),
            "poster": (
                f"https://image.tmdb.org/t/p/w500{m['poster_path']}"
                if m.get("poster_path") else
                "https://via.placeholder.com/300x450?text=No+Image"
            ),
            "year": m.get("release_date", "")[:4],
        })

    return movies


# ------------------------------------------------------
# 5) TMDB RECOMMENDED MOVIES RECOMMENDER
# ------------------------------------------------------
def get_recommended_movies(tmdb_id, limit=10):
    """Return recommended movies from TMDB."""
    url = f"{TMDB_BASE}/movie/{tmdb_id}/recommendations?api_key={TMDB_API_KEY}"
    data = requests.get(url).json()

    results = data.get("results", [])

    movies = []
    for m in results[:limit]:
        movies.append({
            "title": m.get("title"),
            "poster": (
                f"https://image.tmdb.org/t/p/w500{m['poster_path']}"
                if m.get("poster_path") else
                "https://via.placeholder.com/300x450?text=No+Image"
            ),
            "year": m.get("release_date", "")[:4],
        })

    return movies


# ------------------------------------------------------
# 6) FINAL FUNCTION CALLED FROM STREAMLIT
# ------------------------------------------------------
def analyze_movie(selected_movie_dict, max_reviews=50):
    """
    selected_movie_dict = {title, year, tmdb_id, poster, genres, imdb_id}
    """

    tmdb_id = selected_movie_dict["tmdb_id"]

    # refresh details
    imdb_id, genres = get_movie_details(tmdb_id)

    movie_info = {
        "title": selected_movie_dict["title"],
        "year": selected_movie_dict["year"],
        "poster": selected_movie_dict["poster"],
        "genres": genres,
        "imdb_id": imdb_id
    }

    reviews = fetch_imdb_reviews(imdb_id, max_reviews)

    # NEW: recommended movies
    similar = get_similar_movies(tmdb_id)
    recommended = get_recommended_movies(tmdb_id)

    return movie_info, reviews, similar, recommended
