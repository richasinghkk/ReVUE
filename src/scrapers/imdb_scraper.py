import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/119.0 Safari/537.36"
}

# ----------------------------------------
# 1) USE IMDb SUGGESTION API (NEVER BREAKS)
# ----------------------------------------
def search_movie(movie_name):
    """
    Uses the IMDb internal autocomplete API.
    This is extremely stable and returns correct IMDb IDs reliably.
    """
    query = movie_name.strip().replace(" ", "%20")
    first_letter = movie_name[0].lower()

    url = f"https://v2.sg.media-imdb.com/suggestion/{first_letter}/{query}.json"

    try:
        r = requests.get(url, timeout=8)
        data = r.json()

        for item in data.get("d", []):
            if "id" in item and item["id"].startswith("tt"):
                return {
                    "imdb_id": item["id"],
                    "title": item.get("l"),
                    "year": item.get("y"),
                    "poster": item["i"][0] if item.get("i") else None
                }
    except:
        return None

    return None


# ----------------------------------------
# 2) GET GENRES
# ----------------------------------------
def get_genres(imdb_id):
    url = f"https://www.imdb.com/title/{imdb_id}/"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        genres = [g.get_text(strip=True) for g in soup.select("a[href*='/search/title?genres=']")]
        return genres[:5]
    except:
        return []


# ----------------------------------------
# 3) SCRAPE REVIEWS
# ----------------------------------------
def fetch_reviews(imdb_id, max_reviews=50):
    base = f"https://www.imdb.com/title/{imdb_id}/reviews"
    reviews = []

    try:
        r = requests.get(base, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")

        review_blocks = soup.select(".text.show-more__control")

        for block in review_blocks:
            text = block.get_text(" ", strip=True)
            if text:
                reviews.append(text)
            if len(reviews) >= max_reviews:
                break

    except:
        pass

    return reviews


# ----------------------------------------
# 4) MAIN FUNCTION → RETURN ONLY (movie, reviews)
# ----------------------------------------
def analyze_movie(movie_name, max_reviews=50):
    movie = search_movie(movie_name)
    if not movie:
        return None, []

    # Add genres
    movie["genres"] = get_genres(movie["imdb_id"])

    # Fetch reviews
    reviews = fetch_reviews(movie["imdb_id"], max_reviews)

    return movie, reviews   # EXACTLY TWO VALUES
