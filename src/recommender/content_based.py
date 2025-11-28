# src/recommender/content_based.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def build_overview_tfidf(movies_df, max_features=10000):
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1,2))
    overview_matrix = tfidf.fit_transform(movies_df['overview'].fillna(''))
    return overview_matrix, tfidf

def recommend_by_title(movies_df, overview_matrix, title, top_k=10):
    if title not in movies_df['title'].values:
        return pd.DataFrame()
    idx = movies_df.index[movies_df['title'] == title][0]
    sims = cosine_similarity(overview_matrix[idx], overview_matrix).flatten()
    top_idx = sims.argsort()[::-1][1:top_k+1]
    return movies_df.iloc[top_idx].copy()

if __name__ == '__main__':
    print('Run recommend_by_title by importing this module in your script/notebook.')
