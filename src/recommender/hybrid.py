# src/recommender/hybrid.py
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def hybrid_score(user_id, candidate_movie_ids, movies_df, content_matrix, svd_algo=None, user_profiles=None, w_content=0.5, w_collab=0.4, w_sent=0.1):
    scores = {}
    user_pref_indices = user_profiles.get(user_id, {}).get('liked_movie_indices', [])
    if len(user_pref_indices) == 0:
        content_scores = np.zeros(len(candidate_movie_ids))
    else:
        user_vec = np.mean(content_matrix[user_pref_indices], axis=0)
        sims = cosine_similarity(user_vec, content_matrix[candidate_movie_ids]).flatten()
        content_scores = sims
    collab_scores = np.zeros(len(candidate_movie_ids))
    if svd_algo is not None:
        for i, mid in enumerate(candidate_movie_ids):
            est = svd_algo.predict(user_id, mid).est
            collab_scores[i] = est
    sentiment_scores = np.array([movies_df.set_index('movieId').loc[mid]['mean_sentiment'] if mid in set(movies_df['movieId']) else 0.5 for mid in candidate_movie_ids])
    user_sent_pref = user_profiles.get(user_id, {}).get('sentiment_mean', 0.5)
    sent_match = 1 - np.abs(sentiment_scores - user_sent_pref)
    final = w_content * content_scores + w_collab * collab_scores + w_sent * sent_match
    return final
