# src/recommender/collaborative.py
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import pickle, os

def train_svd(ratings_csv='data/raw/movielens/ratings.csv', save_path='saved_models/svd_model.pkl'):
    df = pd.read_csv(ratings_csv)
    reader = Reader(rating_scale=(df['rating'].min(), df['rating'].max()))
    data = Dataset.load_from_df(df[['userId','movieId','rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2)
    algo = SVD(n_factors=50, n_epochs=20, verbose=True)
    algo.fit(trainset)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(algo, f)
    print('Saved SVD model to', save_path)

if __name__ == '__main__':
    train_svd()
