# src/models/train_tfidf_model.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
from src.preprocessing.text_cleaner import clean_text

def load_imdb_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'review' not in df.columns and 'text' in df.columns:
        df.rename(columns={'text':'review'}, inplace=True)
    return df

def train(csv_path: str = 'data/raw/imdb_reviews.csv', save_path: str = 'saved_models/tfidf_model.pkl'):
    print('Loading dataset...', csv_path)
    df = load_imdb_csv(csv_path)
    if 'sentiment' not in df.columns and 'label' in df.columns:
        df.rename(columns={'label':'sentiment'}, inplace=True)

    # Map common string labels to 0/1 if necessary
    if df['sentiment'].dtype == object:
        df['sentiment'] = df['sentiment'].map(lambda x: 1 if str(x).lower().startswith('pos') else 0)

    df['clean_review'] = df['review'].astype(str).apply(clean_text)
    X = df['clean_review']
    y = df['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    vect = TfidfVectorizer(max_features=50000, ngram_range=(1,2))
    X_train_tf = vect.fit_transform(X_train)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_tf, y_train)

    X_test_tf = vect.transform(X_test)
    preds = clf.predict(X_test_tf)
    print(classification_report(y_test, preds))
    print('Accuracy:', accuracy_score(y_test, preds))

    pipeline = {'vectorizer': vect, 'model': clf}
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(pipeline, save_path)
    print('Saved model pipeline to', save_path)

if __name__ == '__main__':
    train()
