# src/preprocessing/text_cleaner.py
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download resources on first run (comment out after installed)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet'); nltk.download('omw-1.4')

STOPWORDS = set(stopwords.words('english'))
LEM = WordNetLemmatizer()

def clean_text(text: str, remove_stopwords=True, lemmatize=True) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\\S+|www\\.\\S+', ' ', text)
    text = re.sub(r"[^a-z0-9\\s']", ' ', text)
    text = re.sub(r'\\s+', ' ', text).strip()
    tokens = text.split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    if lemmatize:
        tokens = [LEM.lemmatize(t) for t in tokens]
    return ' '.join(tokens)
