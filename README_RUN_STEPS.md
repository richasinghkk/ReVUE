# Quick run & training steps (after extracting zip)

1. Open the project folder in VS Code.
2. Create and activate a virtual environment:
   - Windows: `python -m venv venv` then `venv\\Scripts\\activate`
   - Mac/Linux: `python3 -m venv venv` then `source venv/bin/activate`

3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

4. Place your datasets:
   - Put sentiment reviews CSV at `data/raw/imdb_reviews.csv` (columns: review, sentiment)
   - Put MovieLens files under `data/raw/movielens/` and processed movie metadata at `data/processed/movies_final.csv` (columns: movieId,title,overview,genres,mean_sentiment)

5. Train TF-IDF sentiment classifier:
   ```bash
   python src/models/train_tfidf_model.py
   ```
   This will create `saved_models/tfidf_model.pkl`

6. (Optional) Fine-tune BERT (requires GPU & time):
   ```bash
   python src/models/train_bert_model.py
   ```

7. (Optional) Train collaborative SVD model (requires MovieLens ratings):
   ```bash
   python src/recommender/collaborative.py
   ```

8. Run the Streamlit app:
   ```bash
   streamlit run src/app/streamlit_app.py
   ```
   Open the shown URL in your browser (usually http://localhost:8501).

Notes:
- The repository scaffold does NOT include datasets. Add your CSV files to the `data/raw/` and processed CSVs to `data/processed/` before training or running the app.