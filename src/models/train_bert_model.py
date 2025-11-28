# src/models/train_bert_model.py (outline for fine-tuning using Hugging Face)
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import pandas as pd

MODEL_NAME = 'distilbert-base-uncased'

def load_dataset(csv_path='data/raw/imdb_reviews.csv'):
    df = pd.read_csv(csv_path)
    if 'review' not in df.columns and 'text' in df.columns:
        df.rename(columns={'text':'review'}, inplace=True)
    df = df[['review', 'sentiment']]
    df['label'] = df['sentiment'].map(lambda x: 1 if str(x).lower().startswith('pos') else 0)
    ds = Dataset.from_pandas(df[['review','label']])
    return ds

if __name__ == '__main__':
    ds = load_dataset()
    ds = ds.train_test_split(test_size=0.1)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    def tokenize(ex):
        return tokenizer(ex['review'], padding='max_length', truncation=True, max_length=256)
    ds = ds.map(tokenize, batched=True)
    ds.set_format(type='torch', columns=['input_ids','attention_mask','label'])
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    training_args = TrainingArguments(output_dir='saved_models/bert_model', num_train_epochs=2, per_device_train_batch_size=8, evaluation_strategy='epoch', save_strategy='epoch', load_best_model_at_end=True)
    trainer = Trainer(model=model, args=training_args, train_dataset=ds['train'], eval_dataset=ds['test'])
    trainer.train()
    trainer.save_model('saved_models/bert_model')
