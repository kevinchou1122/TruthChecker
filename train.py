import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tqdm import tqdm


csv_file_path = '/kaggle/input/fake-news-detection-dataset/fake_news_dataset.csv'  

label_column_name = 'label'
text_column_name = 'text'
title_column_name = 'title'
value_for_false = "fake"
value_for_real = "real"

try:
    df = pd.read_csv(csv_file_path)
    print(f"Successfully loaded '{csv_file_path}'. Shape: {df.shape}")
    print("\nFirst 5 rows of the DataFrame:")
    print(df.head())
    print(f"\nColumns in the DataFrame: {df.columns.tolist()}")

    # Check if essential columns exist
    if label_column_name not in df.columns:
        raise ValueError(f"Label column '{label_column_name}' not found in the CSV. Available columns: {df.columns.tolist()}")
    if text_column_name not in df.columns:
        raise ValueError(f"Text column '{text_column_name}' not found in the CSV. Available columns: {df.columns.tolist()}")
    if title_column_name and title_column_name not in df.columns:
        print(f"Warning: Title column '{title_column_name}' not found. Will proceed without titles.")
        title_column_name = None # Ensure it's None if not found

except FileNotFoundError:
    print(f"Error: The file '{csv_file_path}' was not found.")
    exit()
except pd.errors.EmptyDataError:
    print(f"Error: The file '{csv_file_path}' is empty.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the CSV: {e}")
    exit()


print(f"\nUnique values in the '{label_column_name}' column: {df[label_column_name].unique()}")
print(f"Data type of '{label_column_name}' column: {df[label_column_name].dtype}")

try:
    false_articles_mask = (df[label_column_name] == value_for_false)
    real_articles_mask = (df[label_column_name] == value_for_real)
except TypeError as e:
    print(f"\nTypeError during mask creation. This might happen if your label column's data type "
          f"(e.g., int) doesn't match the type of `value_for_false` or `value_for_real` (e.g., string).")
    print(f"Label column dtype: {df[label_column_name].dtype}, value_for_false type: {type(value_for_false)}")
    print(f"Error: {e}")
    exit()


# Filter the DataFrame
df_false_articles = df[false_articles_mask]
df_real_articles = df[real_articles_mask]

print(f"\nNumber of false articles: {len(df_false_articles)}")
print(f"Number of real articles: {len(df_real_articles)}")

if not df_false_articles.empty:
    print("\n--- Example of False Articles ---")
    columns_to_show = [col for col in [title_column_name, text_column_name, label_column_name] if col]
    print(df_false_articles[columns_to_show].head(3)) # Show top 3
else:
    print("\nNo false articles found with the specified criteria.")

if not df_real_articles.empty:
    print("\n--- Example of Real Articles ---")
    columns_to_show = [col for col in [title_column_name, text_column_name, label_column_name] if col]
    print(df_real_articles[columns_to_show].head(3))   # Show top 3
else:
    print("\nNo real articles found with the specified criteria.")

# Check for overlaps or missing data
total_separated = len(df_false_articles) + len(df_real_articles)
if total_separated != len(df):
    print(f"\nWarning: Original DataFrame had {len(df)} rows, but separated articles total {total_separated}.")
    print("This could be due to other values in the label column or issues with the label values specified.")
    unclassified_mask = ~(false_articles_mask | real_articles_mask)
    df_unclassified = df[unclassified_mask]
    if not df_unclassified.empty:
        print(f"Found {len(df_unclassified)} unclassified articles.")
        print("Unique labels in unclassified articles:", df_unclassified[label_column_name].unique())

df_false_articles.to_csv('/kaggle/working/false_articles_separated.csv', index=False)
df_real_articles.to_csv('/kaggle/working/real_articles_separated.csv', index=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df_fake1 = pd.read_csv("/kaggle/input/fake-news-detection-datasets/News _dataset/Fake.csv")
df_real1 = pd.read_csv("/kaggle/input/real-news-new/scraped_articles.csv")
df_fake2 = pd.read_csv("//kaggle/working/false_articles_separated.csv")
df_real2 = pd.read_csv("/kaggle/working/real_articles_separated.csv")

df_fake1["label"]=1
df_real1["label"]=0
df_fake2["label"]=1
df_real2["label"]=0

df = pd.concat([df_fake1, df_fake2, df_real1, df_real2], ignore_index=True)

if "title" in df.columns and "text" in df.columns:
    df["text"] = df["title"] + " " + df["text"]

df = df[["text", "label"]]

# Shuffle the data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV
df.to_csv("/kaggle/working/fake_news_combined.csv", index=False)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"].tolist(), df["label"].tolist(), test_size=0.2, random_state=42
)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})

def tokenize(batch):
    return tokenizer(batch["text"], padding =  "max_length", truncation = True, max_length = 512)

train_dataset= train_dataset.map(tokenize, batched = True)
val_dataset= val_dataset.map(tokenize, batched = True)

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

from transformers import BertForSequenceClassification, Trainer, TrainingArguments, get_linear_schedule_with_warmup
from torch.optim import AdamW
import torch
from torch.utils.data import DataLoader
import os
from datasets import Dataset


print("GPU Available:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")


# Create data collator to handle the label vs labels mismatch
def data_collator(features):
    batch = {}

    keys = features[0].keys()
    for key in keys:
        if key != 'label':
            batch[key] = torch.stack([f[key] for f in features])
    if 'label' in keys:
        batch['labels'] = torch.stack([f['label'] for f in features])

    return batch


# Load model and move to GPU
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)

training_args = TrainingArguments(
    output_dir="/kaggle/working/results",
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_dir="/kaggle/working/logs",
    logging_steps=100,
    save_steps=500,
    save_total_limit=2,
    report_to="none"
)

# Set up trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator
)

# Print training info
print("Starting training...")
# Start training
trainer.train()
print("Training done!")

# Save the model
model_save_path = "/kaggle/working/final_model"
model.save_pretrained(model_save_path)
print(f"Model saved to {model_save_path}")

# Evaluate on validation set
print("Evaluating model...")
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

predictions = trainer.predict(validation_dataset)
preds = np.argmax(predictions.predictions, axis=-1)
labels = predictions.label_ids

accuracy = accuracy_score(labels, preds)
precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

predictions = trainer.predict(val_dataset)
preds = np.argmax(predictions.predictions, axis=-1)
labels = predictions.label_ids

accuracy = accuracy_score(labels, preds)
precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

