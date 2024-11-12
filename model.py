import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

def load_model_and_tokenizer(model_name="roberta-base", num_labels=2):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return model, tokenizer

def get_training_args():
    return TrainingArguments(
        output_dir='./results',
        evaluation_strategy="epoch",  # Keep this as "epoch" for evaluation
        save_strategy="epoch",        # Keep this as "epoch" for saving checkpoints
        logging_strategy="epoch",     # Change logging to "epoch"
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=10,
        weight_decay=0.01,
        logging_dir='./logs',
        save_total_limit=2,           # Limit the number of saved checkpoints
        load_best_model_at_end=True,  # Load the best model at the end of training
        metric_for_best_model="eval_loss",  # Use evaluation loss to select the best model
    )

