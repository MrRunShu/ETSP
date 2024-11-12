import torch
from model import load_model_and_tokenizer, get_training_args
from dataset import create_dataset, load_data
from explain import explain_with_shap
from transformers import Trainer

# Load your data
train_texts, train_labels = load_data("datasets/train.csv")
test_texts, test_labels = load_data("datasets/test.csv")

# Load model and tokenizer
model, tokenizer = load_model_and_tokenizer()

# Create datasets
train_dataset = create_dataset(train_texts, train_labels, tokenizer)
test_dataset = create_dataset(test_texts, test_labels, tokenizer)

# Display a few samples from the processed train dataset
for i in range(5):  # Display the first 5 samples
    sample = train_dataset[i]
    print(f"Text: {train_dataset.texts[i]}")
    print(f"Label: {train_dataset.labels[i]}")

# Get training arguments
training_args = get_training_args()

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
trainer.train()

# Evaluate the model on the test dataset
print("Evaluating the model on the test dataset...")
test_results = trainer.evaluate(test_dataset)
print("Test Results:", test_results)


# Use SHAP for explainability
sample_text = "no one ever predicted this was going to happen"
explain_with_shap(model, tokenizer, sample_text)
