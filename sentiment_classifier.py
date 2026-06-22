# =============================================================
# Project 2: Sentiment Classifier — Fine-tuning DistilBERT
# Preethi Amara — Tesla ML Intern Prep
# Dataset: IMDB movie reviews (50,000 reviews, pos/neg labels)
# =============================================================
# What this teaches:
#   - How to use a pretrained transformer (DistilBERT)
#   - Tokenization: converting text → numbers a model understands
#   - Fine-tuning: adapting a large model to your specific task
#   - HuggingFace: the most used ML library in industry
#
# The core skill: "take a giant pretrained model, adapt it to
# a new task in hours instead of months" — exactly what Tesla
# does with their foundation models.
# =============================================================

# --- STEP 0: Install dependencies (run once) -----------------
# pip3 install torch transformers datasets matplotlib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    DistilBertTokenizerFast,   # converts text → token IDs
    DistilBertForSequenceClassification,  # DistilBERT + classifier head
    get_scheduler              # learning rate scheduler
)
from datasets import load_dataset  # HuggingFace datasets library
import matplotlib.pyplot as plt
import numpy as np
from torch.optim import AdamW

# =============================================================
# STEP 1: Config
# =============================================================
MAX_LENGTH    = 256    # max tokens per review (truncate longer ones)
BATCH_SIZE    = 16     # smaller batch since transformers use more memory
EPOCHS        = 3      # transformers fine-tune fast — 3 epochs is enough
LEARNING_RATE = 2e-5   # very small LR — pretrained weights are fragile

# M1 Mac GPU, CUDA, or CPU
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")


# =============================================================
# STEP 2: Load the IMDB dataset
# 25,000 training reviews + 25,000 test reviews
# Each review is labeled: 0 = negative, 1 = positive
# =============================================================
print("\nLoading IMDB dataset...")
dataset = load_dataset("stanfordnlp/imdb")

# Use a subset to keep training fast on your laptop
# (still 10K examples — more than enough to learn from)
train_data = dataset["train"].shuffle(seed=42).select(range(10000))
test_data  = dataset["test"].shuffle(seed=42).select(range(2000))

print(f"Training reviews : {len(train_data)}")
print(f"Test reviews     : {len(test_data)}")
print(f"\nExample review:\n{train_data[0]['text'][:300]}...")
print(f"Label: {'POSITIVE' if train_data[0]['label'] == 1 else 'NEGATIVE'}")


# =============================================================
# STEP 3: Tokenization
#
# Models don't understand raw text — they need numbers.
# A tokenizer splits text into "tokens" (words/subwords)
# and maps each to a number ID.
#
# Example:
#   "I loved this film" →
#   tokens:   ["i", "loved", "this", "film"]
#   input_ids: [1045, 3100, 2023, 2143]
#
# DistilBertTokenizerFast handles this automatically.
# =============================================================
print("\nLoading tokenizer...")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",    # pad short reviews to MAX_LENGTH
        truncation=True,         # cut long reviews at MAX_LENGTH
        max_length=MAX_LENGTH
    )

print("Tokenizing dataset (this takes a minute)...")
train_data = train_data.map(tokenize, batched=True)
test_data  = test_data.map(tokenize, batched=True)

# Set format so PyTorch can read it directly
train_data.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_data.set_format("torch",  columns=["input_ids", "attention_mask", "label"])

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False)

print(f"Tokenization complete.")


# =============================================================
# STEP 4: Load pretrained DistilBERT
#
# DistilBERT is a smaller, faster version of BERT.
# It was trained on Wikipedia + BookCorpus (billions of words).
# It already understands language deeply.
#
# DistilBertForSequenceClassification adds a small classifier
# layer on top of DistilBERT for our pos/neg task.
#
# num_labels=2 means: 2 output classes (positive, negative)
# =============================================================
print("\nLoading pretrained DistilBERT...")
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)
model = model.to(DEVICE)

total     = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters     : {total:,}")
print(f"Trainable parameters : {trainable:,}")


# =============================================================
# STEP 5: Optimizer and scheduler
#
# AdamW: like Adam but with better weight decay handling
# (standard choice for fine-tuning transformers)
#
# Linear scheduler: warms up LR then linearly decays it
# This is the standard recipe for fine-tuning BERT-style models
# =============================================================
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

total_steps = len(train_loader) * EPOCHS
scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=total_steps // 10,  # warm up for first 10% of training
    num_training_steps=total_steps
)


# =============================================================
# STEP 6: Training loop
# =============================================================
def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for batch_idx, batch in enumerate(loader):
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["label"].to(DEVICE)

        optimizer.zero_grad()

        # HuggingFace models return an object, not just logits
        # outputs.loss = cross entropy loss (computed internally)
        # outputs.logits = raw scores for each class
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()

        # Gradient clipping: prevents exploding gradients
        # (important for transformer fine-tuning)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        predicted   = outputs.logits.argmax(dim=1)
        correct    += (predicted == labels).sum().item()
        total      += labels.size(0)

        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}")

    return total_loss / len(loader), 100.0 * correct / total


def evaluate(model, loader):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["label"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            total_loss += outputs.loss.item()
            predicted   = outputs.logits.argmax(dim=1)
            correct    += (predicted == labels).sum().item()
            total      += labels.size(0)

    return total_loss / len(loader), 100.0 * correct / total


# =============================================================
# STEP 7: Run training
# =============================================================
train_losses, test_losses = [], []
train_accs,   test_accs   = [], []
best_acc = 0.0

print("\n--- Training started ---\n")
for epoch in range(1, EPOCHS + 1):
    print(f"Epoch {epoch}/{EPOCHS}")
    tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, scheduler)
    te_loss, te_acc = evaluate(model, test_loader)

    train_losses.append(tr_loss)
    test_losses.append(te_loss)
    train_accs.append(tr_acc)
    test_accs.append(te_acc)

    print(f"  Train loss: {tr_loss:.4f}  |  Train accuracy: {tr_acc:.2f}%")
    print(f"  Test  loss: {te_loss:.4f}  |  Test  accuracy: {te_acc:.2f}%\n")

    if te_acc > best_acc:
        best_acc = te_acc
        model.save_pretrained("sentiment_model_best")
        tokenizer.save_pretrained("sentiment_model_best")
        print(f"  ✓ New best model saved ({best_acc:.2f}%)\n")

print(f"--- Training complete | Best test accuracy: {best_acc:.2f}% ---")


# =============================================================
# STEP 8: Plot training curves
# =============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.plot(train_losses, label='Train loss', marker='o')
ax1.plot(test_losses,  label='Test loss',  marker='o')
ax1.set_title("Loss over epochs")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend()

ax2.plot(train_accs, label='Train accuracy', marker='o')
ax2.plot(test_accs,  label='Test accuracy',  marker='o')
ax2.set_title("Accuracy over epochs")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)"); ax2.legend()

plt.suptitle(f"Sentiment Classifier (DistilBERT) — Best: {best_acc:.2f}%")
plt.tight_layout()
plt.savefig("sentiment_training_curves.png", dpi=150)
print("Saved sentiment_training_curves.png")


# =============================================================
# STEP 9: Try it on your own sentences
# This is the fun part — see your model in action!
# =============================================================
def predict(texts):
    """Run the model on a list of sentences and print results."""
    model.eval()
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        probs   = torch.softmax(outputs.logits, dim=1)
        preds   = outputs.logits.argmax(dim=1)

    labels_map = {0: "NEGATIVE", 1: "POSITIVE"}
    print("\n--- Live predictions ---")
    for text, pred, prob in zip(texts, preds, probs):
        label      = labels_map[pred.item()]
        confidence = prob[pred].item() * 100
        print(f"  '{text[:80]}'")
        print(f"  → {label} ({confidence:.1f}% confidence)\n")

# Test on some example sentences
predict([
    "This was absolutely one of the best films I have ever seen. Incredible!",
    "Terrible movie. Boring plot, bad acting, complete waste of time.",
    "It was okay I guess, nothing special but not awful either.",
    "The cinematography was stunning but the story left a lot to be desired.",
])

print("\nAll done! Files saved:")
print("  sentiment_training_curves.png — loss and accuracy over training")
print("  sentiment_model_best/         — saved model weights")
