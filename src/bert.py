import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- training function for one fold ---
def train_bert_fold(train_texts, train_labels, val_texts, val_labels,
                    tokenizer, epochs=3, batch_size=32, lr=2e-5):

    train_dataset = TweetDataset(train_texts, train_labels, tokenizer)
    val_dataset   = TweetDataset(val_texts,   val_labels,   tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False)

    # load fresh model for each fold
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels  = 2,
        hidden_dropout_prob = 0.1
    ).to(device)

    # class weights for imbalance
    n_neg = sum(1 for l in train_labels if l == 0)
    n_pos = sum(1 for l in train_labels if l == 1)
    weight = torch.tensor([1.0, n_neg/n_pos], dtype=torch.float).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = total_steps // 10,
        num_training_steps = total_steps
    )

    # --- training loop ---
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"  Epoch {epoch+1}/{epochs}",
                          leave=False):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_batch   = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids      = input_ids,
                            attention_mask = attention_mask,
                            labels         = labels_batch)

            # apply class weights manually
            loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
            loss    = loss_fn(outputs.logits, labels_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        print(f"  Epoch {epoch+1} loss: {total_loss/len(train_loader):.4f}")

    # --- evaluation ---
    model.eval()
    all_preds  = []
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_batch   = batch['label'].to(device)

            outputs = model(input_ids      = input_ids,
                            attention_mask = attention_mask)
            probs   = torch.softmax(outputs.logits, dim=1)[:, 1]
            preds   = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())

    return np.array(all_preds), np.array(all_probs), np.array(all_labels)