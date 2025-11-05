import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

sig_path = "data/compact_mc_jpsi_target_pythia8_acc.npy"
bkg_path = "data/comb_target_raw_oct24_acc.npy"

FEATURE_NAMES = [
    "dimu_y", "dimu_eta", "dimu_E", "dimu_pz",
    "dimu_mass", "dimu_pt", "dimu_phi",
    "theta_pos", "theta_neg", "opening_angle",
    "dpt", "dimu_mT", "Epos", "Eneg",
    "x_pos_st1", "x_neg_st1", "px_pos_st1", "px_neg_st1"
]
N_FEATS = len(FEATURE_NAMES)

MAX_EVENTS_PER_CLASS = 1_000_000   # cap per class (applied after loading)
BALANCE_CLASSES = True            
BATCH_SIZE = 64
LR = 1e-3                         
NUM_EPOCHS = 200
PATIENCE = 50
NORMALIZE = True                    
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
sig_data = np.load(sig_path, allow_pickle=True)
bkg_data = np.load(bkg_path, allow_pickle=True)

print(f"Raw Signal shape: {sig_data.shape}")
print(f"Raw Background shape: {bkg_data.shape}")

if sig_data.shape[1] < N_FEATS or bkg_data.shape[1] < N_FEATS:
    raise ValueError(f"Expected at least {N_FEATS} columns; "
                     f"got sig={sig_data.shape[1]}, bkg={bkg_data.shape[1]}.")

def cap_rows(arr, cap):
    if arr.shape[0] <= cap:
        return arr
    idx = np.random.choice(arr.shape[0], size=cap, replace=False)
    return arr[idx]

sig_data = cap_rows(sig_data, MAX_EVENTS_PER_CLASS)
bkg_data = cap_rows(bkg_data, MAX_EVENTS_PER_CLASS)

# NEW: Balance classes to prevent bias
if BALANCE_CLASSES:
    min_events = min(len(sig_data), len(bkg_data))
    sig_data = cap_rows(sig_data, min_events)
    bkg_data = cap_rows(bkg_data, min_events)
    print(f" Balanced classes to {min_events:,} events each")
else:
    print(f"classes unbalanced: {len(sig_data):,} signal vs {len(bkg_data):,} background")

# Keep first N_FEATS columns in given order
sig_X = sig_data[:, :N_FEATS]
bkg_X = bkg_data[:, :N_FEATS]

sig_y = np.ones((sig_X.shape[0], 1), dtype=np.float32)
bkg_y = np.zeros((bkg_X.shape[0], 1), dtype=np.float32)

# Combine, shuffle
X = np.vstack([sig_X, bkg_X]).astype(np.float32)
y = np.vstack([sig_y, bkg_y]).astype(np.float32)
X, y = shuffle(X, y, random_state=SEED)

print(f"Total dataset: {len(X):,} events ({(y==1).sum():.0f} signal, {(y==0).sum():.0f} background)")

# Normalize BEFORE splitting (but will refit on train only)

if NORMALIZE:
    scaler_check = StandardScaler()
    X_normalized = scaler_check.fit_transform(X)
    X = X_normalized  # Use normalized data
else:
    print("ormalization disabled - this may hurt performance!")

# Train / Val / Test split (80/10/10)

dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), 
                       torch.tensor(y, dtype=torch.float32))
n_total = len(dataset)
n_train = int(0.8 * n_total)
n_val   = int(0.1 * n_total)
n_test  = n_total - n_train - n_val

train_ds, val_ds, test_ds_raw = random_split(
    dataset, [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(SEED)
)

print(f" Training Set: {n_train:,} samples")
print(f" Validation Set: {n_val:,} samples")
print(f" Raw Testing Set: {n_test:,} samples")

# Dataloaders
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

def get_adjusted_test_loader(batch_size=64):
    X_test, y_test = zip(*test_ds_raw)
    X_test = torch.stack(X_test)
    y_test = torch.stack(y_test)

    sig_mask = (y_test.squeeze(-1) == 1)
    bkg_mask = (y_test.squeeze(-1) == 0)

    X_sig, Y_sig = X_test[sig_mask], y_test[sig_mask]
    X_bkg, Y_bkg = X_test[bkg_mask], y_test[bkg_mask]

    num_sig = len(X_sig) // 2
    X_sig   = X_sig[:num_sig]
    Y_sig   = Y_sig[:num_sig]

    X_new = torch.cat([X_sig, X_bkg], dim=0)
    Y_new = torch.cat([Y_sig, Y_bkg], dim=0)

    test_loader = DataLoader(TensorDataset(X_new, Y_new), batch_size=batch_size, shuffle=False)
    print(f" Adjusted test set: {len(X_sig):,} signals, {len(X_bkg):,} backgrounds.")
    return test_loader

test_loader = get_adjusted_test_loader(batch_size=BATCH_SIZE)

# =========================================================
# IMPROVED Model: ReLU + BatchNorm + Deeper
# =========================================================
class ParticleClassifier(nn.Module):
    def __init__(self, input_dim=N_FEATS, hidden_dims=(512, 256, 128), dropout=0.3):
        super().__init__()
        h1, h2, h3 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.BatchNorm1d(h1),
            nn.Dropout(dropout),
            
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.BatchNorm1d(h2),
            nn.Dropout(dropout),
            
            nn.Linear(h2, h3),
            nn.ReLU(),
            nn.BatchNorm1d(h3),
            nn.Dropout(dropout),
            
            nn.Linear(h3, 1)  # logits; will use BCEWithLogitsLoss
        )

    def forward(self, x):
        return self.net(x)

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total

model = ParticleClassifier(input_dim=N_FEATS, hidden_dims=(512, 256, 128), dropout=0.3).to(device)
print(f"🧮 Trainable parameters: {count_parameters(model):,}")

# Loss function
criterion = nn.BCEWithLogitsLoss()

# IMPROVED: AdamW optimizer with weight decay
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

# NEW: Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10, verbose=True
)

# =========================================================
# Training loop with early stopping (by val loss)
# =========================================================
best_val_loss = float('inf')
patience_ctr = 0
train_losses, val_losses = [], []
train_accs, val_accs = [], []

def batch_acc_from_logits(logits, targets, thresh=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > thresh).float()
    return (preds == targets).float().mean().item()

print("\n" + "="*60)
print("Starting Training")
print("="*60)

for epoch in range(1, NUM_EPOCHS + 1):
    # ---- train ----
    model.train()
    epoch_loss, epoch_acc, batches = 0.0, 0.0, 0
    for xb, yb in train_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc  += batch_acc_from_logits(logits, yb)
        batches    += 1

    train_loss = epoch_loss / batches
    train_acc  = 100.0 * (epoch_acc / batches)
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # ---- validate ----
    model.eval()
    val_loss_accum, val_acc_accum, v_batches = 0.0, 0.0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            val_loss_accum += loss.item()
            val_acc_accum  += batch_acc_from_logits(logits, yb)
            v_batches += 1

    val_loss = val_loss_accum / v_batches
    val_acc  = 100.0 * (val_acc_accum / v_batches)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    # Update learning rate
    scheduler.step(val_loss)

    print(f"Epoch [{epoch:3d}/{NUM_EPOCHS}] | "
          f"Train Loss={train_loss:.4f} Acc={train_acc:.2f}% | "
          f"Val Loss={val_loss:.4f} Acc={val_acc:.2f}%")

    # Early stopping + checkpoint
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_ctr = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
        }, "best_dimuon_model.pth")
        print(f"Saved best model (val loss={val_loss:.4f})")
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print(f"\n⏹ Early stopping at epoch {epoch}")
            break

# Load best model for evaluation
checkpoint = torch.load("best_dimuon_model.pth")
model.load_state_dict(checkpoint['model_state_dict'])
print(f"\n Loaded best model from epoch {checkpoint['epoch']} (val loss={checkpoint['val_loss']:.4f})")

# Plot Learning Curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(train_losses, label="Train Loss", linewidth=2)
ax1.plot(val_losses, label="Val Loss", linewidth=2)
ax1.set_xlabel("Epoch", fontsize=12)
ax1.set_ylabel("Loss", fontsize=12)
ax1.legend(fontsize=11)
ax1.set_title("Training & Validation Loss", fontsize=13)
ax1.grid(True, alpha=0.3)

ax2.plot(train_accs, label="Train Acc", linewidth=2)
ax2.plot(val_accs, label="Val Acc", linewidth=2)
ax2.set_xlabel("Epoch", fontsize=12)
ax2.set_ylabel("Accuracy (%)", fontsize=12)
ax2.legend(fontsize=11)
ax2.set_title("Training & Validation Accuracy", fontsize=13)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved training curves to training_curves.png")

# Comprehensive test evaluation with multiple thresholds

print("\n" + "="*60)
print("Test Set Evaluation")
print("="*60)

model.eval()
all_probs, all_labels = [], []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        logits = model(xb)
        probs = torch.sigmoid(logits)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(yb.cpu().numpy())

all_probs = np.concatenate(all_probs).flatten()
all_labels = np.concatenate(all_labels).flatten()

# Evaluate at multiple thresholds
print("\nPerformance at different thresholds:")
print("-" * 80)
print(f"{'Threshold':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Purity':<12}")
print("-" * 80)

for thresh in [0.3, 0.5, 0.7, 0.8, 0.9]:
    preds = (all_probs > thresh).astype(float)
    
    tp = ((preds == 1) & (all_labels == 1)).sum()
    fp = ((preds == 1) & (all_labels == 0)).sum()
    tn = ((preds == 0) & (all_labels == 0)).sum()
    fn = ((preds == 0) & (all_labels == 1)).sum()
    
    accuracy = (tp + tn) / len(all_labels) * 100
    precision = tp / (tp + fp + 1e-8) * 100
    recall = tp / (tp + fn + 1e-8) * 100
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    purity = precision  # Purity = Precision for signal
    
    print(f"{thresh:<12.1f} {accuracy:<12.2f} {precision:<12.2f} {recall:<12.2f} {f1:<12.2f} {purity:<12.2f}")
