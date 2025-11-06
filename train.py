import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import optuna
from optuna.samplers import TPESampler
from optuna.trial import TrialState
from optuna.pruners import HyperbandPruner
from optuna.logging import get_logger
from optuna.visualization import plot_optimization_history, plot_param_importances

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
NUM_EPOCHS = 50
PATIENCE = 50
NORMALIZE = True                    
SEED = 42
L1_REG = 1e-5  # L1 regularization coefficient
L2_REG = 1e-4  # L2 regularization coefficient (in addition to weight_decay)

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

class ResidualConnection(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(hidden_dim)  
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        if input_dim != hidden_dim:
            self.proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.proj = None

    def forward(self, x):
        residual = x
        if self.proj is not None:
            residual = self.proj(x)
        
        out = self.relu(self.bn1(self.fc1(x)))
        out = self.fc2(out)
        out = self.relu(self.bn2(out) + residual)
        return out



class ParticleClassifier(nn.Module):
    def __init__(self, input_dim=N_FEATS, hidden_dims=(512, 256, 128), dropout=0.3, n_layers=None):
        super().__init__()
        if n_layers is None:
            n_layers = len(hidden_dims)
        
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(ResidualConnection(input_dim if i == 0 else hidden_dims[i-1], hidden_dims[i]))
        
        self.output = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output(x)

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total

def compute_regularization_loss(model, l1_reg=0.0, l2_reg=0.0):

    l1_loss = 0.0
    l2_loss = 0.0
    
    for param in model.parameters():
        if param.requires_grad:
            if l1_reg > 0.0:
                l1_loss += torch.sum(torch.abs(param))
            if l2_reg > 0.0:
                l2_loss += torch.sum(param ** 2)
    
    return l1_reg * l1_loss + l2_reg * l2_loss

def batch_acc_from_logits(logits, targets, thresh=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > thresh).float()
    return (preds == targets).float().mean().item()

def objective(trial):

    n_layers = trial.suggest_int("n_layers", 2, 8)
    hidden_dims = tuple(trial.suggest_int(f"hidden_{i}", 8, 1024, step=64) for i in range(n_layers))
    
    params = {
        'n_layers': n_layers,
        'hidden_dims': hidden_dims,
        'lr': trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        'weight_decay': trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        'batch_size': trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        'l1_reg': trial.suggest_float("l1_reg", 0.0, 0.1),
        'l2_reg': trial.suggest_float("l2_reg", 0.0, 0.1),
        'beta1': trial.suggest_float("beta1", 0.8, 0.95),
        'beta2': trial.suggest_float("beta2", 0.95, 0.999),
        'epsilon': trial.suggest_float("epsilon", 1e-9, 1e-6, log=True),
        'patience': trial.suggest_int("patience", 5, 20),
    }
    
    print(f"Training with params: {params}")

    model = ParticleClassifier(input_dim=N_FEATS, hidden_dims=params['hidden_dims'], n_layers=params['n_layers']).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(model.parameters(), 
                            lr=params['lr'], 
                            weight_decay=params['weight_decay'],
                            betas=(params['beta1'], params['beta2']),
                            eps=params['epsilon'])

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=params['patience'], verbose=False
    )
    
    train_loader_trial = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True)
    val_loader_trial = DataLoader(val_ds, batch_size=params['batch_size'], shuffle=False)
    

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(1, 101):
        model.train()
        epoch_loss, epoch_acc, batches = 0.0, 0.0, 0
        for xb, yb in train_loader_trial:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            reg_loss = compute_regularization_loss(model, params['l1_reg'], params['l2_reg'])
            loss = loss + reg_loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc  += batch_acc_from_logits(logits, yb)
            batches    += 1

        train_loss = epoch_loss / batches
        train_acc  = 100.0 * (epoch_acc / batches)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        model.eval()
        val_loss_accum, val_acc_accum, v_batches = 0.0, 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader_trial:
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

        scheduler.step(val_loss)

        trial.report(val_loss, epoch)
        
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        print(f"Epoch [{epoch:3d}/100] | "
              f"Train Loss={train_loss:.4f} Acc={train_acc:.2f}% | "
              f"Val Loss={val_loss:.4f} Acc={val_acc:.2f}%")

    return val_losses[-1]

study = optuna.create_study(
    study_name="dimuon_classifier",
    direction="minimize",
    sampler=TPESampler(seed=SEED),
    pruner=HyperbandPruner(
        min_resource=1,
        max_resource=100,
        reduction_factor=3,
        min_early_stopping_rate=0
    ),
    storage="sqlite:///dimuon_classifier.db",
    load_if_exists=True
)

# Run the optimization
print("\n" + "="*60)
print("Starting Optuna Hyperparameter Optimization")
print("="*60)
study.optimize(objective, n_trials=50, timeout=None)

print("\n" + "="*60)
print("Optimization finished!")
print("="*60)
print(f"Number of finished trials: {len(study.trials)}")
print(f"Number of pruned trials: {len([t for t in study.trials if t.state == TrialState.PRUNED])}")
print(f"Number of complete trials: {len([t for t in study.trials if t.state == TrialState.COMPLETE])}")

print("\nBest trial:")
trial = study.best_trial
print(f"  Value: {trial.value:.4f}")
print(f"  Params:")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Get best model parameters
best_params = study.best_params
print(f"\nBest parameters: {best_params}")

# Reconstruct hidden_dims tuple from individual parameters
best_n_layers = best_params.get('n_layers', 3)
best_hidden_dims = tuple(best_params.get(f'hidden_{i}', 128) for i in range(best_n_layers))
if not best_hidden_dims:
    best_hidden_dims = (512, 256, 128)
    best_n_layers = 3

model = ParticleClassifier(input_dim=N_FEATS, hidden_dims=best_hidden_dims, n_layers=best_n_layers).to(device)

print(f"🧮 Trainable parameters: {count_parameters(model):,}")

criterion = nn.BCEWithLogitsLoss()

optimizer = optim.AdamW(model.parameters(), 
                        lr=best_params['lr'], 
                        weight_decay=best_params['weight_decay'],
                        betas=(best_params['beta1'], best_params['beta2']),
                        eps=best_params['epsilon'])

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
        # Add L1 and L2 regularization
        reg_loss = compute_regularization_loss(model, L1_REG, L2_REG)
        loss = loss + reg_loss
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
