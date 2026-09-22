


#%%
#%%

## Without GDD, HW. Additional run without LAI, with SPEI_6

import os
import itertools
import random
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import shap
import optuna 

warnings.filterwarnings('ignore')

# ================================================================================
# Helper function for formatted logging with timestamps — for Spyder console tracking
# ================================================================================
def log(msg, level=0):
    """
    Formatted print with timestamp and indentation level.
    level=0 → Main header
    level=1 → Step details
    level=2 → Sub-details (each feature, epoch, etc.)
    """
    ts     = pd.Timestamp.now().strftime("%H:%M:%S")
    indent = "   " * level
    print(f"[{ts}] {indent}{msg}")

def log_separator(char="=", width=75):
    print(char * width)

def log_section(title):
    log_separator()
    log(f"  {title}")
    log_separator()


# ================================================================================
# Step 0 — Random Seed Fixing and Device Detection
# ================================================================================
log_section("Step 0 | Environment Settings and Seed Initialization")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log(f"Detected Device: {device}")
if torch.cuda.is_available():
    log(f"GPU: {torch.cuda.get_device_name(0)}", level=1)
log(f"Fixed Seed: {SEED} (Python, NumPy, PyTorch, CUDA)", level=1)


# ================================================================================
# Step 1 — Defining Paths, Variables, and Global Parameters
# ================================================================================
log_section("Step 1 | Paths and Global Parameter Definitions")

INPUT_DATA_PATH    = r"D:\Stav\research\Israel_polygons\final_excels\data_for_LSTM_final_with_SPEI_6_no_LAI_no_GDD_HW.csv"
STATIC_BORUTA_PATH = r"D:\Stav\research\Israel_polygons\random forest\boruta_confirmed_static_features_no_LAI.xlsx"
BASE_OUTPUT_DIR    = r"D:\Stav\research\Israel_polygons\LSTM"
DATE_SUFFIX        = "11.6.26_also_static_first_layer_and_prespton"

DYNAMIC_FEATURES_BASE = [
    'T_max', 'T_min', 'T_mean', 'daily_rain_mm', 'VPD kPa_mean', 
    'VPD kPa_min', 'VPD kPa_max', 'ETo_mm_day', 'SPEI_6' 
]

TARGET_COL              = 'GPP'
WINDOWS                 = [7, 14, 30, 60, 90, 120, 180, 270]
MAX_TUNING_COMBINATIONS = 35

log(f"Input File:        {INPUT_DATA_PATH}")
log(f"Target Column:     {TARGET_COL}")


# ================================================================================
# Step 2 — Data Loading and Preprocessing (Strict and Secure Execution Order)
# ================================================================================
log_section("Step 2 | Data Loading and Preprocessing")

df_static_features   = pd.read_excel(STATIC_BORUTA_PATH)
base_static_features = df_static_features.iloc[:, 0].dropna().astype(str).tolist()

df = pd.read_csv(INPUT_DATA_PATH)
# 🎯 Automatic detection and correction mechanism for the TreeType column name in your CSV file
for col_name in df.columns:
    if 'tree' in col_name.lower():
        df = df.rename(columns={col_name: 'TreeType'})
        break
date_col = 'date' if 'date' in df.columns else 'Date'
df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)

if 'Site_ID'  in df.columns: df['Site_ID']  = df['Site_ID'].astype(str)
if 'Altitude' in df.columns: df['Altitude'] = df['Altitude'].astype(int)
df['Altitude_Original'] = df['Altitude'].copy()
df['RainMean_Backup']   = df['IMS_RainMean'].astype(float).copy()

log("Performing One-Hot Encoding for categorical variables...", level=1)
categorical_static   = ['TreeType', 'RockType', 'Aspect']
existing_categorical = [c for c in categorical_static if c in df.columns]
if existing_categorical:
    df = pd.get_dummies(df, columns=existing_categorical, drop_first=False, dtype=float)

STATIC_FEATURES = []
for col in base_static_features:
    if col in categorical_static:
        encoded = [c for c in df.columns if c.startswith(f"{col}_")]
        STATIC_FEATURES.extend(encoded)
    else:
        STATIC_FEATURES.append(col)

for col in ['Altitude', 'Slope', 'T/He_2022', 'Rock cover %']:
    if col not in STATIC_FEATURES:
        STATIC_FEATURES.append(col)

if 'Planting_year' in df.columns:
    df['Forest_Age'] = df[date_col].dt.year - df['Planting_year']
    df['Forest_Age'] = df['Forest_Age'].fillna(df['Forest_Age'].mean())
else:
    df['Forest_Age'] = 0
if 'Forest_Age' not in STATIC_FEATURES:
    STATIC_FEATURES.append('Forest_Age')

# 1. Filter out missing values first
log("Filtering out rows with missing values...", level=1)
df_final = df.dropna(subset=[TARGET_COL] + DYNAMIC_FEATURES_BASE + STATIC_FEATURES).copy()

# 2. Strict chronological sorting and index reset to preserve smooth daily sequence
df_final = df_final.sort_values([date_col]).reset_index(drop=True)

# 3. Only now calculate temporal features (preventing indexing shifts and window distortions)
log("Calculating Hydrological Year and DOY on cleaned data...", level=1)
df_final['Hydrological_Year'] = df_final[date_col].dt.year
df_final.loc[df_final[date_col].dt.month >= 10, 'Hydrological_Year'] = df_final[date_col].dt.year + 1
df_final['DOY'] = df_final[date_col].dt.dayofyear

# Dynamic features remain clean without DOY or Hydrological Year
DYNAMIC_FEATURES = DYNAMIC_FEATURES_BASE

df_original_physical = df_final.copy()

log("Normalizing with StandardScaler (Z-score) preventing test year data leakage...", level=1)

# 🛡️ Strict protection mechanism: Merging lists and actively filtering non-numeric columns in DataFrame
ALL_INPUT_COLS = DYNAMIC_FEATURES + STATIC_FEATURES
ALL_FEATURE_COLS = [c for c in ALL_INPUT_COLS if c in df_final.columns and pd.api.types.is_numeric_dtype(df_final[c])]

# Updating static features list based on filtering so Step 5 won't look for text columns in metadata
STATIC_FEATURES = [c for c in ALL_FEATURE_COLS if c in STATIC_FEATURES or any(c.startswith(f"{cat}_") for cat in categorical_static)]

# 📢 Verbose console output of all input variables fed into the model
print("\n" + "="*75)
print(f"📊 🔍 HARDWARE & FEATURE AUDIT MONITOR (CONSOLE VERBOSE TRACKING)")
print(f"    - Total Rows Loaded & Cleaned: {len(df_final):,}")
print(f"    - Target Column (Y): '{TARGET_COL}'")
print(f"    - Dynamic Features to LSTM ({len(DYNAMIC_FEATURES)}): {DYNAMIC_FEATURES}")
print(f"    - Static Features post-OneHot ({len(STATIC_FEATURES)}): {STATIC_FEATURES}")
print(f"    - Total Input Dimension for Late Fusion: {len(ALL_FEATURE_COLS)}")
print("="*75 + "\n")

scaler_x = StandardScaler()
scaler_y = StandardScaler()

# Fit scaler parameters solely on training/validation years prior to hydrological year 2024
train_val_indices = df_final[df_final['Hydrological_Year'] < 2024].index
scaler_x.fit(df_final.loc[train_val_indices, ALL_FEATURE_COLS])
scaler_y.fit(df_final.loc[train_val_indices, [TARGET_COL]])

# Transform the complete dataset
df_final[ALL_FEATURE_COLS] = scaler_x.transform(df_final[ALL_FEATURE_COLS])
df_final[[TARGET_COL]]     = scaler_y.transform(df_final[[TARGET_COL]])

def denormalize_gpp(val_array):
    if isinstance(val_array, torch.Tensor):
        val_array = val_array.detach().cpu().numpy()
    if val_array.ndim == 1:
        val_array = val_array.reshape(-1, 1)
    return scaler_y.inverse_transform(val_array).flatten()

def calc_metrics(actual, pred):
    mae    = float(np.mean(np.abs(actual - pred)))
    rmse   = float(np.sqrt(np.mean((actual - pred) ** 2)))
    bias   = float(np.mean(pred - actual))
    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2     = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return r2, rmse, mae, bias

os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)


# ================================================================================
# Step 3 — Strict Spatial and Temporal Split (Hydrological 2024)
# ================================================================================
log_section("Step 3 | Strict Spatial and Temporal Split (Hydrological Spatial-Temporal Split)")

df_test_2024      = df_final[df_final['Hydrological_Year'] == 2024].copy()
df_train_val_pool = df_final[df_final['Hydrological_Year'] < 2024].copy()

# 1. Split sites within existing pool into 80% training and 20% validation
all_sites = df_train_val_pool['Site_ID'].unique()
train_sites, val_sites = train_test_split(all_sites, test_size=0.20, random_state=SEED)

# 2. Build final DataFrames for train and validation
df_train = df_train_val_pool[df_train_val_pool['Site_ID'].isin(train_sites)].copy()
df_val   = df_train_val_pool[df_train_val_pool['Site_ID'].isin(val_sites)].copy()

log(f"Train (80% sites): {len(df_train):,} rows")
log(f"Val   (20% sites): {len(df_val):,} rows")

log(f"Test (Hydrological Year 2024): {len(df_test_2024):,} rows")


# ================================================================================
# Step 4 — Model Architecture: Static-Conditioned Attention + Deep Fusion Head
# ================================================================================
import torch
import torch.nn as nn

class StaticConditionedAttention(nn.Module):
    def __init__(self, hidden_dim, static_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim + static_dim, 1)

    def forward(self, lstm_out, x_static):
        batch_size, seq_len, hidden_dim = lstm_out.size()
        x_static_expanded = x_static.unsqueeze(1).repeat(1, seq_len, 1)
        combined_for_attn = torch.cat((lstm_out, x_static_expanded), dim=2)
        
        scores  = self.attn(combined_for_attn).squeeze(-1)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        context = (weights * lstm_out).sum(dim=1)
        return context

class LateFusionAttentionLSTM(nn.Module):
    def __init__(self, dynamic_dim, static_dim, hidden_dim, num_layers, dropout_rate):
        super().__init__()
        self.lstm      = nn.LSTM(
            input_size=dynamic_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )
        
        # --- UPGRADE 1: Conditioned Attention ---
        self.attention = StaticConditionedAttention(hidden_dim, static_dim)
        self.bn        = nn.BatchNorm1d(hidden_dim)
        self.dropout   = nn.Dropout(dropout_rate)
        
        # --- UPGRADE 2: Deep Head ---
        self.fc_dense  = nn.Linear(hidden_dim + static_dim, hidden_dim)
        self.activation = nn.GELU()
        self.fc_out    = nn.Linear(hidden_dim, 1)

    def forward(self, x_dynamic, x_static):
        lstm_out, _ = self.lstm(x_dynamic)
        
        # --- UPGRADE 1 FLOW ---
        context     = self.attention(lstm_out, x_static)
        context     = self.bn(context)
        
        combined    = torch.cat((context, x_static), dim=1)
        combined    = self.dropout(combined)
        
        # --- UPGRADE 2 FLOW ---
        dense_out   = self.fc_dense(combined)
        dense_out   = self.activation(dense_out)
        dense_out   = self.dropout(dense_out)
        
        return self.fc_out(dense_out).squeeze(-1)
    
# ================================================================================
# Step 4.5 — Architecture Visualization Graph Export (torchviz)
# ================================================================================
log_section("Step 4.5 | Architecture Flow Diagram Generation (Visual Graph)")

try:
    from torchviz import make_dot
    
    log("Creating dummy model and tensors for computational graph mapping...", level=1)
    
    # 1. Create a temporary dummy model using current feature counts
    dummy_model = LateFusionAttentionLSTM(
        dynamic_dim=len(DYNAMIC_FEATURES),
        static_dim=len(STATIC_FEATURES),
        hidden_dim=64, # Arbitrary for visualization
        num_layers=2,
        dropout_rate=0.2
    ).to(device)
    
    # 🏆 Set dummy model to evaluation mode to disable BatchNorm tracking
    dummy_model.eval()
    
    # 2. Create dummy tensors matching expected input shape (Batch=1, Seq_len=30)
    x_dyn_dummy  = torch.randn(1, 30, len(DYNAMIC_FEATURES)).to(device)
    x_stat_dummy = torch.randn(1, len(STATIC_FEATURES)).to(device)
    
    # 3. Forward pass to trace computational graph
    y_pred_dummy = dummy_model(x_dyn_dummy, x_stat_dummy)
    
    # 4. Generate computational graph
    graph = make_dot(
        y_pred_dummy,
        params=dict(dummy_model.named_parameters()),
        show_attrs=True,
        show_saved=True
    )
    
    # 5. Define output path
    graph_filename = f"LSTM_Architecture_Graph_{DATE_SUFFIX}"
    graph_output_path = os.path.join(BASE_OUTPUT_DIR, graph_filename)
    
    # 6. Render and save as PNG (cleanup=True deletes intermediate .gv file)
    graph.format = 'png'
    graph.render(filename=graph_output_path, format='png', cleanup=True)
    
    log(f"Architecture graph saved successfully:", level=1)
    log(f"📂 {graph_output_path}.png", level=2)
    
    # 7. Clean up VRAM/RAM memory to prevent impact on Step 6 Tuning
    del dummy_model, x_dyn_dummy, x_stat_dummy, y_pred_dummy
    torch.cuda.empty_cache()

except ImportError:
    log("⚠️ 'torchviz' library is not installed. Skipping architecture graph rendering.", level=1)
    log("To install: pip install torchviz graphviz", level=2)
except FileNotFoundError:
    log("⚠️ Graphviz binary missing on system OS (required for torchviz). Skipping.", level=1)
except Exception as e:
    log(f"⚠️ Error generating architecture diagram: {str(e)}", level=1)

# ================================================================================


# ================================================================================
# Step 5 — Geographically Synchronized Dataset Class
# ================================================================================
log_section("Step 5 | AlignedSequencesDataset Definition")

class AlignedSequencesDataset(Dataset):
    def __init__(self, norm_df, orig_df, sequence_length, date_column, dynamic_cols, static_cols, target_col, split_name=""):
        self.sequence_length = sequence_length
        self.reconstructed_meta = []
        
        norm_df_reset = norm_df.reset_index(drop=True)
        orig_df_reset = orig_df.reset_index(drop=True)
        
        self.dynamic_tensor = torch.tensor(norm_df_reset[dynamic_cols].values, dtype=torch.float32)
        self.static_tensor = torch.tensor(norm_df_reset[static_cols].values, dtype=torch.float32)
        self.target_tensor = torch.tensor(norm_df_reset[target_col].values, dtype=torch.float32)
        
        self.valid_indices = []
        unique_alts = sorted(norm_df_reset['Altitude_Original'].unique())
        log(f"    [{split_name}] Building Dataset across {len(unique_alts)} sites", level=1)

        for alt_key in unique_alts:
            indices = norm_df_reset[norm_df_reset['Altitude_Original'] == alt_key].index.tolist()
            if len(indices) > sequence_length:
                self.valid_indices.extend(indices[sequence_length:])
                
                grp_orig = orig_df_reset.loc[indices[sequence_length:]]
                for _, row in grp_orig.iterrows():
                    meta = {'Date': row[date_column], 'Site_ID': row['Site_ID'], 'Altitude': row['Altitude_Original']}
                    for feat_name in dynamic_cols + static_cols:
                        meta[feat_name] = float(row[feat_name])
                    self.reconstructed_meta.append(meta)

    def __len__(self): return len(self.valid_indices)
    def __getitem__(self, idx):
        target_idx = self.valid_indices[idx]
        return self.dynamic_tensor[target_idx - self.sequence_length : target_idx], self.static_tensor[target_idx], self.target_tensor[target_idx]


# ================================================================================
# Step 6 — Hyperparameter Search Grid
# ================================================================================
HYPERPARAM_GRID = {
    'hidden_units':  [32, 64, 128, 256], 'lstm_layers':   [2], 'dropout':       [0.1, 0.2, 0.3, 0.4],
    'learning_rate': [0.001, 0.0005, 0.0001, 1e-5], 'batch_size':    [32, 64], 'weight_decay':  [0, 1e-5, 1e-4, 1e-3],
}
keys, values          = zip(*HYPERPARAM_GRID.items())
all_grid_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

def get_trial_params(trial):
    return {
        'hidden_units':  trial.suggest_categorical('hidden_units', [32, 64, 128, 256]),
        'lstm_layers':   trial.suggest_categorical('lstm_layers', [2]),
        'dropout':       trial.suggest_categorical('dropout', [0.1, 0.2, 0.3, 0.4]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.001, 0.0005, 0.0001, 1e-5]),
        'batch_size':    trial.suggest_categorical('batch_size', [32, 64]),
        'weight_decay':  trial.suggest_categorical('weight_decay', [0, 1e-5, 1e-4, 1e-3])
    }

# ================================================================================
# Main Time-Lag Windows Loop
# ================================================================================
summary_stats = []
summary_output_path = os.path.join(BASE_OUTPUT_DIR, f"LSTM_Model_Performance_Summary_no_LAI_SPEI_6_no_GDD_HW{DATE_SUFFIX}.xlsx")

if os.path.exists(summary_output_path):
    try: summary_stats = pd.read_excel(summary_output_path).to_dict('records')
    except Exception: pass

for L in WINDOWS:
    log_separator("═")
    log(f"Lag Window L={L} Days — Initiating Process")
    log_separator("═")

    val_path         = os.path.join(BASE_OUTPUT_DIR, f"Validation_Data_no_LAI_SPEI_6_no_GDD_HW_L{L}_{DATE_SUFFIX}.csv")
    shap_path        = os.path.join(BASE_OUTPUT_DIR, f"Detailed_Daily_SHAP_no_LAI_SPEI_6_no_GDD_HW_L{L}_{DATE_SUFFIX}.csv")
    importance_path  = os.path.join(BASE_OUTPUT_DIR, f"Feature_Importance_no_LAI_SPEI_6_no_GDD_HW_L{L}_{DATE_SUFFIX}.csv")
    loss_output_path = os.path.join(BASE_OUTPUT_DIR, f"Loss_Curve_no_LAI_SPEI_6_no_GDD_HW_L{L}_{DATE_SUFFIX}.xlsx")

    if all(os.path.exists(p) and os.path.getsize(p) > 0 for p in [val_path, shap_path, importance_path]) and any(r['Lag_Window_Days'] == L for r in summary_stats):
        log(f"[CHECKPOINT] L={L} already completed — Skipping")
        continue

    summary_stats = [r for r in summary_stats if r['Lag_Window_Days'] != L]

    df_train_pool = df_train_val_pool[df_train_val_pool['Site_ID'].isin(train_sites)].copy()
    df_val_pool   = df_train_val_pool[df_train_val_pool['Site_ID'].isin(val_sites)].copy()
    
    df_orig_test  = df_original_physical[df_original_physical['Hydrological_Year'] == 2024].copy()
    df_orig_tv    = df_original_physical[df_original_physical['Hydrological_Year'] < 2024].copy()
    df_orig_train = df_orig_tv[df_orig_tv['Site_ID'].isin(train_sites)].copy()
    df_orig_val   = df_orig_tv[df_orig_tv['Site_ID'].isin(val_sites)].copy()

    train_dataset = AlignedSequencesDataset(df_train_pool, df_orig_train, L, date_col, DYNAMIC_FEATURES, STATIC_FEATURES, TARGET_COL, "TRAIN")
    val_dataset   = AlignedSequencesDataset(df_val_pool, df_orig_val, L, date_col, DYNAMIC_FEATURES, STATIC_FEATURES, TARGET_COL, "VAL")
    test_dataset  = AlignedSequencesDataset(df_test_2024, df_orig_test, L, date_col, DYNAMIC_FEATURES, STATIC_FEATURES, TARGET_COL, "TEST")

    if len(train_dataset) == 0 or len(val_dataset) == 0: continue

    # 📢 TPE Tuning Engine (Optuna)
    log(f"-> Starting TPE Tuning Scan (L={L}): Running {MAX_TUNING_COMBINATIONS} trials...", level=1)
    
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
    global_best_val_loss = float('inf')
    best_config, final_epochs_globally = None, 0
    best_model_weights_final = None 
    
    for trial_idx in range(MAX_TUNING_COMBINATIONS):
        trial = study.ask()
        config = get_trial_params(trial)
        
        # 📢 Detailed log output
        log(f" Combo [{trial_idx+1:02d}/{MAX_TUNING_COMBINATIONS}] Testing: Hidden={config['hidden_units']} | Layers={config['lstm_layers']} | LR={config['learning_rate']} | Batch={config['batch_size']} | WD={config['weight_decay']}", level=1)

        t_model = LateFusionAttentionLSTM(len(DYNAMIC_FEATURES), len(STATIC_FEATURES), config['hidden_units'], config['lstm_layers'], config['dropout']).to(device)
        t_train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True, num_workers=0, drop_last=True)
        t_val_loader   = DataLoader(val_dataset,   batch_size=config['batch_size'], shuffle=False, pin_memory=True, num_workers=0, drop_last=True)

        t_optimizer = torch.optim.Adam(t_model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        t_criterion = nn.HuberLoss()
        t_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(t_optimizer, patience=5, factor=0.5)

        # 🛡️ Variable re-initialization per Trial to prevent size mismatch errors
        best_val_loss, patience_cnt = float('inf'), 0
        best_model_weights = None
        best_epoch = 0 
        loss_history = []
        window_train_loss = []
        window_val_loss = [] 
    
        for epoch in range(1, 1001):
            t_model.train()
            t_loss_sum = 0.0
            for bx_dyn, bx_stat, by in t_train_loader:
                bx_dyn, bx_stat, by = bx_dyn.to(device), bx_stat.to(device), by.to(device)
                t_optimizer.zero_grad()
                loss = t_criterion(t_model(bx_dyn, bx_stat), by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(t_model.parameters(), max_norm=1.0)
                t_optimizer.step()
                t_loss_sum += loss.item() * bx_dyn.size(0)
            t_train_loss = t_loss_sum / len(t_train_loader.dataset)

            t_model.eval()
            v_loss_sum = 0.0
            with torch.no_grad():
                for bx_dyn, bx_stat, by in t_val_loader:
                    bx_dyn, bx_stat, by = bx_dyn.to(device), bx_stat.to(device), by.to(device)
                    v_loss_sum += t_criterion(t_model(bx_dyn, bx_stat), by).item() * bx_dyn.size(0)
            v_loss = v_loss_sum / len(t_val_loader.dataset)
            
            # Step Learning Rate Scheduler
            t_scheduler.step(v_loss)
            
            loss_history.append({'Epoch': epoch, 'Train_Loss': t_train_loss, 'Val_Loss': v_loss})
            log(f"[L={L}] Epoch {epoch:03d}/1000 | Train: {t_train_loss:.6f} | Val: {v_loss:.6f}", level=2)

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_epoch = epoch
                patience_cnt = 0
                best_model_weights = {k: v.cpu() for k, v in t_model.state_dict().items()}
            else:
                patience_cnt += 1

            # Early stopping checks
            window_train_loss.append(t_train_loss)
            window_val_loss.append(v_loss)
            if len(window_train_loss) > 10:
                window_train_loss.pop(0)
                window_val_loss.pop(0)

            if len(window_train_loss) == 10:
                if ((window_train_loss[0] - window_train_loss[-1] < 0.003) and 
                    (window_val_loss[-1] - window_val_loss[0] > 0.004)) or \
                   ((v_loss - best_val_loss) > 0.015):
                    log(f"-> Early stopping triggered at epoch {epoch}!", level=2)
                    break
                    
            if patience_cnt >= 50: 
                log(f"Classic Early Stopping triggered.", level=2)
                break

        # Optuna trial reporting
        study.tell(trial, best_val_loss)
        
        # Best model tracking
        if best_val_loss < global_best_val_loss:
            global_best_val_loss = best_val_loss
            best_config = config
            best_model_weights_final = best_model_weights
            final_epochs_globally = best_epoch

        # Save training loss logs for trial
        df_loss = pd.DataFrame(loss_history)
        df_loss['Is_Best_Epoch'] = (df_loss['Epoch'] == best_epoch)
        df_loss.to_excel(loss_output_path, index=False)
        
        log(f"         ==> Training finished! Best Val Huber Loss = {best_val_loss:.6f} (Best Epoch: {best_epoch})", level=1)
        
    

    log_section(f"Step 7 | Generating Winning Model and Computing Final Metrics for Lag L={L}")
    log(f"Selected Winning Config: {best_config}", level=1)
    
    model = LateFusionAttentionLSTM(len(DYNAMIC_FEATURES), len(STATIC_FEATURES), best_config['hidden_units'], best_config['lstm_layers'], best_config['dropout']).to(device)
    log(f"DEBUG: Loading best weights into final model...", level=1)
    model.load_state_dict({k: v.to(device) for k, v in best_model_weights_final.items()})
    
    model.eval()

    def get_preds(dataset_obj):
        loader = DataLoader(dataset_obj, batch_size=256, shuffle=False, pin_memory=True, num_workers=0, drop_last=False)
        all_preds = []
        with torch.no_grad():
            for bx_dyn, bx_stat, _ in loader:
                all_preds.extend(model(bx_dyn.to(device), bx_stat.to(device)).cpu().numpy())
        return np.array(all_preds)

    df_train_out           = pd.DataFrame(train_dataset.reconstructed_meta)
    df_train_out['Actual'] = denormalize_gpp(train_dataset.target_tensor[train_dataset.valid_indices])
    df_train_out['Pred']   = denormalize_gpp(get_preds(train_dataset))
    df_train_out['Set']    = 'Train'

    df_val_out             = pd.DataFrame(val_dataset.reconstructed_meta)
    df_val_out['Actual']   = denormalize_gpp(val_dataset.target_tensor[val_dataset.valid_indices])
    df_val_out['Pred']     = denormalize_gpp(get_preds(val_dataset))
    df_val_out['Set']      = 'Validation'

    df_test_out            = pd.DataFrame(test_dataset.reconstructed_meta)
    df_test_out['Actual']  = denormalize_gpp(test_dataset.target_tensor[test_dataset.valid_indices])
    df_test_out['Pred']    = denormalize_gpp(get_preds(test_dataset))
    df_test_out['Set']     = 'Test'

    r2_val, rmse_val, mae_val, bias_val = calc_metrics(df_val_out['Actual'].values, df_val_out['Pred'].values)
    r2_train, rmse_train, mae_train, bias_train = calc_metrics(df_train_out['Actual'].values, df_train_out['Pred'].values)
    r2_test, rmse_test, mae_test, bias_test = calc_metrics(df_test_out['Actual'].values, df_test_out['Pred'].values)

    # 📢 Print final ecological metrics report for current lag window to console
    print("\n" + "✨"*35)
    print(f"📈 [OPTIMIZED LSTM ACCURACY REPORT FOR LAG L={L}]:")
    print(f"     >>> SPATIAL VALIDATION SET (Unseen sites):")
    print(f"          * R²:    {r2_val:.4f}   * RMSE: {rmse_val:.4f}   * MAE: {mae_val:.4f}   * Bias: {bias_val:.4f}")
    print(f"     >>> SPATIAL TRAIN SET (Training sites):")
    print(f"          * R²:    {r2_train:.4f}   * RMSE: {rmse_train:.4f}   * MAE: {mae_train:.4f}   * Bias: {bias_train:.4f}")
    print(f"     >>> TEMPORAL TEST SET (Hydrological Year 2024):")
    print(f"          * R²:    {r2_test:.4f}   * RMSE: {rmse_test:.4f}   * MAE: {mae_test:.4f}   * Bias: {bias_test:.4f}")
    print("✨"*35 + "\n")

    keep_cols  = ['Date', 'Site_ID', 'Altitude', 'Actual', 'Pred', 'Set']
    df_val_all = pd.concat([df_train_out[keep_cols], df_val_out[keep_cols], df_test_out[keep_cols]], axis=0).reset_index(drop=True)
    df_val_all.to_csv(val_path, index=False, encoding='utf-8-sig')

    # ================================================================================
    # ⚡ 🚀 Steps 9-11 — SHAP Calculation
    # ================================================================================
    log_section(f"Steps 9-11 | SHAP Computation Across All Data Splits | L={L}")
    
    model.train()
    torch.backends.cudnn.enabled = False

    def flatten_and_unify_dataset(dataset_obj, indices_to_sample):
        flat_list = []
        for idx_s in indices_to_sample:
            dyn_t, stat_t, _ = dataset_obj[idx_s]
            flat_unified = torch.cat((dyn_t.flatten(), stat_t))
            flat_list.append(flat_unified)
        return torch.stack(flat_list).to(device)

    # Prepare background dataset for SHAP
    bg_size = min(200, len(train_dataset))
    bg_idx = np.random.choice(len(train_dataset), bg_size, replace=False)
    bg_unified = flatten_and_unify_dataset(train_dataset, bg_idx)

    class HarmonizedSingleInputSHAPWrapper(nn.Module):
        def __init__(self, base_model, dynamic_dim, static_dim, sequence_length):
            super().__init__()
            self.base_model = base_model
            self.dynamic_dim = dynamic_dim
            self.static_dim = static_dim
            self.L = sequence_length
        def forward(self, x_flat_unified):
            batch_sz = x_flat_unified.shape[0]
            x_dynamic = x_flat_unified[:, :self.L * self.dynamic_dim].reshape(batch_sz, self.L, self.dynamic_dim)
            x_static = x_flat_unified[:, self.L * self.dynamic_dim:]
            out = self.base_model(x_dynamic, x_static)
            return out.unsqueeze(-1) if out.ndim == 1 else out

    shap_wrapper = HarmonizedSingleInputSHAPWrapper(model, len(DYNAMIC_FEATURES), len(STATIC_FEATURES), L)
    explainer = shap.DeepExplainer(shap_wrapper, bg_unified)

    datasets_to_explain = {
        'Train': (train_dataset, df_train_out),
        'Validation': (val_dataset, df_val_out),
        'Test': (test_dataset, df_test_out)
    }

    all_shap_dfs = [] 
    for set_name, (ds_obj, df_out_obj) in datasets_to_explain.items():
        log(f"Computing SHAP values for dataset: {set_name}...", level=1)
        n_shap = len(ds_obj)
        ds_unified = flatten_and_unify_dataset(ds_obj, range(n_shap))
        
        shap_values = explainer.shap_values(ds_unified, check_additivity=False)
        shap_raw = shap_values.detach().cpu().numpy() if isinstance(shap_values, torch.Tensor) else shap_values
        if isinstance(shap_raw, list): shap_raw = shap_raw[0]
        if shap_raw.ndim == 3 and shap_raw.shape[-1] == 1: shap_raw = shap_raw.squeeze(-1)

        split_point = L * len(DYNAMIC_FEATURES)
        shap_dyn_mean = shap_raw[:, :split_point].reshape(n_shap, L, len(DYNAMIC_FEATURES)).mean(axis=1)
        shap_stat_mean = shap_raw[:, split_point:]
        if shap_stat_mean.ndim == 3: shap_stat_mean = shap_stat_mean[:, 0, :]

        shap_dyn_df = pd.DataFrame(shap_dyn_mean, columns=[f"SHAP_{c}" for c in DYNAMIC_FEATURES])
        shap_stat_df = pd.DataFrame(shap_stat_mean, columns=[f"SHAP_{c}" for c in STATIC_FEATURES])
        
        df_shap_meta = pd.DataFrame(ds_obj.reconstructed_meta).reset_index(drop=True)
        df_shap_meta['Set'] = set_name
        df_shap_meta['GPP_Actual'] = df_out_obj['Actual'].values
        df_shap_meta['GPP_Predicted'] = df_out_obj['Pred'].values

        all_shap_dfs.append(pd.concat([df_shap_meta, shap_dyn_df, shap_stat_df], axis=1))

    # Save SHAP outputs
    df_shap_all_sets = pd.concat(all_shap_dfs, axis=0).reset_index(drop=True)
    df_shap_all_sets.to_csv(shap_path, index=False, encoding='utf-8-sig')

    all_shap_cols = [f"SHAP_{c}" for c in DYNAMIC_FEATURES + STATIC_FEATURES]
    mean_abs_shap = np.mean(np.abs(df_shap_all_sets[all_shap_cols].values), axis=0)
    df_imp_raw = pd.DataFrame({'Raw_Feature': DYNAMIC_FEATURES + STATIC_FEATURES, 'Importance': mean_abs_shap})

    def map_to_base(feat):
        for base_cat in ['TreeType', 'RockType', 'Aspect']:
            if feat.startswith(f"{base_cat}_"): return base_cat
        return feat

    df_imp_raw['Feature'] = df_imp_raw['Raw_Feature'].apply(map_to_base)
    df_importance = df_imp_raw.groupby('Feature', as_index=False)['Importance'].sum().sort_values('Importance', ascending=False).rename(columns={'Importance': 'Global_Mean_Abs_SHAP'})
    df_importance.to_csv(importance_path, index=False, encoding='utf-8-sig')

    torch.backends.cudnn.enabled = True
    model.eval()

    # ================================================================================
    # Step 12 — Updating Central Summary File, Disk Verification, and Memory Cleanup
    # ================================================================================
    summary_stats.append({
        'Lag_Window_Days': L, 'Epochs_Run': final_epochs_globally,
        'Val_R2': r2_val, 'Val_RMSE': rmse_val, 'Val_MAE': mae_val, 'Val_Bias': bias_val,
        'Train_R2': r2_train, 'Train_RMSE': rmse_train, 'Train_MAE': mae_train, 'Train_Bias': bias_train,
        'Test_R2': r2_test, 'Test_RMSE': rmse_test, 'Test_MAE': mae_test, 'Test_Bias': bias_test, 'Test_Year': 2024,
        'Hidden_Units': best_config['hidden_units'], 'LSTM_Layers': best_config['lstm_layers'], 'Dropout': best_config['dropout'],
        'Learning_Rate': best_config['learning_rate'], 'Batch_Size': best_config['batch_size'], 'Weight_Decay': best_config['weight_decay'],
        'Loss_Function': 'Huber', 'Architecture': 'LateFusion+Attention+BN', 'Normalization': 'StandardScaler',
    })
    pd.DataFrame(summary_stats).to_excel(summary_output_path, index=False)

    # 🛡️ Physical disk verification check before clearing memory
    all_secured, verify_attempts = False, 0
    while not all_secured and verify_attempts < 10:
        time.sleep(2)
        if all(os.path.exists(p) and os.path.getsize(p) > 0 for p in [val_path, shap_path, importance_path, loss_output_path, summary_output_path]): 
            all_secured = True
        else: 
            verify_attempts += 1

    if all_secured:
        del model, train_dataset, val_dataset, test_dataset
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log(f"[SUCCESS]: Lag L={L} results fully committed, cache cleared, and verified on disk.\n")
    else: 
        log(f"[ERROR]: Failed to verify disk writes for L={L}. Halting to prevent data loss.", level=1)
        break

# ================================================================================
# Pipeline Completion
# ================================================================================
log_section("Final Summary — Pipeline Execution Completed")
pd.DataFrame(summary_stats).to_excel(summary_output_path, index=False)
print("✨ PIPELINE COMPLETED SUCCESSFULLY WITH ALL SHAP FIXES ENFORCED ✨")


#%%

# Additional run with structural LAI calculated based on summer months, without GDD, HW
import os
import itertools
import random
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import shap
import optuna 

warnings.filterwarnings('ignore')

# ================================================================================
# Helper function for formatted logging with timestamps — for Spyder console tracking
# ================================================================================
def log(msg, level=0):
    """
    Formatted print with timestamp and indentation level.
    level=0 → Main header
    level=1 → Step details
    level=2 → Sub-details (each feature, epoch, etc.)
    """
    ts     = pd.Timestamp.now().strftime("%H:%M:%S")
    indent = "   " * level
    print(f"[{ts}] {indent}{msg}")

def log_separator(char="=", width=75):
    print(char * width)

def log_section(title):
    log_separator()
    log(f"  {title}")
    log_separator()


# ================================================================================
# Step 0 — Random Seed Fixing and Device Detection
# ================================================================================
log_section("Step 0 | Environment Settings and Seed Initialization")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log(f"Detected Device: {device}")
if torch.cuda.is_available():
    log(f"GPU: {torch.cuda.get_device_name(0)}", level=1)
log(f"Fixed Seed: {SEED} (Python, NumPy, PyTorch, CUDA)", level=1)


# ================================================================================
# Step 1 — Defining Paths, Variables, and Global Parameters
# ================================================================================
log_section("Step 1 | Paths and Global Parameter Definitions")

INPUT_DATA_PATH    = r"D:\Stav\research\Israel_polygons\final_excels\data_for_LSTM_final_with_SPEI_6_summer_LAI_stractual_no_GDD_HW.csv"
STATIC_BORUTA_PATH = r"D:\Stav\research\Israel_polygons\random forest\boruta_confirmed_static_features_LAI_summer_stractual.xlsx"
BASE_OUTPUT_DIR    = r"D:\Stav\research\Israel_polygons\LSTM"
DATE_SUFFIX        = "11.6.26_also_static_first_layer_and_prespton"

DYNAMIC_FEATURES_BASE = [
    'T_max', 'T_min', 'T_mean', 'daily_rain_mm', 'VPD kPa_mean', 
    'VPD kPa_min', 'VPD kPa_max', 'ETo_mm_day',  
     'SPEI_6'
]

TARGET_COL              = 'GPP'
WINDOWS                 = [7, 14, 30, 60, 90, 120, 180, 270]
MAX_TUNING_COMBINATIONS = 35

log(f"Input File:        {INPUT_DATA_PATH}")
log(f"Target Column:     {TARGET_COL}")


# ================================================================================
# Step 2 — Data Loading and Preprocessing (Strict and Secure Execution Order)
# ================================================================================
log_section("Step 2 | Data Loading and Preprocessing")

df_static_features   = pd.read_excel(STATIC_BORUTA_PATH)
base_static_features = df_static_features.iloc[:, 0].dropna().astype(str).tolist()

df = pd.read_csv(INPUT_DATA_PATH)
# 🎯 Automatic detection and correction mechanism for the TreeType column name in your CSV file
for col_name in df.columns:
    if 'tree' in col_name.lower():
        df = df.rename(columns={col_name: 'TreeType'})
        break
date_col = 'date' if 'date' in df.columns else 'Date'
df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)

if 'Site_ID'  in df.columns: df['Site_ID']  = df['Site_ID'].astype(str)
if 'Altitude' in df.columns: df['Altitude'] = df['Altitude'].astype(int)
df['Altitude_Original'] = df['Altitude'].copy()
df['RainMean_Backup']   = df['IMS_RainMean'].astype(float).copy()

log("Performing One-Hot Encoding for categorical variables...", level=1)
categorical_static   = ['TreeType', 'RockType', 'Aspect']
existing_categorical = [c for c in categorical_static if c in df.columns]
if existing_categorical:
    df = pd.get_dummies(df, columns=existing_categorical, drop_first=False, dtype=float)

STATIC_FEATURES = []
for col in base_static_features:
    if col in categorical_static:
        encoded = [c for c in df.columns if c.startswith(f"{col}_")]
        STATIC_FEATURES.extend(encoded)
    else:
        STATIC_FEATURES.append(col)

for col in ['Altitude', 'Slope', 'T/He_2022', 'Rock cover %']:
    if col not in STATIC_FEATURES:
        STATIC_FEATURES.append(col)

if 'Planting_year' in df.columns:
    df['Forest_Age'] = df[date_col].dt.year - df['Planting_year']
    df['Forest_Age'] = df['Forest_Age'].fillna(df['Forest_Age'].mean())
else:
    df['Forest_Age'] = 0
if 'Forest_Age' not in STATIC_FEATURES:
    STATIC_FEATURES.append('Forest_Age')

# 1. Filter out missing values first
log("Filtering out rows with missing values...", level=1)
df_final = df.dropna(subset=[TARGET_COL] + DYNAMIC_FEATURES_BASE + STATIC_FEATURES).copy()

# 2. Strict chronological sorting and index reset to preserve smooth daily sequence
df_final = df_final.sort_values([date_col]).reset_index(drop=True)

# 3. Only now calculate temporal features (preventing indexing shifts and window distortions)
log("Calculating Hydrological Year and DOY on cleaned data...", level=1)
df_final['Hydrological_Year'] = df_final[date_col].dt.year
df_final.loc[df_final[date_col].dt.month >= 10, 'Hydrological_Year'] = df_final[date_col].dt.year + 1
df_final['DOY'] = df_final[date_col].dt.dayofyear

# Dynamic features remain clean without DOY or Hydrological Year
DYNAMIC_FEATURES = DYNAMIC_FEATURES_BASE

df_original_physical = df_final.copy()

log("Normalizing with StandardScaler (Z-score) preventing test year data leakage...", level=1)

# 🛡️ Strict protection mechanism: Merging lists and actively filtering non-numeric columns in DataFrame
ALL_INPUT_COLS = DYNAMIC_FEATURES + STATIC_FEATURES
ALL_FEATURE_COLS = [c for c in ALL_INPUT_COLS if c in df_final.columns and pd.api.types.is_numeric_dtype(df_final[c])]

# Updating static features list based on filtering so Step 5 won't look for text columns in metadata
STATIC_FEATURES = [c for c in ALL_FEATURE_COLS if c in STATIC_FEATURES or any(c.startswith(f"{cat}_") for cat in categorical_static)]

# 📢 Verbose console output of all input variables fed into the model
print("\n" + "="*75)
print(f"📊 🔍 HARDWARE & FEATURE AUDIT MONITOR (CONSOLE VERBOSE TRACKING)")
print(f"    - Total Rows Loaded & Cleaned: {len(df_final):,}")
print(f"    - Target Column (Y): '{TARGET_COL}'")
print(f"    - Dynamic Features to LSTM ({len(DYNAMIC_FEATURES)}): {DYNAMIC_FEATURES}")
print(f"    - Static Features post-OneHot ({len(STATIC_FEATURES)}): {STATIC_FEATURES}")
print(f"    - Total Input Dimension for Late Fusion: {len(ALL_FEATURE_COLS)}")
print("="*75 + "\n")

scaler_x = StandardScaler()
scaler_y = StandardScaler()

# Fit scaler parameters solely on training/validation years prior to hydrological year 2024
train_val_indices = df_final[df_final['Hydrological_Year'] < 2024].index
scaler_x.fit(df_final.loc[train_val_indices, ALL_FEATURE_COLS])
scaler_y.fit(df_final.loc[train_val_indices, [TARGET_COL]])

# Transform the complete dataset
df_final[ALL_FEATURE_COLS] = scaler_x.transform(df_final[ALL_FEATURE_COLS])
df_final[[TARGET_COL]]     = scaler_y.transform(df_final[[TARGET_COL]])

def denormalize_gpp(val_array):
    if isinstance(val_array, torch.Tensor):
        val_array = val_array.detach().cpu().numpy()
    if val_array.ndim == 1:
        val_array = val_array.reshape(-1, 1)
    return scaler_y.inverse_transform(val_array).flatten()

def calc_metrics(actual, pred):
    mae    = float(np.mean(np.abs(actual - pred)))
    rmse   = float(np.sqrt(np.mean((actual - pred) ** 2)))
    bias   = float(np.mean(pred - actual))
    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2     = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return r2, rmse, mae, bias

os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)


# ================================================================================
# Step 3 — Strict Spatial and Temporal Split (Hydrological 2024)
# ================================================================================
log_section("Step 3 | Strict Spatial and Temporal Split (Hydrological Spatial-Temporal Split)")

df_test_2024      = df_final[df_final['Hydrological_Year'] == 2024].copy()
df_train_val_pool = df_final[df_final['Hydrological_Year'] < 2024].copy()

# 1. Split sites within existing pool into 80% training and 20% validation
all_sites = df_train_val_pool['Site_ID'].unique()
train_sites, val_sites = train_test_split(all_sites, test_size=0.20, random_state=SEED)

# 2. Build final DataFrames for train and validation
df_train = df_train_val_pool[df_train_val_pool['Site_ID'].isin(train_sites)].copy()
df_val   = df_train_val_pool[df_train_val_pool['Site_ID'].isin(val_sites)].copy()

log(f"Train (80% sites): {len(df_train):,} rows")
log(f"Val   (20% sites): {len(df_val):,} rows")

log(f"Test (Hydrological Year 2024): {len(df_test_2024):,} rows")


# ================================================================================
# Step 4 — Model Architecture: Static-Conditioned Attention + Deep Fusion Head
# ================================================================================
import torch
import torch.nn as nn

class StaticConditionedAttention(nn.Module):
    def __init__(self, hidden_dim, static_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim + static_dim, 1)

    def forward(self, lstm_out, x_static):
        batch_size, seq_len, hidden_dim = lstm_out.size()
        x_static_expanded = x_static.unsqueeze(1).repeat(1, seq_len, 1)
        combined_for_attn = torch.cat((lstm_out, x_static_expanded), dim=2)
        
        scores  = self.attn(combined_for_attn).squeeze(-1)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        context = (weights * lstm_out).sum(dim=1)
        return context

class LateFusionAttentionLSTM(nn.Module):
    def __init__(self, dynamic_dim, static_dim, hidden_dim, num_layers, dropout_rate):
        super().__init__()
        self.lstm      = nn.LSTM(
            input_size=dynamic_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )
        
        # --- UPGRADE 1: Conditioned Attention ---
        self.attention = StaticConditionedAttention(hidden_dim, static_dim)
        self.bn        = nn.BatchNorm1d(hidden_dim)
        self.dropout   = nn.Dropout(dropout_rate)
        
        # --- UPGRADE 2: Deep Head ---
        self.fc_dense  = nn.Linear(hidden_dim + static_dim, hidden_dim)
        self.activation = nn.GELU()
        self.fc_out    = nn.Linear(hidden_dim, 1)

    def forward(self, x_dynamic, x_static):
        lstm_out, _ = self.lstm(x_dynamic)
        
        # --- UPGRADE 1 FLOW ---
        context     = self.attention(lstm_out, x_static)
        context     = self.bn(context)
        
        combined    = torch.cat((context, x_static), dim=1)
        combined    = self.dropout(combined)
        
        # --- UPGRADE 2 FLOW ---
        dense_out   = self.fc_dense(combined)
        dense_out   = self.activation(dense_out)
        dense_out   = self.dropout(dense_out)
        
        return self.fc_out(dense_out).squeeze(-1)
    
# ================================================================================
# Step 4.5 — Architecture Visualization Graph Export (torchviz)
# ================================================================================
log_section("Step 4.5 | Architecture Flow Diagram Generation (Visual Graph)")

try:
    from torchviz import make_dot
    
    log("Creating dummy model and tensors for computational graph mapping...", level=1)
    
    # 1. Create a temporary dummy model using current feature counts
    dummy_model = LateFusionAttentionLSTM(
        dynamic_dim=len(DYNAMIC_FEATURES),
        static_dim=len(STATIC_FEATURES),
        hidden_dim=64, # Arbitrary for visualization
        num_layers=2,
        dropout_rate=0.2
    ).to(device)
    
    # 🏆 Set dummy model to evaluation mode to disable BatchNorm tracking
    dummy_model.eval()
    
    # 2. Create dummy tensors matching expected input shape (Batch=1, Seq_len=30)
    x_dyn_dummy  = torch.randn(1, 30, len(DYNAMIC_FEATURES)).to(device)
    x_stat_dummy = torch.randn(1, len(STATIC_FEATURES)).to(device)
    
    # 3. Forward pass to trace computational graph
    y_pred_dummy = dummy_model(x_dyn_dummy, x_stat_dummy)
    
    # 4. Generate computational graph
    graph = make_dot(
        y_pred_dummy,
        params=dict(dummy_model.named_parameters()),
        show_attrs=True,
        show_saved=True
    )
    
    # 5. Define output path
    graph_filename = f"LSTM_Architecture_Graph_{DATE_SUFFIX}"
    graph_output_path = os.path.join(BASE_OUTPUT_DIR, graph_filename)
    
    # 6. Render and save as PNG (cleanup=True deletes intermediate .gv file)
    graph.format = 'png'
    graph.render(filename=graph_output_path, format='png', cleanup=True)
    
    log(f"Architecture graph saved successfully:", level=1)
    log(f"📂 {graph_output_path}.png", level=2)
    
    # 7. Clean up VRAM/RAM memory to prevent impact on Step 6 Tuning
    del dummy_model, x_dyn_dummy, x_stat_dummy, y_pred_dummy
    torch.cuda.empty_cache()

except ImportError:
    log("⚠️ 'torchviz' library is not installed. Skipping architecture graph rendering.", level=1)
    log("To install: pip install torchviz graphviz", level=2)
except FileNotFoundError:
    log("⚠️ Graphviz binary missing on system OS (required for torchviz). Skipping.", level=1)
except Exception as e:
    log(f"⚠️ Error generating architecture diagram: {str(e)}", level=1)

# ================================================================================


# ================================================================================
# Step 5 — Geographically Synchronized Dataset Class
# ================================================================================
log_section("Step 5 | AlignedSequencesDataset Definition")

class AlignedSequencesDataset(Dataset):
    def __init__(self, norm_df, orig_df, sequence_length, date_column, dynamic_cols, static_cols, target_col, split_name=""):
        self.sequence_length = sequence_length
        self.reconstructed_meta = []
        
        norm_df_reset = norm_df.reset_index(drop=True)
        orig_df_reset = orig_df.reset_index(drop=True)
        
        self.dynamic_tensor = torch.tensor(norm_df_reset[dynamic_cols].values, dtype=torch.float32)
        self.static_tensor = torch.tensor(norm_df_reset[static_cols].values, dtype=torch.float32)
        self.target_tensor = torch.tensor(norm_df_reset[target_col].values, dtype=torch.float32)
        
        self.valid_indices = []
        unique_alts = sorted(norm_df_reset['Altitude_Original'].unique())
        log(f"    [{split_name}] Building Dataset across {len(unique_alts)} sites", level=1)

        for alt_key in unique_alts:
            indices = norm_df_reset[norm_df_reset['Altitude_Original'] == alt_key].index.tolist()
            if len(indices) > sequence_length:
                self.valid_indices.extend(indices[sequence_length:])
                
                grp_orig = orig_df_reset.loc[indices[sequence_length:]]
                for _, row in grp_orig.iterrows():
                    meta = {'Date': row[date_column], 'Site_ID': row['Site_ID'], 'Altitude': row['Altitude_Original']}
                    for feat_name in dynamic_cols + static_cols:
                        meta[feat_name] = float(row[feat_name])
                    self.reconstructed_meta.append(meta)

    def __len__(self): return len(self.valid_indices)
    def __getitem__(self, idx):
        target_idx = self.valid_indices[idx]
        return self.dynamic_tensor[target_idx - self.sequence_length : target_idx], self.static_tensor[target_idx], self.target_tensor[target_idx]


# ================================================================================
# Step 6 — Hyperparameter Search Grid
# ================================================================================
HYPERPARAM_GRID = {
    'hidden_units':  [32, 64, 128, 256], 'lstm_layers':   [2], 'dropout':       [0.1, 0.2, 0.3, 0.4],
    'learning_rate': [0.001, 0.0005, 0.0001, 1e-5], 'batch_size':    [32, 64], 'weight_decay':  [0, 1e-5, 1e-4, 1e-3],
}
keys, values          = zip(*HYPERPARAM_GRID.items())
all_grid_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

def get_trial_params(trial):
    return {
        'hidden_units':  trial.suggest_categorical('hidden_units', [32, 64, 128, 256]),
        'lstm_layers':   trial.suggest_categorical('lstm_layers', [2]),
        'dropout':       trial.suggest_categorical('dropout', [0.1, 0.2, 0.3, 0.4]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.001, 0.0005, 0.0001, 1e-5]),
        'batch_size':    trial.suggest_categorical('batch_size', [32, 64]),
        'weight_decay':  trial.suggest_categorical('weight_decay', [0, 1e-5, 1e-4, 1e-3])
    }

# ================================================================================
# Main Time-Lag Windows Loop
# ================================================================================
summary_stats = []
summary_output_path = os.path.join(BASE_OUTPUT_DIR, f"LSTM_Model_Performance_Summary_LAI_stractual_SPEI_6_no_GDD_HW{DATE_SUFFIX}.xlsx")

if os.path.exists(summary_output_path):
    try: summary_stats = pd.read_excel(summary_output_path).to_dict('records')
    except Exception: pass

for L in WINDOWS:
    log_separator("═")
    log(f"Lag Window L={L} Days — Initiating Process")
    log_separator("═")

    val_path         = os.path.join(BASE_OUTPUT_DIR, f"Validation_Data_LAI_stractual_SPEI_6_no_GDD_HW_L{L}_{DATE_SUFFIX}.csv")
    shap_path        = os.path.join(BASE_OUTPUT_DIR, f"Detailed_Daily_SHAP_LAI_stractual_SPEI_6_no_GDD_HW_L{L}_{DATE_SUFFIX}.csv")
    importance_path  = os.path.join(BASE_OUTPUT_DIR, f"Feature_Importance_LAI_stractual_SPEI_6no_GDD_HW__L{L}_{DATE_SUFFIX}.csv")
    loss_output_path = os.path.join(BASE_OUTPUT_DIR, f"Loss_Curve_LAI_stractual_SPEI_6_no_GDD_HW_L{L}_{DATE_SUFFIX}.xlsx")

    if all(os.path.exists(p) and os.path.getsize(p) > 0 for p in [val_path, shap_path, importance_path]) and any(r['Lag_Window_Days'] == L for r in summary_stats):
        log(f"[CHECKPOINT] L={L} already completed — Skipping")
        continue

    summary_stats = [r for r in summary_stats if r['Lag_Window_Days'] != L]

    df_train_pool = df_train_val_pool[df_train_val_pool['Site_ID'].isin(train_sites)].copy()
    df_val_pool   = df_train_val_pool[df_train_val_pool['Site_ID'].isin(val_sites)].copy()
    
    df_orig_test  = df_original_physical[df_original_physical['Hydrological_Year'] == 2024].copy()
    df_orig_tv    = df_original_physical[df_original_physical['Hydrological_Year'] < 2024].copy()
    df_orig_train = df_orig_tv[df_orig_tv['Site_ID'].isin(train_sites)].copy()
    df_orig_val   = df_orig_tv[df_orig_tv['Site_ID'].isin(val_sites)].copy()

    train_dataset = AlignedSequencesDataset(df_train_pool, df_orig_train, L, date_col, DYNAMIC_FEATURES, STATIC_FEATURES, TARGET_COL, "TRAIN")
    val_dataset   = AlignedSequencesDataset(df_val_pool, df_orig_val, L, date_col, DYNAMIC_FEATURES, STATIC_FEATURES, TARGET_COL, "VAL")
    test_dataset  = AlignedSequencesDataset(df_test_2024, df_orig_test, L, date_col, DYNAMIC_FEATURES, STATIC_FEATURES, TARGET_COL, "TEST")

    if len(train_dataset) == 0 or len(val_dataset) == 0: continue

    # 📢 TPE Tuning Engine (Optuna)
    log(f"-> Starting TPE Tuning Scan (L={L}): Running {MAX_TUNING_COMBINATIONS} trials...", level=1)
    
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
    global_best_val_loss = float('inf')
    best_config, final_epochs_globally = None, 0
    best_model_weights_final = None 
    
    for trial_idx in range(MAX_TUNING_COMBINATIONS):
        trial = study.ask()
        config = get_trial_params(trial)
        
        # 📢 Detailed log output
        log(f" Combo [{trial_idx+1:02d}/{MAX_TUNING_COMBINATIONS}] Testing: Hidden={config['hidden_units']} | Layers={config['lstm_layers']} | LR={config['learning_rate']} | Batch={config['batch_size']} | WD={config['weight_decay']}", level=1)

        t_model = LateFusionAttentionLSTM(len(DYNAMIC_FEATURES), len(STATIC_FEATURES), config['hidden_units'], config['lstm_layers'], config['dropout']).to(device)
        t_train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True, num_workers=0, drop_last=True)
        t_val_loader   = DataLoader(val_dataset,   batch_size=config['batch_size'], shuffle=False, pin_memory=True, num_workers=0, drop_last=True)

        t_optimizer = torch.optim.Adam(t_model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        t_criterion = nn.HuberLoss()
        t_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(t_optimizer, patience=5, factor=0.5)

        # 🛡️ Variable re-initialization per Trial to prevent size mismatch errors
        best_val_loss, patience_cnt = float('inf'), 0
        best_model_weights = None
        best_epoch = 0 
        loss_history = []
        window_train_loss = []
        window_val_loss = [] 
    
        for epoch in range(1, 1001):
            t_model.train()
            t_loss_sum = 0.0
            for bx_dyn, bx_stat, by in t_train_loader:
                bx_dyn, bx_stat, by = bx_dyn.to(device), bx_stat.to(device), by.to(device)
                t_optimizer.zero_grad()
                loss = t_criterion(t_model(bx_dyn, bx_stat), by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(t_model.parameters(), max_norm=1.0)
                t_optimizer.step()
                t_loss_sum += loss.item() * bx_dyn.size(0)
            t_train_loss = t_loss_sum / len(t_train_loader.dataset)

            t_model.eval()
            v_loss_sum = 0.0
            with torch.no_grad():
                for bx_dyn, bx_stat, by in t_val_loader:
                    bx_dyn, bx_stat, by = bx_dyn.to(device), bx_stat.to(device), by.to(device)
                    v_loss_sum += t_criterion(t_model(bx_dyn, bx_stat), by).item() * bx_dyn.size(0)
            v_loss = v_loss_sum / len(t_val_loader.dataset)
            
            # Step Learning Rate Scheduler
            t_scheduler.step(v_loss)
            
            loss_history.append({'Epoch': epoch, 'Train_Loss': t_train_loss, 'Val_Loss': v_loss})
            log(f"[L={L}] Epoch {epoch:03d}/1000 | Train: {t_train_loss:.6f} | Val: {v_loss:.6f}", level=2)

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_epoch = epoch
                patience_cnt = 0
                best_model_weights = {k: v.cpu() for k, v in t_model.state_dict().items()}
            else:
                patience_cnt += 1

            # Early stopping checks
            window_train_loss.append(t_train_loss)
            window_val_loss.append(v_loss)
            if len(window_train_loss) > 10:
                window_train_loss.pop(0)
                window_val_loss.pop(0)

            if len(window_train_loss) == 10:
                if ((window_train_loss[0] - window_train_loss[-1] < 0.003) and 
                    (window_val_loss[-1] - window_val_loss[0] > 0.004)) or \
                   ((v_loss - best_val_loss) > 0.015):
                    log(f"-> Early stopping triggered at epoch {epoch}!", level=2)
                    break
                    
            if patience_cnt >= 50: 
                log(f"Classic Early Stopping triggered.", level=2)
                break

        # Optuna trial reporting
        study.tell(trial, best_val_loss)
        
        # Best model tracking
        if best_val_loss < global_best_val_loss:
            global_best_val_loss = best_val_loss
            best_config = config
            best_model_weights_final = best_model_weights
            final_epochs_globally = best_epoch

        # Save training loss logs for trial
        df_loss = pd.DataFrame(loss_history)
        df_loss['Is_Best_Epoch'] = (df_loss['Epoch'] == best_epoch)
        df_loss.to_excel(loss_output_path, index=False)
        
        log(f"         ==> Training finished! Best Val Huber Loss = {best_val_loss:.6f} (Best Epoch: {best_epoch})", level=1)
        
    

    log_section(f"Step 7 | Generating Winning Model and Computing Final Metrics for Lag L={L}")
    log(f"Selected Winning Config: {best_config}", level=1)
    
    model = LateFusionAttentionLSTM(len(DYNAMIC_FEATURES), len(STATIC_FEATURES), best_config['hidden_units'], best_config['lstm_layers'], best_config['dropout']).to(device)
    log(f"DEBUG: Loading best weights into final model...", level=1)
    model.load_state_dict({k: v.to(device) for k, v in best_model_weights_final.items()})
    
    model.eval()

    def get_preds(dataset_obj):
        loader = DataLoader(dataset_obj, batch_size=256, shuffle=False, pin_memory=True, num_workers=0, drop_last=False)
        all_preds = []
        with torch.no_grad():
            for bx_dyn, bx_stat, _ in loader:
                all_preds.extend(model(bx_dyn.to(device), bx_stat.to(device)).cpu().numpy())
        return np.array(all_preds)

    df_train_out           = pd.DataFrame(train_dataset.reconstructed_meta)
    df_train_out['Actual'] = denormalize_gpp(train_dataset.target_tensor[train_dataset.valid_indices])
    df_train_out['Pred']   = denormalize_gpp(get_preds(train_dataset))
    df_train_out['Set']    = 'Train'

    df_val_out             = pd.DataFrame(val_dataset.reconstructed_meta)
    df_val_out['Actual']   = denormalize_gpp(val_dataset.target_tensor[val_dataset.valid_indices])
    df_val_out['Pred']     = denormalize_gpp(get_preds(val_dataset))
    df_val_out['Set']      = 'Validation'

    df_test_out            = pd.DataFrame(test_dataset.reconstructed_meta)
    df_test_out['Actual']  = denormalize_gpp(test_dataset.target_tensor[test_dataset.valid_indices])
    df_test_out['Pred']    = denormalize_gpp(get_preds(test_dataset))
    df_test_out['Set']     = 'Test'

    r2_val, rmse_val, mae_val, bias_val = calc_metrics(df_val_out['Actual'].values, df_val_out['Pred'].values)
    r2_train, rmse_train, mae_train, bias_train = calc_metrics(df_train_out['Actual'].values, df_train_out['Pred'].values)
    r2_test, rmse_test, mae_test, bias_test = calc_metrics(df_test_out['Actual'].values, df_test_out['Pred'].values)

    # 📢 Print final ecological metrics report for current lag window to console
    print("\n" + "✨"*35)
    print(f"📈 [OPTIMIZED LSTM ACCURACY REPORT FOR LAG L={L}]:")
    print(f"     >>> SPATIAL VALIDATION SET (Unseen sites):")
    print(f"          * R²:    {r2_val:.4f}   * RMSE: {rmse_val:.4f}   * MAE: {mae_val:.4f}   * Bias: {bias_val:.4f}")
    print(f"     >>> SPATIAL TRAIN SET (Training sites):")
    print(f"          * R²:    {r2_train:.4f}   * RMSE: {rmse_train:.4f}   * MAE: {mae_train:.4f}   * Bias: {bias_train:.4f}")
    print(f"     >>> TEMPORAL TEST SET (Hydrological Year 2024):")
    print(f"          * R²:    {r2_test:.4f}   * RMSE: {rmse_test:.4f}   * MAE: {mae_test:.4f}   * Bias: {bias_test:.4f}")
    print("✨"*35 + "\n")

    keep_cols  = ['Date', 'Site_ID', 'Altitude', 'Actual', 'Pred', 'Set']
    df_val_all = pd.concat([df_train_out[keep_cols], df_val_out[keep_cols], df_test_out[keep_cols]], axis=0).reset_index(drop=True)
    df_val_all.to_csv(val_path, index=False, encoding='utf-8-sig')

    # ================================================================================
    # ⚡ 🚀 Steps 9-11 — SHAP Computation
    # ================================================================================
    log_section(f"Steps 9-11 | SHAP Computation Across All Data Splits | L={L}")
    
    model.train()
    torch.backends.cudnn.enabled = False

    def flatten_and_unify_dataset(dataset_obj, indices_to_sample):
        flat_list = []
        for idx_s in indices_to_sample:
            dyn_t, stat_t, _ = dataset_obj[idx_s]
            flat_unified = torch.cat((dyn_t.flatten(), stat_t))
            flat_list.append(flat_unified)
        return torch.stack(flat_list).to(device)

    # Prepare background dataset for SHAP
    bg_size = min(200, len(train_dataset))
    bg_idx = np.random.choice(len(train_dataset), bg_size, replace=False)
    bg_unified = flatten_and_unify_dataset(train_dataset, bg_idx)

    class HarmonizedSingleInputSHAPWrapper(nn.Module):
        def __init__(self, base_model, dynamic_dim, static_dim, sequence_length):
            super().__init__()
            self.base_model = base_model
            self.dynamic_dim = dynamic_dim
            self.static_dim = static_dim
            self.L = sequence_length
        def forward(self, x_flat_unified):
            batch_sz = x_flat_unified.shape[0]
            x_dynamic = x_flat_unified[:, :self.L * self.dynamic_dim].reshape(batch_sz, self.L, self.dynamic_dim)
            x_static = x_flat_unified[:, self.L * self.dynamic_dim:]
            out = self.base_model(x_dynamic, x_static)
            return out.unsqueeze(-1) if out.ndim == 1 else out

    shap_wrapper = HarmonizedSingleInputSHAPWrapper(model, len(DYNAMIC_FEATURES), len(STATIC_FEATURES), L)
    explainer = shap.DeepExplainer(shap_wrapper, bg_unified)

    datasets_to_explain = {
        'Train': (train_dataset, df_train_out),
        'Validation': (val_dataset, df_val_out),
        'Test': (test_dataset, df_test_out)
    }

    all_shap_dfs = [] 
    for set_name, (ds_obj, df_out_obj) in datasets_to_explain.items():
        log(f"Computing SHAP values for dataset: {set_name}...", level=1)
        n_shap = len(ds_obj)
        ds_unified = flatten_and_unify_dataset(ds_obj, range(n_shap))
        
        shap_values = explainer.shap_values(ds_unified, check_additivity=False)
        shap_raw = shap_values.detach().cpu().numpy() if isinstance(shap_values, torch.Tensor) else shap_values
        if isinstance(shap_raw, list): shap_raw = shap_raw[0]
        if shap_raw.ndim == 3 and shap_raw.shape[-1] == 1: shap_raw = shap_raw.squeeze(-1)

        split_point = L * len(DYNAMIC_FEATURES)
        shap_dyn_mean = shap_raw[:, :split_point].reshape(n_shap, L, len(DYNAMIC_FEATURES)).mean(axis=1)
        shap_stat_mean = shap_raw[:, split_point:]
        if shap_stat_mean.ndim == 3: shap_stat_mean = shap_stat_mean[:, 0, :]

        shap_dyn_df = pd.DataFrame(shap_dyn_mean, columns=[f"SHAP_{c}" for c in DYNAMIC_FEATURES])
        shap_stat_df = pd.DataFrame(shap_stat_mean, columns=[f"SHAP_{c}" for c in STATIC_FEATURES])
        
        df_shap_meta = pd.DataFrame(ds_obj.reconstructed_meta).reset_index(drop=True)
        df_shap_meta['Set'] = set_name
        df_shap_meta['GPP_Actual'] = df_out_obj['Actual'].values
        df_shap_meta['GPP_Predicted'] = df_out_obj['Pred'].values

        all_shap_dfs.append(pd.concat([df_shap_meta, shap_dyn_df, shap_stat_df], axis=1))

    # Save SHAP outputs
    df_shap_all_sets = pd.concat(all_shap_dfs, axis=0).reset_index(drop=True)
    df_shap_all_sets.to_csv(shap_path, index=False, encoding='utf-8-sig')

    all_shap_cols = [f"SHAP_{c}" for c in DYNAMIC_FEATURES + STATIC_FEATURES]
    mean_abs_shap = np.mean(np.abs(df_shap_all_sets[all_shap_cols].values), axis=0)
    df_imp_raw = pd.DataFrame({'Raw_Feature': DYNAMIC_FEATURES + STATIC_FEATURES, 'Importance': mean_abs_shap})

    def map_to_base(feat):
        for base_cat in ['TreeType', 'RockType', 'Aspect']:
            if feat.startswith(f"{base_cat}_"): return base_cat
        return feat

    df_imp_raw['Feature'] = df_imp_raw['Raw_Feature'].apply(map_to_base)
    df_importance = df_imp_raw.groupby('Feature', as_index=False)['Importance'].sum().sort_values('Importance', ascending=False).rename(columns={'Importance': 'Global_Mean_Abs_SHAP'})
    df_importance.to_csv(importance_path, index=False, encoding='utf-8-sig')

    torch.backends.cudnn.enabled = True
    model.eval()

    # ================================================================================
    # Step 12 — Updating Central Summary File, Disk Verification, and Memory Cleanup
    # ================================================================================
    summary_stats.append({
        'Lag_Window_Days': L, 'Epochs_Run': final_epochs_globally,
        'Val_R2': r2_val, 'Val_RMSE': rmse_val, 'Val_MAE': mae_val, 'Val_Bias': bias_val,
        'Train_R2': r2_train, 'Train_RMSE': rmse_train, 'Train_MAE': mae_train, 'Train_Bias': bias_train,
        'Test_R2': r2_test, 'Test_RMSE': rmse_test, 'Test_MAE': mae_test, 'Test_Bias': bias_test, 'Test_Year': 2024,
        'Hidden_Units': best_config['hidden_units'], 'LSTM_Layers': best_config['lstm_layers'], 'Dropout': best_config['dropout'],
        'Learning_Rate': best_config['learning_rate'], 'Batch_Size': best_config['batch_size'], 'Weight_Decay': best_config['weight_decay'],
        'Loss_Function': 'Huber', 'Architecture': 'LateFusion+Attention+BN', 'Normalization': 'StandardScaler',
    })
    pd.DataFrame(summary_stats).to_excel(summary_output_path, index=False)

    # 🛡️ Physical disk verification check before clearing memory
    all_secured, verify_attempts = False, 0
    while not all_secured and verify_attempts < 10:
        time.sleep(2)
        if all(os.path.exists(p) and os.path.getsize(p) > 0 for p in [val_path, shap_path, importance_path, loss_output_path, summary_output_path]): 
            all_secured = True
        else: 
            verify_attempts += 1

    if all_secured:
        del model, train_dataset, val_dataset, test_dataset
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log(f"[SUCCESS]: Lag L={L} results fully committed, cache cleared, and verified on disk.\n")
    else: 
        log(f"[ERROR]: Failed to verify disk writes for L={L}. Halting to prevent data loss.", level=1)
        break

# ================================================================================
# Pipeline Completion
# ================================================================================
log_section("Final Summary — Pipeline Execution Completed")
pd.DataFrame(summary_stats).to_excel(summary_output_path, index=False)
print("✨ PIPELINE COMPLETED SUCCESSFULLY WITH ALL SHAP FIXES ENFORCED ✨")