"""
Amazon ML Challenge - Product Price Prediction
Transformer-based approach using DeBERTa/RoBERTa

Usage:
    python train_transformer.py

Requirements:
    pip install torch transformers pandas numpy tqdm scikit-learn
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoConfig,
    get_linear_schedule_with_warmup
)
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import warnings
from datetime import datetime
import json

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    """Training configuration - OPTIMIZED FOR 48GB VRAM + FP16"""
    
    # Model settings
    MODEL_NAME = 'roberta-large'  # Upgraded from roberta-base (355M params)
    MAX_LENGTH = 512  # Increased to 512 to capture full product descriptions (48GB VRAM)
    DROPOUT = 0.2
    
    # Training settings - Optimized for 10 epochs with FP16
    BATCH_SIZE = 16  # Increased from 12 (FP16 uses less memory!)
    GRADIENT_ACCUMULATION_STEPS = 3  # Effective batch size = 16 * 3 = 48
    EPOCHS = 10  # Extended for continued improvement
    LEARNING_RATE = 3e-5  # Optimized learning rate
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    MAX_GRAD_NORM = 1.0
    EARLY_STOPPING_PATIENCE = 3  # Increased to 3 for 10 epochs
    
    # Mixed Precision Training (FP16)
    USE_FP16 = True  # Enables automatic mixed precision (faster + less memory)
    FP16_OPT_LEVEL = 'O1'  # O1 = mixed precision
    
    # Data paths
    TRAIN_PATH = 'dataset/dataset/train_enhanced.csv'
    VAL_PATH = 'dataset/dataset/val_enhanced.csv'
    
    # Output paths
    OUTPUT_DIR = 'outputs'
    MODEL_SAVE_PATH = 'best_model.pt'
    
    # Device settings
    GPU_DEVICE = 1  # Use 2nd GPU (cuda:1)
    DEVICE = f'cuda:{GPU_DEVICE}' if torch.cuda.is_available() else 'cpu'
    REQUIRE_GPU = True  # Set to False to allow CPU training (NOT RECOMMENDED)
    
    # Random seed for reproducibility
    SEED = 42

def set_seed(seed):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ============================================================================
# DATASET CLASS
# ============================================================================
class ProductPriceDataset(Dataset):
    """Dataset for product price prediction"""
    
    def __init__(self, df, tokenizer, max_length=256, is_test=False):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
        
        # Clean data - remove NaN prices
        if not is_test:
            self.df = self.df.dropna(subset=['log_price']).reset_index(drop=True)
            print(f"Dataset size after cleaning: {len(self.df)}")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Tokenize text
        text = str(row['catalog_content'])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }
        
        if not self.is_test:
            item['target'] = torch.tensor(row['log_price'], dtype=torch.float)
        
        return item

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
class TransformerPricePredictor(nn.Module):
    """Transformer-based price prediction model"""
    
    def __init__(self, model_name, dropout=0.2):
        super().__init__()
        
        print(f"Loading transformer: {model_name}")
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Regression head with multiple layers
        hidden_size = self.config.hidden_size
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(128, 1)
        )
        
        # Initialize regression head weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights of regression head"""
        for module in self.regressor.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask):
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token embedding (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Predict log price
        log_price = self.regressor(cls_output)
        
        return log_price.squeeze(-1)

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, scaler=None):
    """Train for one epoch with gradient accumulation and optional FP16"""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    accumulation_steps = Config.GRADIENT_ACCUMULATION_STEPS
    use_amp = Config.USE_FP16 and scaler is not None
    
    optimizer.zero_grad()
    pbar = tqdm(dataloader, desc=f'Training Epoch {epoch}')
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['target'].to(device)
        
        # Forward pass with automatic mixed precision
        if use_amp:
            with torch.cuda.amp.autocast():
                predictions = model(input_ids, attention_mask)
                loss = nn.MSELoss()(predictions, targets)
                loss = loss / accumulation_steps
        else:
            predictions = model(input_ids, attention_mask)
            loss = nn.MSELoss()(predictions, targets)
            loss = loss / accumulation_steps
        
        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        total_loss += loss.item() * accumulation_steps
        
        # Update weights every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == num_batches:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=Config.MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=Config.MAX_GRAD_NORM)
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad()
        
        # Update progress bar
        avg_loss = total_loss / (batch_idx + 1)
        pbar.set_postfix({
            'loss': f'{loss.item() * accumulation_steps:.4f}',
            'avg_loss': f'{avg_loss:.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}',
            'eff_batch': f'{Config.BATCH_SIZE * accumulation_steps}',
            'fp16': 'ON' if use_amp else 'OFF'
        })
    
    return total_loss / num_batches

def validate(model, dataloader, device):
    """Validate model with optional FP16"""
    model.eval()
    predictions = []
    targets = []
    total_loss = 0
    use_amp = Config.USE_FP16
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target = batch['target'].to(device)
            
            # Predict with optional AMP
            if use_amp:
                with torch.cuda.amp.autocast():
                    pred = model(input_ids, attention_mask)
                    loss = nn.MSELoss()(pred, target)
            else:
                pred = model(input_ids, attention_mask)
                loss = nn.MSELoss()(pred, target)
            
            total_loss += loss.item()
            
            predictions.extend(pred.cpu().numpy())
            targets.extend(target.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    return np.array(predictions), np.array(targets), avg_loss

def compute_metrics(y_true_log, y_pred_log):
    """Compute evaluation metrics"""
    # Transform back to original scale
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    
    # SMAPE (Primary metric)
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    denominator = np.where(denominator == 0, 1e-10, denominator)
    smape = np.mean(numerator / denominator) * 100
    
    # Additional metrics
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    mse_log = np.mean((y_pred_log - y_true_log) ** 2)
    
    return {
        'smape': smape,
        'mae': mae,
        'rmse': rmse,
        'mse_log': mse_log
    }

# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================
def main():
    """Main training pipeline"""
    
    # Set seed
    set_seed(Config.SEED)
    
    # Create output directory
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    print("=" * 80)
    print("AMAZON ML CHALLENGE - TRANSFORMER PRICE PREDICTION")
    print("=" * 80)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # GPU Check - CRITICAL for fast training
    print("\n" + "=" * 80)
    print("DEVICE CHECK")
    print("=" * 80)
    
    if Config.REQUIRE_GPU and not Config.DEVICE.startswith('cuda'):
        print("❌ GPU REQUIRED BUT NOT AVAILABLE!")
        print("\nTraining on CPU would take 8-12 hours and is not recommended.")
        print("\nSOLUTIONS:")
        print("1. Fix GPU/CUDA installation:")
        print("   Run: python check_gpu.py")
        print("\n2. Use Google Colab (FREE GPU):")
        print("   - Go to: https://colab.research.google.com/")
        print("   - Upload train_transformer.py")
        print("   - Runtime > Change runtime type > GPU")
        print("\n3. Allow CPU training (NOT RECOMMENDED):")
        print("   - Edit train_transformer.py")
        print("   - Set: REQUIRE_GPU = False")
        print("\n" + "=" * 80)
        sys.exit(1)
    
    print(f" Device: {Config.DEVICE}")
    
    if Config.DEVICE.startswith('cuda'):
        gpu_id = Config.GPU_DEVICE if hasattr(Config, 'GPU_DEVICE') else 0
        print(f"\nGPU Information:")
        print(f"  GPU ID: {gpu_id}")
        print(f"  Name: {torch.cuda.get_device_name(gpu_id)}")
        print(f"  Memory: {torch.cuda.get_device_properties(gpu_id).total_memory / 1e9:.2f} GB")
        print(f"  CUDA Version: {torch.version.cuda}")
        
        # Verify GPU is working
        try:
            test_tensor = torch.randn(100, 100).to(Config.DEVICE)
            _ = test_tensor @ test_tensor.T
            print(f"  Status:  GPU {gpu_id} is working correctly")
        except Exception as e:
            print(f"  Status:  GPU test failed - {str(e)}")
            print("\nRun: python check_gpu.py for diagnostics")
            sys.exit(1)
    else:
        print("  WARNING: Training on CPU (will be very slow)")
    
    print(f"\nConfiguration:")
    print(f"  Model: {Config.MODEL_NAME}")
    print(f"  Batch Size: {Config.BATCH_SIZE}")
    print(f"  Gradient Accumulation: {Config.GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Effective Batch Size: {Config.BATCH_SIZE * Config.GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Epochs: {Config.EPOCHS}")
    print(f"  Learning Rate: {Config.LEARNING_RATE}")
    print(f"  Max Length: {Config.MAX_LENGTH} tokens")
    print(f"  Mixed Precision (FP16): {'ENABLED ✨' if Config.USE_FP16 else 'DISABLED'}")
    print(f"  Early Stopping Patience: {Config.EARLY_STOPPING_PATIENCE} epochs")
    print(f"\nEstimated Memory Usage (with FP16):")
    print(f"  Model: ~0.7 GB (FP16)")
    print(f"  Activations: ~6-8 GB (batch={Config.BATCH_SIZE}, length={Config.MAX_LENGTH}, FP16)")
    print(f"  Total Expected: ~7-9 GB / 48 GB available")
    print(f"  Speed Boost: ~1.5-2x faster than FP32")
    
    # Load data
    print("\n" + "=" * 80)
    print("1. LOADING DATA")
    print("=" * 80)
    
    print(f"Loading training data from: {Config.TRAIN_PATH}")
    train_df = pd.read_csv(Config.TRAIN_PATH, low_memory=False)
    
    print(f"Loading validation data from: {Config.VAL_PATH}")
    val_df = pd.read_csv(Config.VAL_PATH, low_memory=False)
    
    print(f"\nOriginal sizes - Train: {len(train_df)}, Val: {len(val_df)}")
    
    # Clean data
    train_df = train_df.dropna(subset=['log_price']).reset_index(drop=True)
    val_df = val_df.dropna(subset=['log_price']).reset_index(drop=True)
    
    print(f"After cleaning - Train: {len(train_df)}, Val: {len(val_df)}")
    
    # Load tokenizer
    print("\n" + "=" * 80)
    print("2. LOADING TOKENIZER")
    print("=" * 80)
    
    # Load tokenizer with use_fast=False to avoid conversion issues
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME, use_fast=False)
    print(f"Tokenizer loaded: {Config.MODEL_NAME}")
    print(f"Vocabulary size: {len(tokenizer)}")
    print(f"Using slow tokenizer (more compatible)")
    
    # Create datasets
    print("\n" + "=" * 80)
    print("3. CREATING DATASETS")
    print("=" * 80)
    
    train_dataset = ProductPriceDataset(train_df, tokenizer, Config.MAX_LENGTH)
    val_dataset = ProductPriceDataset(val_df, tokenizer, Config.MAX_LENGTH)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True if Config.DEVICE == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=True if Config.DEVICE == 'cuda' else False
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Create model
    print("\n" + "=" * 80)
    print("4. CREATING MODEL")
    print("=" * 80)
    
    model = TransformerPricePredictor(
        model_name=Config.MODEL_NAME,
        dropout=Config.DROPOUT
    )
    model = model.to(Config.DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1e9:.2f} GB (FP32)")
    
    # Setup optimizer and scheduler
    print("\n" + "=" * 80)
    print("5. SETTING UP OPTIMIZER")
    print("=" * 80)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    total_steps = len(train_loader) * Config.EPOCHS
    warmup_steps = int(total_steps * Config.WARMUP_RATIO)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    print(f"Total training steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")
    
    # Training loop with early stopping and FP16
    print("\n" + "=" * 80)
    print("6. TRAINING")
    print("=" * 80)
    print(f"Effective batch size: {Config.BATCH_SIZE * Config.GRADIENT_ACCUMULATION_STEPS}")
    print(f"Early stopping patience: {Config.EARLY_STOPPING_PATIENCE} epochs")
    print(f"Mixed precision (FP16): {'ENABLED' if Config.USE_FP16 else 'DISABLED'}")
    
    # Create GradScaler for FP16 training
    scaler = torch.cuda.amp.GradScaler() if Config.USE_FP16 else None
    if scaler:
        print(f"✨ Using automatic mixed precision (AMP) for faster training")
    
    best_smape = float('inf')
    patience_counter = 0
    history = []
    
    for epoch in range(1, Config.EPOCHS + 1):
        print(f"\n{'=' * 80}")
        print(f"EPOCH {epoch}/{Config.EPOCHS}")
        print(f"{'=' * 80}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, Config.DEVICE, epoch, scaler)
        
        # Validate
        val_preds_log, val_targets_log, val_loss = validate(model, val_loader, Config.DEVICE)
        
        # Compute metrics
        metrics = compute_metrics(val_targets_log, val_preds_log)
        
        # Print results
        print(f"\n{'=' * 80}")
        print(f"EPOCH {epoch} RESULTS")
        print(f"{'=' * 80}")
        print(f"  Train Loss:     {train_loss:.6f}")
        print(f"  Val Loss:       {val_loss:.6f}")
        print(f"  Val MSE (log):  {metrics['mse_log']:.6f}")
        print(f"  Val SMAPE:      {metrics['smape']:.4f}% ← PRIMARY METRIC")
        print(f"  Val MAE:        ${metrics['mae']:.2f}")
        print(f"  Val RMSE:       ${metrics['rmse']:.2f}")
        print(f"{'=' * 80}")
        
        # Save history
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_smape': metrics['smape'],
            'val_mae': metrics['mae'],
            'val_rmse': metrics['rmse']
        })
        
        # Save best model and check early stopping
        if metrics['smape'] < best_smape:
            best_smape = metrics['smape']
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'smape': metrics['smape'],
                'mae': metrics['mae'],
                'config': {
                    'model_name': Config.MODEL_NAME,
                    'max_length': Config.MAX_LENGTH,
                    'dropout': Config.DROPOUT
                }
            }
            
            torch.save(checkpoint, Config.MODEL_SAVE_PATH)
            print(f"\n✅ NEW BEST MODEL SAVED!")
            print(f"   Path: {Config.MODEL_SAVE_PATH}")
            print(f"   SMAPE: {metrics['smape']:.4f}%")
        else:
            patience_counter += 1
            print(f"\n No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}")
            
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"\n EARLY STOPPING at epoch {epoch}")
                print(f"   Best SMAPE: {best_smape:.4f}% (epoch {epoch - patience_counter})")
                break
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_path = os.path.join(Config.OUTPUT_DIR, 'training_history.csv')
    history_df.to_csv(history_path, index=False)
    
    # Final summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nBest Validation SMAPE: {best_smape:.4f}%")
    print(f"Baseline SMAPE: 79.74%")
    print(f"Improvement: {79.74 - best_smape:.2f} percentage points")
    print(f"\nModel saved to: {Config.MODEL_SAVE_PATH}")
    print(f"History saved to: {history_path}")
    print("\n" + "=" * 80)
    
    return model, history, best_smape

if __name__ == "__main__":
    try:
        model, history, best_smape = main()
        print("\n Training completed successfully!")
    except Exception as e:
        print(f"\n Error during training: {str(e)}")
        import traceback
        traceback.print_exc()

