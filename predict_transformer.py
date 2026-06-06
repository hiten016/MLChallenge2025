"""
Amazon ML Challenge - Generate Test Predictions
Inference script for trained transformer model

Usage:
    python predict_transformer.py

Requirements:
    - Trained model file: best_model.pt
    - Test data: dataset/dataset/test.csv
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import warnings

warnings.filterwarnings('ignore')

class Config:
    """Inference configuration"""
    
    # Paths
    MODEL_PATH = 'best_model.pt'
    TEST_DATA_PATH = 'dataset/dataset/test.csv'
    OUTPUT_PATH = 'dataset/dataset/test_out.csv'
    
    # Model settings (should match training)
    MAX_LENGTH = 512  # Updated to match training
    BATCH_SIZE = 16  # Larger for inference (can handle more)
    
    # Device settings
    GPU_DEVICE = 1  # Use 2nd GPU (cuda:1) - same as training
    DEVICE = f'cuda:{GPU_DEVICE}' if torch.cuda.is_available() else 'cpu'

class ProductPriceDataset(Dataset):
    """Dataset for product price prediction"""
    
    def __init__(self, df, tokenizer, max_length=256, is_test=False):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
        
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
        
        return item

class TransformerPricePredictor(nn.Module):
    """Transformer-based price prediction model"""
    
    def __init__(self, model_name, dropout=0.2):
        super().__init__()
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        
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
    
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        cls_output = outputs.last_hidden_state[:, 0, :]
        log_price = self.regressor(cls_output)
        
        return log_price.squeeze(-1)

def load_model(model_path, device):
    """Load trained model from checkpoint"""
    
    print(f"Loading model from: {model_path}")
    
    # Load checkpoint (weights_only=False for PyTorch 2.6+ compatibility)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Get model config from checkpoint
    model_config = checkpoint.get('config', {})
    model_name = model_config.get('model_name', 'roberta-large')  # Updated default
    dropout = model_config.get('dropout', 0.2)
    
    print(f"Model: {model_name}")
    print(f"Best SMAPE: {checkpoint['smape']:.4f}%")
    
    # Create model
    model = TransformerPricePredictor(model_name=model_name, dropout=dropout)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, model_name

def predict_test_set(model, dataloader, device):
    """Generate predictions for test set"""
    
    predictions_log = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Generating predictions'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Predict log prices
            pred_log = model(input_ids, attention_mask)
            predictions_log.extend(pred_log.cpu().numpy())
    
    # Transform back to original scale
    predictions = np.expm1(np.array(predictions_log))
    
    # Ensure all prices are positive (minimum $0.01)
    predictions = np.maximum(predictions, 0.01)
    
    return predictions

def main():
    """Main inference pipeline"""
    
    print("=" * 80)
    print("AMAZON ML CHALLENGE - GENERATING TEST PREDICTIONS")
    print("=" * 80)
    
    print(f"\nDevice: {Config.DEVICE}")
    if Config.DEVICE.startswith('cuda'):
        gpu_id = Config.GPU_DEVICE if hasattr(Config, 'GPU_DEVICE') else 0
        print(f"GPU ID: {gpu_id}")
        print(f"GPU: {torch.cuda.get_device_name(gpu_id)}")
    
    # Check if model exists
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {Config.MODEL_PATH}\n"
            f"Please train the model first using: python train_transformer.py"
        )
    
    # Load model
    print("\n" + "=" * 80)
    print("1. LOADING MODEL")
    print("=" * 80)
    
    model, model_name = load_model(Config.MODEL_PATH, Config.DEVICE)
    
    # Load tokenizer
    print("\n" + "=" * 80)
    print("2. LOADING TOKENIZER")
    print("=" * 80)
    
    # Load tokenizer with use_fast=False to avoid conversion issues
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    print(f"Tokenizer loaded: {model_name}")
    print(f"Using slow tokenizer (more compatible)")
    
    # Load test data
    print("\n" + "=" * 80)
    print("3. LOADING TEST DATA")
    print("=" * 80)
    
    print(f"Reading: {Config.TEST_DATA_PATH}")
    
    # Read with robust error handling
    try:
        test_df = pd.read_csv(Config.TEST_DATA_PATH, low_memory=False)
    except Exception as e:
        print(f"  Standard CSV reading failed: {e}")
        print("Attempting to read with error handling...")
        test_df = pd.read_csv(
            Config.TEST_DATA_PATH,
            on_bad_lines='skip',
            engine='python',
            low_memory=False
        )
        print(f" Successfully read with error handling")
    
    print(f"Test samples: {len(test_df)}")
    
    # Verify required columns
    required_cols = ['sample_id', 'catalog_content', 'image_link']
    missing_cols = [col for col in required_cols if col not in test_df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in test data: {missing_cols}")
    
    # Create dataset
    print("\n" + "=" * 80)
    print("4. CREATING DATASET")
    print("=" * 80)
    
    test_dataset = ProductPriceDataset(
        test_df, 
        tokenizer, 
        max_length=Config.MAX_LENGTH,
        is_test=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if Config.DEVICE == 'cuda' else False
    )
    
    print(f"Test batches: {len(test_loader)}")
    
    # Generate predictions
    print("\n" + "=" * 80)
    print("5. GENERATING PREDICTIONS")
    print("=" * 80)
    
    predictions = predict_test_set(model, test_loader, Config.DEVICE)
    
    # Create submission file
    print("\n" + "=" * 80)
    print("6. CREATING SUBMISSION FILE")
    print("=" * 80)
    
    submission = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': predictions
    })
    
    # Verify submission format
    print("\nSubmission Statistics:")
    print(f"  Total samples: {len(submission)}")
    print(f"  Price range: ${submission['price'].min():.2f} - ${submission['price'].max():.2f}")
    print(f"  Mean price: ${submission['price'].mean():.2f}")
    print(f"  Median price: ${submission['price'].median():.2f}")
    
    print("\nFirst 10 predictions:")
    print(submission.head(10).to_string(index=False))
    
    # Check for any issues
    if submission['sample_id'].duplicated().any():
        print("\n  WARNING: Duplicate sample_ids found!")
    
    if submission.isnull().any().any():
        print("\n  WARNING: NaN values found!")
    
    if (submission['price'] <= 0).any():
        print("\n  WARNING: Non-positive prices found!")
    
    # Save submission
    submission.to_csv(Config.OUTPUT_PATH, index=False)
    
    print("\n" + "=" * 80)
    print("PREDICTIONS COMPLETE!")
    print("=" * 80)
    print(f"\n Submission saved to: {Config.OUTPUT_PATH}")
    print(f"\nYou can now upload this file to the competition portal.")
    print("=" * 80)
    
    return submission

if __name__ == "__main__":
    try:
        submission = main()
        print("\n Prediction completed successfully!")
    except Exception as e:
        print(f"\n Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()

