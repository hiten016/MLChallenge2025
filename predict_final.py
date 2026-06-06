"""
Amazon ML Challenge - Generate Multimodal Test Predictions
Inference script for trained RoBERTa + SigLIP model

Usage:
    python predict_transformer.py

Requirements:
    - Trained model file: best_model.pt
    - Test data: dataset/dataset/test.csv
    - Images directory downloaded locally
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoProcessor, AutoModel, AutoConfig
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

class Config:
    """Inference configuration"""
    
    # Paths
    MODEL_PATH = 'best_model.pt'
    TEST_DATA_PATH = 'dataset/dataset/test.csv'
    OUTPUT_PATH = 'dataset/dataset/test_out.csv'
    IMAGE_DIR = 'dataset/dataset/images'  # Path where images are saved locally
    
    # Model settings (should match training)
    TEXT_MODEL_NAME = 'roberta-large' 
    VISION_MODEL_NAME = 'google/siglip-base-patch16-224'
    MAX_LENGTH = 512  
    BATCH_SIZE = 16  
    
    # Device settings
    GPU_DEVICE = 1  # Use 2nd GPU (cuda:1)
    DEVICE = f'cuda:{GPU_DEVICE}' if torch.cuda.is_available() else 'cpu'

class MultimodalProductDataset(Dataset):
    """Dataset for multimodal (Text + Image) product price prediction"""
    
    def __init__(self, df, tokenizer, processor, max_length=512, image_dir=None):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length
        self.image_dir = image_dir
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Process Text
        text = str(row['catalog_content'])
        text_encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 2. Process Image safely
        # Fallback to a blank image if file is missing or corrupted
        image_input = None
        try:
            # Assumes image file name can be extracted from image_link or a column
            img_filename = str(row['image_link']).split('/')[-1]
            img_path = os.path.join(self.image_dir, img_filename) if self.image_dir else img_filename
            
            if os.path.exists(img_path):
                image = Image.open(img_path).convert('RGB')
            else:
                image = Image.new('RGB', (224, 224), color='white')
        except Exception:
            image = Image.new('RGB', (224, 224), color='white')
            
        image_encoding = self.processor(images=image, return_tensors="pt")
        
        item = {
            'input_ids': text_encoding['input_ids'].flatten(),
            'attention_mask': text_encoding['attention_mask'].flatten(),
            'pixel_values': image_encoding['pixel_values'].squeeze(0)
        }
        
        return item

class MultimodalPricePredictor(nn.Module):
    """Combined RoBERTa + SigLIP price prediction model"""
    
    def __init__(self, text_model_name, vision_model_name, dropout=0.2):
        super().__init__()
        
        # Encoders
        self.text_config = AutoConfig.from_pretrained(text_model_name)
        self.text_model = AutoModel.from_pretrained(text_model_name)
        
        self.vision_config = AutoConfig.from_pretrained(vision_model_name)
        self.vision_model = AutoModel.from_pretrained(vision_model_name)
        
        # Feature dimensions
        text_hidden_size = self.text_config.hidden_size      # 1024 for roberta-large
        vision_hidden_size = self.vision_config.vision_config.hidden_size  # 768 for siglip-base
        combined_size = text_hidden_size + vision_hidden_size
        
        # Multimodal Regressor
        self.regressor = nn.Sequential(
            nn.Linear(combined_size, 512),
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
    
    def forward(self, input_ids, attention_mask, pixel_values):
        # Extract Text Embeddings (CLS token)
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]
        
        # Extract Vision Embeddings (Pooled output representation)
        vision_outputs = self.vision_model.vision_model(pixel_values=pixel_values)
        vision_features = vision_outputs.pooler_output
        
        # Fuse via Concatenation
        fused_features = torch.cat((text_features, vision_features), dim=-1)
        
        # Predict Log Price
        log_price = self.regressor(fused_features)
        
        return log_price.squeeze(-1)

def load_model(model_path, device):
    """Load trained multimodal model from checkpoint"""
    print(f"Loading model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model_config = checkpoint.get('config', {})
    text_model_name = model_config.get('text_model_name', Config.TEXT_MODEL_NAME)
    vision_model_name = model_config.get('vision_model_name', Config.VISION_MODEL_NAME)
    dropout = model_config.get('dropout', 0.2)
    
    print(f"Text Encoder  : {text_model_name}")
    print(f"Vision Encoder: {vision_model_name}")
    if 'smape' in checkpoint:
        print(f"Best Training SMAPE: {checkpoint['smape']:.4f}%")
    
    # Initialize model with weights matching checkpoint strategy
    model = MultimodalPricePredictor(
        text_model_name=text_model_name, 
        vision_model_name=vision_model_name, 
        dropout=dropout
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, text_model_name, vision_model_name

def predict_test_set(model, dataloader, device):
    """Generate predictions for test set across text and vision features"""
    predictions_log = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Generating predictions'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            
            pred_log = model(input_ids, attention_mask, pixel_values)
            predictions_log.extend(pred_log.cpu().numpy())
    
    predictions = np.expm1(np.array(predictions_log))
    predictions = np.maximum(predictions, 0.01)  # Ensure minimum pricing bounds
    
    return predictions

def main():
    print("=" * 80)
    print("AMAZON ML CHALLENGE - MULTIMODAL TEST PREDICTIONS (SIGLIP)")
    print("=" * 80)
    
    print(f"\nDevice: {Config.DEVICE}")
    if Config.DEVICE.startswith('cuda'):
        gpu_id = Config.GPU_DEVICE if hasattr(Config, 'GPU_DEVICE') else 0
        print(f"GPU: {torch.cuda.get_device_name(gpu_id)}")
    
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(f"Model checkpoint not found: {Config.MODEL_PATH}")
    
    # 1. Load Multimodal Model
    print("\n" + "=" * 80)
    print("1. LOADING MULTIMODAL MODEL")
    print("=" * 80)
    model, text_name, vision_name = load_model(Config.MODEL_PATH, Config.DEVICE)
    
    # 2. Load Processors & Tokenizers
    print("\n" + "=" * 80)
    print("2. LOADING TOKENIZER & SIGLIP PROCESSOR")
    print("=" * 80)
    tokenizer = AutoTokenizer.from_pretrained(text_name, use_fast=False)
    processor = AutoProcessor.from_pretrained(vision_name)
    print("Processors fully loaded.")
    
    # 3. Load Test Data
    print("\n" + "=" * 80)
    print("3. LOADING TEST DATA")
    print("=" * 80)
    try:
        test_df = pd.read_csv(Config.TEST_DATA_PATH, low_memory=False)
    except Exception as e:
        test_df = pd.read_csv(Config.TEST_DATA_PATH, on_bad_lines='skip', engine='python', low_memory=False)
    print(f"Test samples: {len(test_df)}")
    
    # 4. Create Multimodal Dataset
    print("\n" + "=" * 80)
    print("4. CREATING DATASET")
    print("=" * 80)
    test_dataset = MultimodalProductDataset(
        test_df, 
        tokenizer, 
        processor,
        max_length=Config.MAX_LENGTH,
        image_dir=Config.IMAGE_DIR
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,  # Increased for faster image disk I/O handling
        pin_memory=True if Config.DEVICE.startswith('cuda') else False
    )
    
    # 5. Predictions Generation
    print("\n" + "=" * 80)
    print("5. GENERATING MULTIMODAL PREDICTIONS")
    print("=" * 80)
    predictions = predict_test_set(model, test_loader, Config.DEVICE)
    
    # 6. Build Submission
    print("\n" + "=" * 80)
    print("6. CREATING SUBMISSION FILE")
    print("=" * 80)
    submission = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': predictions
    })
    
    print("\nSubmission Statistics:")
    print(f"  Price range: ${submission['price'].min():.2f} - ${submission['price'].max():.2f}")
    print(f"  Mean price: ${submission['price'].mean():.2f}")
    
    submission.to_csv(Config.OUTPUT_PATH, index=False)
    print(f"\nSubmission saved to: {Config.OUTPUT_PATH}")
    print("=" * 80)
    return submission

if __name__ == "__main__":
    try:
        submission = main()
        print("\nPrediction completed successfully!")
    except Exception as e:
        print(f"\nError during prediction: {str(e)}")
        import traceback
        traceback.print_exc()