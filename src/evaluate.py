import os
import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from skingpt4.common.config import Config
from skingpt4.common.registry import registry
from skingpt4.conversation.conversation import Chat, CONV_VISION

class SkinGPTEvaluator:
    def __init__(self, config_path, device='cuda:0'):
        """Initialize SkinGPT evaluator."""
        self.device = device
        self.cfg = Config(config_path)
        
        # Initialize model
        model_config = self.cfg.model_cfg
        model_config.device_8bit = 0
        model_cls = registry.get_model_class(model_config.arch)
        self.model = model_cls.from_config(model_config).to(device)
        
        # Initialize chat
        vis_processor_cfg = self.cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        self.chat = Chat(self.model, vis_processor, device=device)
    
    def evaluate_image(self, image_path, prompt="What is the diagnosis for this skin lesion?"):
        """Evaluate a single image."""
        try:
            # Load and preprocess image
            image = Image.open(image_path)
            
            # Initialize chat state
            chat_state = CONV_VISION.copy()
            img_list = []
            
            # Upload image
            self.chat.upload_img(image, chat_state, img_list)
            
            # Ask for diagnosis
            self.chat.ask(prompt, chat_state)
            
            # Get response
            response = self.chat.answer(
                conv=chat_state,
                img_list=img_list,
                num_beams=1,
                temperature=1.0,
                max_new_tokens=300,
                max_length=2000
            )[0]
            
            return response
        except Exception as e:
            print(f"Error evaluating image {image_path}: {e}")
            return None
    
    def evaluate_dataset(self, dataset_path, metadata_path, num_samples=None):
        """Evaluate model on a dataset."""
        # Load metadata
        df = pd.read_csv(metadata_path)
        if num_samples:
            df = df.sample(n=num_samples, random_state=42)
        
        results = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            image_path = os.path.join(dataset_path, row['image'])
            prediction = self.evaluate_image(image_path)
            
            results.append({
                'image': row['image'],
                'true_label': row['dx'],
                'prediction': prediction
            })
        
        return results
    
    def analyze_results(self, results, save_dir):
        """Analyze and save evaluation results."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Convert results to DataFrame
        df = pd.DataFrame(results)
        
        # Save raw results
        df.to_csv(os.path.join(save_dir, 'raw_results.csv'), index=False)
        
        # Generate summary statistics
        summary = {
            'total_samples': len(df),
            'unique_true_labels': df['true_label'].nunique(),
            'unique_predictions': df['prediction'].nunique()
        }
        
        # Save summary
        with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)
        
        # Generate confusion matrix
        # Note: This is a simplified version. You might need to implement
        # more sophisticated matching between predictions and true labels
        cm = confusion_matrix(df['true_label'], df['prediction'])
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
        plt.close()
        
        # Generate classification report
        report = classification_report(
            df['true_label'],
            df['prediction'],
            output_dict=True
        )
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(os.path.join(save_dir, 'classification_report.csv'))
        
        return summary, report_df

def main():
    # Example usage
    evaluator = SkinGPTEvaluator('eval_configs/skingpt4_eval_llama2_13bchat.yaml')
    
    # Evaluate HAM10000
    ham10000_results = evaluator.evaluate_dataset(
        '../data/ham10000',
        '../data/ham10000/HAM10000_metadata.csv',
        num_samples=100
    )
    evaluator.analyze_results(ham10000_results, '../results/ham10000')
    
    # Evaluate SCIN
    scin_results = evaluator.evaluate_dataset(
        '../data/scin',
        '../data/scin/SCIN_metadata.csv',
        num_samples=100
    )
    evaluator.analyze_results(scin_results, '../results/scin')

if __name__ == '__main__':
    main() 