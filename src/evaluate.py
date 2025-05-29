import sys
import os
from omegaconf import OmegaConf
from argparse import Namespace
import pandas as pd
from tqdm import tqdm
import torch
from PIL import Image

# Add SkinGPT-4 directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
skingpt_dir = os.path.join(os.path.dirname(current_dir), 'SkinGPT-4')
sys.path.append(skingpt_dir)

# Import necessary modules
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
    
    def evaluate_dataset(self, dataset_path, metadata_path, num_samples=None, image_column='image_id', label_column='dx'):
        """Evaluate model on a dataset."""
        # Load metadata
        df = pd.read_csv(metadata_path)
        if num_samples:
            df = df.sample(n=num_samples, random_state=42)
        
        results = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            # Add .jpg extension if not present
            image_id = row[image_column]
            if not image_id.endswith('.jpg'):
                image_id = f"{image_id}.jpg"
            
            image_path = os.path.join(dataset_path, image_id)
            prediction = self.evaluate_image(image_path)
            
            results.append({
                'image': image_id,
                'true_label': row[label_column],
                'prediction': prediction
            })
        
        return results

def main():
    # Create configuration
    config = {
        'model': {
            'arch': 'skin_gpt4',
            'model_type': 'pretrain_llama2_13bchat',
            'freeze_vit': True,
            'freeze_qformer': True,
            'max_txt_len': 160,
            'end_sym': "###",
            'prompt_path': os.path.join(skingpt_dir, "prompts/alignment_skin.txt"),
            'prompt_template': '###Human: {} ###Assistant: ',
            'ckpt': os.path.join(skingpt_dir, "weights/skingpt4_llama2_13bchat_base_pretrain_stage2.pth"),
            'llm_model': "meta-llama/Llama-2-13b-chat-hf",  # Use Hugging Face model ID
            'low_resource': False,  # Don't use 8-bit quantization
            'device_8bit': 0,  # Not used when low_resource is False
            'use_fast': False,  # Use slow tokenizer for compatibility
            'torch_dtype': 'float16'  # Use float16 for efficiency
        },
        'datasets': {
            'cc_sbu_align': {
                'vis_processor': {
                    'train': {
                        'name': "blip2_image_eval",
                        'image_size': 224
                    }
                },
                'text_processor': {
                    'train': {
                        'name': "blip_caption"
                    }
                }
            }
        },
        'run': {
            'task': 'image_text_pretrain'
        }
    }
    
    # Save configuration to file
    config_path = os.path.join(skingpt_dir, "eval_configs/skingpt4_eval_llama2_13bchat_local.yaml")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    OmegaConf.save(config, config_path)
    
    # Create args object
    args = Namespace(
        cfg_path=config_path,
        options=[]
    )
    
    # Initialize evaluator
    evaluator = SkinGPTEvaluator(args)
    
    # Create results directory
    results_dir = os.path.join(os.path.dirname(current_dir), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Evaluate HAM10000 dataset
    print("Evaluating HAM10000 dataset...")
    ham10000_results = evaluator.evaluate_dataset(
        os.path.join(os.path.dirname(current_dir), "data/ham10000/images"),
        os.path.join(os.path.dirname(current_dir), "data/ham10000/HAM10000_metadata.csv"),
        num_samples=100,
        image_column='image_id',
        label_column='dx'
    )
    
    # Save HAM10000 results
    ham_df = pd.DataFrame(ham10000_results)
    ham_df.to_csv(os.path.join(results_dir, "ham10000_results.csv"), index=False)
    print(f"Saved HAM10000 results to {os.path.join(results_dir, 'ham10000_results.csv')}")
    
    # Evaluate SCIN dataset
    print("\nEvaluating SCIN dataset...")
    scin_results = evaluator.evaluate_dataset(
        os.path.join(os.path.dirname(current_dir), "data/scin"),
        os.path.join(os.path.dirname(current_dir), "data/scin/scin_merged.csv"),
        num_samples=100,
        image_column='image_1_path',
        label_column='dermatologist_skin_condition_on_label_name'
    )
    
    # Save SCIN results
    scin_df = pd.DataFrame(scin_results)
    scin_df.to_csv(os.path.join(results_dir, "scin_results.csv"), index=False)
    print(f"Saved SCIN results to {os.path.join(results_dir, 'scin_results.csv')}")

if __name__ == "__main__":
    main() 