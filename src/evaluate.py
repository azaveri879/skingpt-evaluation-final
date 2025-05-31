import sys
import os
from omegaconf import OmegaConf
from argparse import Namespace
import pandas as pd
from tqdm import tqdm
import torch
from PIL import Image
import glob
import re
import ast
from sklearn.metrics import classification_report
import difflib
from collections import Counter

# Add SkinGPT-4 directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
skingpt_dir = os.path.join(os.path.dirname(current_dir), 'SkinGPT-4')
sys.path.append(skingpt_dir)

# Import necessary modules
from skingpt4.common.config import Config
from skingpt4.common.registry import registry
from skingpt4.conversation.conversation import Chat, CONV_VISION

# --- PROMPT ENSEMBLE SETUP ---
# Define a fixed set of prompt templates (with a placeholder for class list)
prompt_templates = [
    "What is the most likely diagnosis for this skin lesion? Please answer with one of the following classes: {class_list}.",
    "Which of the following diagnoses best describes this skin lesion? {class_list}.",
    "Select the most appropriate diagnosis from this list: {class_list}.",
    "This is a dermatology image. What is the diagnosis? Choose from: {class_list}.",
    "What is the diagnosis? (Choose from: {class_list})",
    "What is the single best diagnosis for this image? {class_list}.",
    "What is the most probable skin condition? {class_list}.",
]

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
    
    def evaluate_image(self, image_path, prompt="What is the most likely diagnosis for this skin lesion? Please answer with one of the following classes: eczema, psoriasis, tinea, impetigo, urticaria, ... (list all classes for your dataset)."):
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
    
    def evaluate_image_with_prompts(self, image_path, prompts):
        predictions = []
        for prompt in prompts:
            pred = self.evaluate_image(image_path, prompt=prompt)
            predictions.append(pred)
        return predictions
    
    def evaluate_dataset(self, dataset_path, metadata_path, num_samples=None, image_column='image_id', label_column='dx', prompts=None, extract_class_fn=None):
        """Evaluate model on a dataset using prompt ensemble."""
        df = pd.read_csv(metadata_path)
        if num_samples:
            df = df.sample(n=num_samples, random_state=42)
        results = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            image_id = row[image_column].strip()
            if image_id.startswith('dataset/'):
                image_id = image_id[len('dataset/'):]
            if not (image_id.endswith('.jpg') or image_id.endswith('.png') or image_id.endswith('.jpeg')):
                image_id = f"{image_id}.jpg"
            image_path = os.path.join(dataset_path, image_id)
            if not os.path.exists(image_path):
                print("Files in directory:", glob.glob(os.path.dirname(image_path) + "/*"))
            all_preds = self.evaluate_image_with_prompts(image_path, prompts)
            # Extract class for each prediction
            all_classes = [extract_class_fn(pred) for pred in all_preds if pred]
            # Aggregate (majority vote)
            flat = [cls for cls in all_classes if cls]
            if flat:
                final_pred = Counter(flat).most_common(1)[0][0]
            else:
                final_pred = None
            results.append({
                'image': image_id,
                'true_label': row[label_column],
                'prediction': final_pred
            })
        return results

def extract_predicted_class(text):
    text = text.lower()
    for key, val in label_map.items():
        if key in text:
            return val
    # fallback to previous logic
    matches = [cls for cls in known_classes if cls in text]
    if matches:
        return matches[0]
    words = text.split()
    for word in words:
        close = difflib.get_close_matches(word, known_classes, n=1, cutoff=0.8)
        if close:
            return close[0]
    return None

def safe_extract_labels(x):
    if pd.isnull(x):
        return set()
    try:
        # Try to parse as a list
        labels = ast.literal_eval(x)
        if isinstance(labels, list):
            return set([l.lower() for l in labels])
        elif isinstance(labels, str):
            return set([labels.lower()])
        else:
            return set()
    except Exception:
        # If not a list, treat as a single label string
        return set([str(x).lower()])

def is_correct(row):
    return row['predicted_class'] in row['true_classes']

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
    
    # Build the set of all known classes from your true labels in the metadata
    ham_metadata = pd.read_csv(os.path.join(os.path.dirname(current_dir), "data/ham10000/HAM10000_metadata.csv"))
    all_classes = set()
    for labels in ham_metadata['dx']:
        try:
            for l in ast.literal_eval(labels):
                all_classes.add(l.lower())
        except:
            all_classes.add(str(labels).lower())
    known_classes = list(all_classes)
    
    # After loading/defining known_classes and label_map for HAM10000:
    ham_class_list = ', '.join(sorted(known_classes))
    ham_prompts = [p.format(class_list=ham_class_list) for p in prompt_templates]
    print("Evaluating HAM10000 dataset...")
    ham10000_results = evaluator.evaluate_dataset(
        os.path.join(os.path.dirname(current_dir), "data/ham10000/images"),
        os.path.join(os.path.dirname(current_dir), "data/ham10000/HAM10000_metadata.csv"),
        num_samples=100,
        image_column='image_id',
        label_column='dx',
        prompts=ham_prompts,
        extract_class_fn=extract_predicted_class
    )
    
    # Save HAM10000 results
    ham_df = pd.DataFrame(ham10000_results)
    
    print(ham_df['true_label'].head(10))
    print(ham_df['true_label'].unique())
    
    # For SCIN dataset, after defining scin_label_map and extracting all SCIN classes:
    scin_class_list = ', '.join(sorted(set(scin_label_map.values())))
    scin_prompts = [p.format(class_list=scin_class_list) for p in prompt_templates]
    print("\nEvaluating SCIN dataset...")
    scin_results = evaluator.evaluate_dataset(
        os.path.join(os.path.dirname(current_dir), "data/scin"),
        os.path.join(os.path.dirname(current_dir), "data/scin/scin_merged.csv"),
        num_samples=100,
        image_column='image_1_path',
        label_column='dermatologist_skin_condition_on_label_name',
        prompts=scin_prompts,
        extract_class_fn=extract_scin_class
    )
    
    # Save SCIN results
    scin_df = pd.DataFrame(scin_results)
    scin_df.to_csv(os.path.join(results_dir, "scin_results.csv"), index=False)
    print(f"Saved SCIN results to {os.path.join(results_dir, 'scin_results.csv')}")

    # Apply classification report
    df_valid = scin_df[scin_df['predicted_classes'].notnull() & (scin_df['true_classes'].apply(len) > 0)]
    
    # Create a binary classification report for each class
    all_classes = set()
    for labels in scin_df['true_classes']:
        all_classes.update(labels)
    all_classes = sorted(list(all_classes))
    
    # Initialize results dictionary
    results = {cls: {'true_pos': 0, 'false_pos': 0, 'false_neg': 0} for cls in all_classes}
    
    # Calculate metrics for each class
    for _, row in df_valid.iterrows():
        preds = set(row['predicted_classes'])
        trues = row['true_classes']
        
        for cls in all_classes:
            if cls in preds and cls in trues:
                results[cls]['true_pos'] += 1
            elif cls in preds and cls not in trues:
                results[cls]['false_pos'] += 1
            elif cls not in preds and cls in trues:
                results[cls]['false_neg'] += 1
    
    # Print detailed classification report
    print("\nDetailed Classification Report:")
    print(f"{'Class':<40} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-" * 82)
    
    total_true_pos = 0
    total_false_pos = 0
    total_false_neg = 0
    
    for cls in all_classes:
        stats = results[cls]
        true_pos = stats['true_pos']
        false_pos = stats['false_pos']
        false_neg = stats['false_neg']
        
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        support = true_pos + false_neg
        
        print(f"{cls:<40} {precision:>10.3f} {recall:>10.3f} {f1:>10.3f} {support:>10}")
        
        total_true_pos += true_pos
        total_false_pos += false_pos
        total_false_neg += false_neg
    
    # Print macro averages
    macro_precision = sum(p['true_pos'] / (p['true_pos'] + p['false_pos']) if (p['true_pos'] + p['false_pos']) > 0 else 0 
                         for p in results.values()) / len(results)
    macro_recall = sum(p['true_pos'] / (p['true_pos'] + p['false_neg']) if (p['true_pos'] + p['false_neg']) > 0 else 0 
                      for p in results.values()) / len(results)
    macro_f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall) if (macro_precision + macro_recall) > 0 else 0
    
    print("-" * 82)
    print(f"{'macro avg':<40} {macro_precision:>10.3f} {macro_recall:>10.3f} {macro_f1:>10.3f} {total_true_pos + total_false_neg:>10}")
    
    # Print micro averages
    micro_precision = total_true_pos / (total_true_pos + total_false_pos) if (total_true_pos + total_false_pos) > 0 else 0
    micro_recall = total_true_pos / (total_true_pos + total_false_neg) if (total_true_pos + total_false_neg) > 0 else 0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    
    print(f"{'micro avg':<40} {micro_precision:>10.3f} {micro_recall:>10.3f} {micro_f1:>10.3f} {total_true_pos + total_false_neg:>10}")

    # Analyze unmatched SCIN predictions
    unmatched = scin_df[scin_df['predicted_classes'].isnull()]
    if not unmatched.empty:
        print("\nAnalyzing unmatched SCIN predictions:")
        for _, row in unmatched.head(5).iterrows():
            print(f"\nPrediction: {row['prediction'][:100]}...")
            print(f"True label: {row['true_label']}")
            print("Found terms:", analyze_prediction(row['prediction']))

    # --- Automation: Suggest new label_map candidates ---
    def suggest_label_map_candidates(unmatched_preds, known_classes, label_map, top_n=30):
        # Common English words to filter out
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about',
            'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'shall', 'should', 'may', 'might', 'must', 'can', 'could', 'of',
            'from', 'up', 'down', 'out', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
            'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
            'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'this', 'that', 'these',
            'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'doing', 'would', 'should', 'could', 'ought', 'i', 'you', 'he', 'she',
            'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our',
            'their', 'mine', 'yours', 'hers', 'ours', 'theirs', 'who', 'whom', 'whose', 'which', 'what',
            'where', 'when', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
            'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
            's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'image', 'shows', 'appears',
            'appears to', 'appears to be', 'to be', 'on the', 'the image', 'the lesion', 'image shows',
            'shows a', 'the image shows', 'image shows a'
        }
        
        # Medical/dermatological terms to look for
        medical_terms = {
            'lesion', 'skin', 'rash', 'spot', 'mark', 'patch', 'growth', 'tumor', 'cancer', 'carcinoma',
            'melanoma', 'nevus', 'mole', 'keratosis', 'dermatitis', 'eczema', 'psoriasis', 'acne',
            'wart', 'cyst', 'ulcer', 'blister', 'nodule', 'papule', 'plaque', 'scale', 'crust',
            'erosion', 'fissure', 'scar', 'atrophy', 'lichenification', 'macule', 'vesicle', 'pustule',
            'wheal', 'telangiectasia', 'purpura', 'petechiae', 'ecchymosis', 'angioma', 'hemangioma',
            'fibroma', 'dermatofibroma', 'seborrheic', 'actinic', 'basal', 'squamous', 'malignant',
            'benign', 'precancerous', 'dysplastic', 'atypical', 'inflammatory', 'infectious', 'fungal',
            'bacterial', 'viral', 'allergic', 'autoimmune', 'congenital', 'hereditary', 'acquired'
        }
        
        # Tokenize predictions and count n-grams (1-3 words)
        from collections import Counter
        ngram_counts = Counter()
        for pred in unmatched_preds:
            tokens = pred.lower().split()
            # Look for medical terms and their context
            for i, token in enumerate(tokens):
                if token in medical_terms:
                    # Add the medical term
                    ngram_counts[token] += 1
                    # Add 2-word phrases containing medical terms
                    if i < len(tokens) - 1:
                        ngram = f"{token} {tokens[i+1]}"
                        if not all(word in stop_words for word in ngram.split()):
                            ngram_counts[ngram] += 1
                    # Add 3-word phrases containing medical terms
                    if i < len(tokens) - 2:
                        ngram = f"{token} {tokens[i+1]} {tokens[i+2]}"
                        if not all(word in stop_words for word in ngram.split()):
                            ngram_counts[ngram] += 1
        
        # Exclude ngrams already in label_map or known_classes
        exclude = set(label_map.keys()) | set(known_classes) | stop_words
        candidates = [(ngram, count) for ngram, count in ngram_counts.items() 
                     if ngram not in exclude and count > 1]
        candidates.sort(key=lambda x: -x[1])
        
        print("\nTop new mapping candidates from unmatched predictions:")
        print("(Focusing on medical/dermatological terms and their context)")
        for ngram, count in candidates[:top_n]:
            print(f"  '{ngram}': '',  # seen {count} times")
        print("\nCopy any relevant ones into your label_map!")
        
        # Also show some example predictions that weren't matched
        print("\nExample unmatched predictions:")
        for pred in unmatched_preds[:5]:
            print(f"  {pred[:100]}...")

    suggest_label_map_candidates(unmatched, known_classes, label_map)

if __name__ == "__main__":
    main() 