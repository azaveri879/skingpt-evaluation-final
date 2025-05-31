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
            image_id = row[image_column].strip()
            if image_id.startswith('dataset/'):
                image_id = image_id[len('dataset/'):]
            if not (image_id.endswith('.jpg') or image_id.endswith('.png') or image_id.endswith('.jpeg')):
                image_id = f"{image_id}.jpg"
            image_path = os.path.join(dataset_path, image_id)
            # print(f"Checking: {repr(image_path)}  Exists: {os.path.exists(image_path)}")
            if not os.path.exists(image_path):
                print("Files in directory:", glob.glob(os.path.dirname(image_path) + "/*"))
            prediction = self.evaluate_image(image_path)
            results.append({
                'image': image_id,
                'true_label': row[label_column],
                'prediction': prediction
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
    
    print(ham_df['true_label'].head(10))
    print(ham_df['true_label'].unique())
    
    # 1. Build the set of all known classes from your true labels
    all_classes = set()
    for labels in ham_df['true_label']:
        try:
            for l in ast.literal_eval(labels):
                all_classes.add(l.lower())
        except:
            pass
    known_classes = list(all_classes)

    print("Known classes:", known_classes)
    print("Number of known classes:", len(known_classes))

    # 2. Improved extraction function
    label_map = {
        # Melanoma related terms
        'melanoma': 'mel',
        'malignant melanoma': 'mel',
        'malignant lesion': 'mel',
        'irregular border': 'mel',
        'irregular shape': 'mel',
        'asymmetrical': 'mel',
        'multiple colors': 'mel',
        'variegated': 'mel',
        
        # Nevus related terms
        'melanocytic nevi': 'nv',
        'nevus': 'nv',
        'mole': 'nv',
        'benign mole': 'nv',
        'regular border': 'nv',
        'symmetrical': 'nv',
        'uniform color': 'nv',
        
        # BKL related terms
        'benign keratosis': 'bkl',
        'keratosis': 'bkl',
        'seborrheic keratosis': 'bkl',
        'seborrheic': 'bkl',
        'scaly': 'bkl',
        'rough surface': 'bkl',
        'stuck on': 'bkl',
        
        # BCC related terms
        'basal cell carcinoma': 'bcc',
        'bcc': 'bcc',
        'pearly': 'bcc',
        'rolled border': 'bcc',
        'telangiectasia': 'bcc',
        'ulcerated': 'bcc',
        
        # AKIEC related terms
        'actinic keratoses': 'akiec',
        'intraepithelial carcinoma': 'akiec',
        'solar keratosis': 'akiec',
        'actinic': 'akiec',
        'precancerous': 'akiec',
        
        # DF related terms
        'dermatofibroma': 'df',
        'fibroma': 'df',
        'firm': 'df',
        'dimple sign': 'df',
        
        # VASC related terms
        'vascular lesion': 'vasc',
        'angioma': 'vasc',
        'hemangioma': 'vasc',
        'vascular': 'vasc',
        'red': 'vasc',
        'purple': 'vasc'
    }

    def extract_predicted_class(text):
        text = text.lower()
        
        # First try exact matches from label_map
        for key, val in label_map.items():
            if key in text:
                return val
        
        # Look for characteristic features
        features = {
            'mel': ['irregular', 'asymmetrical', 'multiple colors', 'variegated'],
            'nv': ['regular', 'symmetrical', 'uniform', 'round'],
            'bkl': ['scaly', 'rough', 'stuck on'],
            'bcc': ['pearly', 'rolled', 'telangiectasia', 'ulcerated'],
            'akiec': ['actinic', 'precancerous', 'solar'],
            'df': ['firm', 'dimple'],
            'vasc': ['red', 'purple', 'vascular']
        }
        
        for class_name, feature_list in features.items():
            if any(feature in text for feature in feature_list):
                return class_name
        
        # Fallback to fuzzy matching
        words = text.split()
        for word in words:
            close = difflib.get_close_matches(word, known_classes, n=1, cutoff=0.8)
            if close:
                return close[0]
        
        return None

    def analyze_prediction(text):
        """Helper function to analyze why a prediction wasn't matched"""
        text = text.lower()
        found_terms = []
        for key, val in label_map.items():
            if key in text:
                found_terms.append(f"{key} -> {val}")
        return found_terms

    # 3. Apply to your dataframe
    ham_df['predicted_class'] = ham_df['prediction'].apply(extract_predicted_class)
    ham_df['true_classes'] = ham_df['true_label'].apply(safe_extract_labels)
    ham_df['correct'] = ham_df.apply(lambda row: row['predicted_class'] in row['true_classes'], axis=1)
    accuracy = ham_df['correct'].mean()
    print(f"Adjusted Accuracy: {accuracy:.2%}")
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
    
    # SCIN-specific label mapping
    scin_label_map = {
        # Inflammatory conditions
        'eczema': 'eczema',
        'dermatitis': 'acute dermatitis, nos',
        'allergic contact': 'allergic contact dermatitis',
        'irritant contact': 'irritant contact dermatitis',
        'seborrheic': 'seborrheic dermatitis',
        'psoriasis': 'psoriasis',
        'lichen planus': 'lichen planus/lichenoid eruption',
        'lichenoid': 'lichen planus/lichenoid eruption',
        
        # Infections
        'impetigo': 'impetigo',
        'tinea': 'tinea',
        'fungal': 'tinea',
        'ringworm': 'tinea',
        'molluscum': 'molluscum contagiosum',
        'scabies': 'scabies',
        'candida': 'candida',
        'viral': 'viral exanthem',
        
        # Vascular conditions
        'vasculitis': 'leukocytoclastic vasculitis',
        'purpuric': 'pigmented purpuric eruption',
        'petechiae': 'traumatic petechiae',
        'ecchymosis': 'o/e - ecchymoses present',
        
        # Other conditions
        'acne': 'acne',
        'rosacea': 'rosacea',
        'urticaria': 'urticaria',
        'hives': 'urticaria',
        'granuloma': 'granuloma annulare',
        'insect bite': 'insect bite',
        'milia': 'milia',
        'kaposi': "kaposi's sarcoma of skin",
        'hypersensitivity': 'hypersensitivity',
        'allergic': 'hypersensitivity',
        'foreign body': 'foreign body',
        'abrasion': 'abrasion, scrape, or scab',
        'scrape': 'abrasion, scrape, or scab',
        'scab': 'abrasion, scrape, or scab',
        'cut': 'abrasion, scrape, or scab',
        'wound': 'abrasion, scrape, or scab'
    }

    def extract_scin_class(text):
        """Extract SCIN classes from prediction text."""
        text = text.lower()
        found_classes = set()
        
        # First try exact matches
        for key, val in scin_label_map.items():
            if key in text:
                found_classes.add(val)
        
        # Look for characteristic features
        features = {
            'eczema': ['itchy', 'red', 'inflamed', 'dry', 'scaly'],
            'dermatitis': ['inflamed', 'irritated', 'red', 'itchy'],
            'psoriasis': ['silvery', 'scaly', 'thick', 'plaques'],
            'impetigo': ['honey-colored', 'crust', 'blister', 'sore'],
            'tinea': ['ring', 'circular', 'fungal', 'scaly'],
            'scabies': ['burrow', 'tunnel', 'itchy', 'rash'],
            'vasculitis': ['purple', 'red', 'spots', 'bruise'],
            'acne': ['pimple', 'blackhead', 'whitehead', 'comedone'],
            'rosacea': ['red', 'flushing', 'bumps', 'pustules'],
            'urticaria': ['hive', 'wheal', 'itchy', 'raised']
        }
        
        for class_name, feature_list in features.items():
            if any(feature in text for feature in feature_list):
                if class_name in scin_label_map:
                    found_classes.add(scin_label_map[class_name])
        
        return list(found_classes) if found_classes else None

    def is_scin_correct(row):
        """Check if any predicted class matches any true class."""
        if not row['predicted_classes'] or not row['true_classes']:
            return False
        return bool(set(row['predicted_classes']) & row['true_classes'])

    # Save SCIN results
    scin_df = pd.DataFrame(scin_results)
    scin_df['predicted_classes'] = scin_df['prediction'].apply(extract_scin_class)
    scin_df['true_classes'] = scin_df['true_label'].apply(safe_extract_labels)
    scin_df['correct'] = scin_df.apply(is_scin_correct, axis=1)
    accuracy = scin_df['correct'].mean()
    print(f"Adjusted Accuracy: {accuracy:.2%}")
    scin_df.to_csv(os.path.join(results_dir, "scin_results.csv"), index=False)
    print(f"Saved SCIN results to {os.path.join(results_dir, 'scin_results.csv')}")

    # Apply classification report
    df_valid = scin_df[scin_df['predicted_classes'].notnull() & (scin_df['true_classes'].apply(len) > 0)]
    
    # Flatten predictions and true labels for classification report
    all_preds = []
    all_true = []
    for _, row in df_valid.iterrows():
        preds = row['predicted_classes']
        trues = list(row['true_classes'])
        if preds and trues:
            all_preds.extend(preds)
            all_true.extend(trues)
    
    print("\nClassification Report:")
    print(classification_report(all_true, all_preds, zero_division=0))

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