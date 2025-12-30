"""
Configuration file for fetal ultrasound biometry landmark detection
"""
import os
from pathlib import Path

# Project Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "images"
CSV_PATH = BASE_DIR / "role_challenge_dataset_ground_truth.csv"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
LOG_DIR = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"

# Create directories
CHECKPOINT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Data Configuration
IMAGE_SIZE = (512, 512)  # Standard size for ultrasound images
HEATMAP_SIZE = (128, 128)  # Size for heatmap generation
NUM_LANDMARKS = 4  # 2 for BPD, 2 for OFD
HEATMAP_SIGMA = 2.0  # Gaussian sigma for heatmap generation

# Training Configuration
BATCH_SIZE = 16
NUM_EPOCHS = 150
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 4
PIN_MEMORY = True

# Cross-validation
K_FOLDS = 5
TRAIN_VAL_SPLIT = 0.8
RANDOM_SEED = 42

# Augmentation Configuration
AUG_ROTATION_LIMIT = 20  # degrees
AUG_SCALE_LIMIT = 0.15  # ±15% scale
AUG_SHIFT_LIMIT = 0.1  # ±10% shift
AUG_BRIGHTNESS_LIMIT = 0.2
AUG_CONTRAST_LIMIT = 0.2
AUG_GAMMA_LIMIT = (80, 120)
AUG_PROBABILITY = 0.5

# Preprocessing Configuration
NORMALIZE_MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
NORMALIZE_STD = [0.229, 0.224, 0.225]   # ImageNet std
USE_CLAHE = True
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = (8, 8)

# Model Configuration - Landmark Detection
LANDMARK_MODELS = {
    'heatmap': {
        'backbone': 'resnet50',
        'pretrained': True,
        'num_landmarks': NUM_LANDMARKS,
        'heatmap_size': HEATMAP_SIZE,
    },
    'coordinate': {
        'backbone': 'efficientnet_b3',
        'pretrained': True,
        'dropout': 0.3,
    },
    'attention_pyramid': {
        'backbone': 'resnet34',
        'pretrained': True,
        'fpn_channels': 256,
        'attention_type': 'spatial',
    },
    'transformer': {
        'model_name': 'vit_base_patch16_224',
        'pretrained': True,
        'num_landmarks': NUM_LANDMARKS,
        'embed_dim': 768,
    }
}

# Model Configuration - Segmentation
SEGMENTATION_MODELS = {
    'unet': {
        'encoder': 'resnet34',
        'encoder_weights': 'imagenet',
        'in_channels': 3,
        'classes': 1,
    },
    'attention_unet': {
        'encoder': 'resnet50',
        'encoder_weights': 'imagenet',
        'in_channels': 3,
        'classes': 1,
        'attention_type': 'scse',
    },
    'deeplabv3plus': {
        'encoder': 'resnet50',
        'encoder_weights': 'imagenet',
        'in_channels': 3,
        'classes': 1,
    }
}

# Loss Configuration
LOSS_CONFIG = {
    'landmark': {
        'type': 'adaptive_wing',  # Options: 'mse', 'smooth_l1', 'wing', 'adaptive_wing'
        'alpha': 2.1,
        'omega': 14,
        'epsilon': 1,
        'theta': 0.5,
        'use_geometric_constraint': True,
        'geometric_weight': 0.1,
    },
    'segmentation': {
        'type': 'combined',  # Options: 'dice', 'focal', 'bce', 'combined'
        'dice_weight': 0.5,
        'focal_weight': 0.3,
        'boundary_weight': 0.2,
        'focal_alpha': 0.25,
        'focal_gamma': 2.0,
    }
}

# Optimizer Configuration
OPTIMIZER_CONFIG = {
    'type': 'adamw',  # Options: 'adam', 'adamw', 'sgd'
    'lr': LEARNING_RATE,
    'weight_decay': WEIGHT_DECAY,
    'betas': (0.9, 0.999),
}

# Scheduler Configuration
SCHEDULER_CONFIG = {
    'type': 'cosine_warmup',  # Options: 'cosine', 'cosine_warmup', 'reduce_on_plateau'
    'warmup_epochs': 10,
    'min_lr': 1e-7,
    'patience': 15,  # for reduce_on_plateau
}

# Evaluation Metrics Thresholds
EVAL_THRESHOLDS = {
    'sdr': [2.0, 2.5, 3.0, 4.0],  # mm - Successful Detection Rate thresholds
    'acceptable_error': 3.0,  # mm - Clinical acceptable error
}

# Ellipse Fitting Configuration (for segmentation approach)
ELLIPSE_CONFIG = {
    'min_area': 5000,  # Minimum cranium area in pixels
    'max_area': 150000,  # Maximum cranium area in pixels
    'use_ransac': True,
    'ransac_iterations': 1000,
    'ransac_threshold': 2.0,
}

# Visualization Configuration
VIZ_CONFIG = {
    'landmark_color': (0, 255, 0),  # Green
    'bpd_color': (255, 0, 0),  # Red
    'ofd_color': (0, 0, 255),  # Blue
    'line_thickness': 2,
    'point_radius': 5,
    'show_heatmap': True,
    'heatmap_alpha': 0.5,
}

# Hardware Configuration
DEVICE = 'cuda'  # Will be set to 'cpu' if CUDA not available
MIXED_PRECISION = True  # Use automatic mixed precision training
CUDNN_BENCHMARK = True  # Optimize CUDNN performance

# Logging Configuration
LOG_INTERVAL = 10  # Log every N batches
SAVE_INTERVAL = 5  # Save checkpoint every N epochs
VISUALIZE_INTERVAL = 25  # Save visualization every N batches

# Experiment Tracking
USE_WANDB = False  # Set to True if using Weights & Biases
WANDB_PROJECT = "fetal-ultrasound-biometry"
WANDB_ENTITY = None  # Your wandb username/team

# Clinical Context
CLINICAL_INFO = {
    'bpd_name': 'Biparietal Diameter',
    'ofd_name': 'Occipitofrontal Diameter',
    'description': 'Fetal head biometry measurements for gestational age estimation',
    'typical_bpd_range': (20, 100),  # mm, typical range across gestations
    'typical_ofd_range': (25, 130),  # mm
}
