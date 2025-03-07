# Brain Tumor Detection using Deep Learning

## Project Overview
This project aims to develop a robust deep learning-based system for the automated detection of brain tumors using MRI (Magnetic Resonance Imaging) scans. By leveraging various machine learning approaches including supervised, unsupervised, and state-of-the-art deep learning models, we seek to create an accurate and reliable diagnostic tool that can assist medical professionals.

## Dataset
The project utilizes the Brain MRI Images for Brain Tumor Detection dataset available on Kaggle:
[Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection/code)

This dataset contains MRI scans of the brain, categorized into:
- MRI scans with tumors
- MRI scans without tumors

## Team Members and Responsibilities

| Learning Approach | Team Members | Responsibilities |
|-------------------|--------------|------------------|
| **Supervised Learning** | Jessica, Jane | - Implementation of CNN architectures<br>- Transfer learning with pre-trained models<br>- Model fine-tuning<br>- Ensemble methods<br>- Performance evaluation |
| **Unsupervised Learning** | Arcan, Diego | - Autoencoder implementation for anomaly detection<br>- Clustering algorithms<br>- Dimensionality reduction techniques<br>- Feature extraction<br>- Unsupervised pre-training |
| **State-of-the-Art Models** | Sean, Vincent | - Vision Transformer implementation<br>- Advanced segmentation models (U-Net, V-Net)<br>- Self-supervised learning approaches<br>- Latest research integration<br>- Performance benchmarking |

## Methodologies
Our approach encompasses multiple deep learning and machine learning paradigms:

### Supervised Learning Models
- Convolutional Neural Networks (CNNs)
- Transfer Learning with pre-trained models (e.g., ResNet, VGG, EfficientNet)
- Ensemble methods

### Unsupervised Learning Approaches
- Autoencoders for anomaly detection
- Clustering techniques for pattern discovery
- Dimensionality reduction methods

### State-of-the-Art Models
- Vision Transformers (ViT)
- Advanced segmentation models (U-Net, V-Net)
- Self-supervised learning approaches

## Project Structure
```
DeepLearning_for_BrainTumorDetection/
├── data/                     # Dataset storage
│   ├── raw/                  # Original downloaded dataset
│   └── processed/            # Preprocessed dataset
├── notebooks/                # Exploratory data analysis and model prototyping
│   ├── 01_data_exploration.ipynb
│   ├── 02_supervised_learning.ipynb
│   ├── 03_unsupervised_learning.ipynb
│   └── 04_sota_models.ipynb
├── src/                      # Source code
│   ├── data_processing/      # Data loading and preprocessing
│   │   ├── data_setup.py     # Dataset download and organization
│   │   ├── loader.py         # Data loaders for PyTorch and TensorFlow
│   │   ├── preprocessor.py   # Image preprocessing utilities
│   │   └── run_preprocessing.py # Main preprocessing script
│   ├── models/               # Model implementations
│   │   ├── supervised.py     # Supervised learning models
│   │   ├── unsupervised.py   # Unsupervised learning models
│   │   └── sota.py           # State-of-the-art models
│   ├── training/             # Training scripts
│   │   └── trainer.py        # Training loop implementations
│   ├── evaluation/           # Model evaluation and metrics
│   │   ├── metrics.py        # Evaluation metrics
│   │   └── visualize.py      # Results visualization
│   └── utils/                # Utility functions
│       ├── config.py         # Configuration parameters
│       └── logger.py         # Logging utilities
├── results/                  # Results, model checkpoints, visualizations
├── main.py                   # Main entry point for the application
└── requirements.txt          # Dependencies
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Git
- CUDA-compatible GPU (recommended for training deep learning models)

### Basic Installation

```bash
# Clone the repository
git clone [repository-url]
cd DeepLearning_for_BrainTumorDetection

# Set up a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Detailed Installation Instructions

#### 1. Clone the repository
```bash
git clone [repository-url]
cd DeepLearning_for_BrainTumorDetection
```

#### 2. Set up a virtual environment

**On Linux/macOS:**
```bash
python -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

#### 3. Install dependencies

**Standard installation:**
```bash
pip install -r requirements.txt
```

**GPU-specific installation (for CUDA support):**
If you have a CUDA-compatible GPU, you may want to install the GPU versions of TensorFlow and PyTorch:

```bash
# For PyTorch with CUDA 11.8 (modify based on your CUDA version)
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 -f https://download.pytorch.org/whl/cu118/torch_stable.html

# Install other requirements
pip install -r requirements.txt
```

#### 4. Verify installation

```bash
# Check if TensorFlow can see your GPU
python -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"

# Check if PyTorch can see your GPU
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('CUDA Device Count:', torch.cuda.device_count() if torch.cuda.is_available() else 'N/A')"
```

#### 5. Install Kaggle API credentials

To download the dataset automatically, you need to set up Kaggle API credentials:

1. Register for a Kaggle account at https://www.kaggle.com
2. Go to your account settings and click "Create New API Token"
3. This will download a `kaggle.json` file
4. Create a directory for your Kaggle credentials:

```bash
mkdir -p ~/.kaggle
cp /path/to/downloaded/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Troubleshooting

**Issue: Package installation errors**
```bash
# Try updating pip first
pip install --upgrade pip

# Install packages one by one
pip install numpy pandas scipy scikit-learn
pip install tensorflow torch torchvision torchaudio
# Continue with other packages...
```

**Issue: CUDA compatibility errors**
```bash
# Check your CUDA version
nvidia-smi

# Install compatible PyTorch version from https://pytorch.org/get-started/locally/
```

**Issue: Memory errors during dataset processing**
```bash
# Run preprocessing with smaller batch size
python src/data_processing/run_preprocessing.py --batch_size 16
```

For more detailed troubleshooting and installation guidance, please refer to the documentation for each library:
- [TensorFlow Installation Guide](https://www.tensorflow.org/install)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)

## Usage

### Data Preparation

1. **Download and preprocess the dataset**:
   ```bash
   # Make the preprocessing script executable
   chmod +x src/data_processing/run_preprocessing.py

   # Run the preprocessing pipeline
   python src/data_processing/run_preprocessing.py
   ```

   This will:
   - Download the Brain MRI dataset from Kaggle using kagglehub
   - Split the data into train/validation/test sets (70%/15%/15%)
   - Preprocess images (resize, normalize, enhance contrast)
   - Apply data augmentation for the training set
   - Create metadata and visualize sample images

2. **Alternative manual download**:
   If you prefer to download the dataset manually, you can:
   - Download from Kaggle: [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection/code)
   - Place the downloaded files in the `data/raw` directory
   - Run the preprocessing script as above

### Model Training

Each model type can be trained using the corresponding notebook or by running:

```bash
# Train a model (e.g., supervised CNN)
python main.py --mode train --model_type supervised

# Train unsupervised model
python main.py --mode train --model_type unsupervised

# Train state-of-the-art model
python main.py --mode train --model_type sota
```

### Evaluation

Evaluate trained models on the test set:

```bash
python main.py --mode evaluate --model_type [supervised|unsupervised|sota]
```

### Prediction

Run predictions on new MRI images:

```bash
python main.py --mode predict --model_type [supervised|unsupervised|sota] --data_path [path_to_image(s)]
```

## Data Preprocessing Details

The preprocessing pipeline performs the following steps:

1. **Data Organization**:
   - Splits the dataset into train (70%), validation (15%), and test (15%) sets
   - Maintains class balance in all splits
   - Creates directories for each split and class

2. **Image Preprocessing**:
   - Resizes images to a standard size (224x224 by default)
   - Normalizes pixel values to [0,1]
   - Enhances contrast using histogram equalization
   - Converts images to a consistent format

3. **Data Augmentation** (training set only):
   - Random horizontal flips
   - Random rotations (±15 degrees)
   - Brightness and contrast adjustments
   - Maintains original images alongside augmented ones

4. **Metadata Generation**:
   - Creates a metadata.csv file with dataset statistics
   - Generates visualizations of sample images

## Results
[To be added as the project progresses]

## Future Work
- Integration of multi-modal data
- Explainable AI components for better interpretability
- Development of a web-based interface for non-technical users
- Clinical validation studies

## Collaboration Guidelines

- Weekly team meetings to share progress and challenges
- Cross-team collaboration for integration of different approaches
- Knowledge sharing between teams to ensure consistent methodology
- Joint evaluation of all models against common metrics
- Shared documentation responsibilities

## Contributors
- Supervised Learning Team: Jessica, Jane
- Unsupervised Learning Team: Arcan, Diego
- State-of-the-Art Models Team: Sean, Vincent

## License
[To be added]

---
*Note: This README will be revised as the project develops.* 