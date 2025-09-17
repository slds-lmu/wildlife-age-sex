# Wildlife Age-Sex Classification

A machine learning pipeline for classifying wildlife by age and sex characteristics using camera trap images. This project provides a complete end-to-end solution for wildlife researchers and conservationists to automatically analyze camera trap data.

## 🎯 The 7-Step Workflow

1. **Import Raw Images** - Organize your camera trap images
2. **Run MegaDetector** - Detect wildlife and extract bounding boxes  
3. **Annotate with Interface** - Label images using the web interface
4. **Convert to Training Format** - Use enrichment script to prepare data
5. **Create Experiment Configs** - Set up configuration files for training
6. **Run Experiments** - Train and evaluate models
7. **Check Results** - Analyze performance and iterate

## 🎯 Overview

This project is designed as a flexible tool for running machine learning experiments on camera trap image data. It provides an end-to-end pipeline for data preprocessing, model training, and evaluation, with a focus on ease of experimentation and extensibility.

As a proof of concept, we implemented age and sex classification for wildlife images, using deep learning models to predict:
- **Sex**: Male, Female, Unknown
- **Age**: Juvenile, Yearling, Adult, Unknown

The pipeline integrates with **MegaDetector** for automated wildlife detection and offers a web interface for data annotation and model evaluation.

## ✨ Key Features

- **End-to-end ML Pipeline**: From raw images to trained models
- **MegaDetector Integration**: Automatic wildlife detection and bounding box extraction
- **Flexible Configuration**: TOML-based configuration system for easy experimentation
- **Web Interface**: Streamlit-based frontend for annotation and evaluation
- **Multiple Model Architectures**: Support for ResNet, VGG, DenseNet backbones
- **Transfer Learning**: Efficient training with pretrained models
- **Comprehensive Evaluation**: Detailed metrics and uncertainty analysis
- **Data Augmentation**: Robust training with realistic image variations

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- 8GB+ RAM (16GB+ recommended)
- GPU recommended for training

### Installation

1. **Install uv** (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Clone and setup**:
```bash
git clone <repository-url>
cd wildlife-age-sex

# Create virtual environment and install dependencies
uv sync
```

3. **Activate virtual environment** (required for each new session):
```bash
# Activate the virtual environment
source .venv/bin/activate  # On Linux/macOS
# or
.venv\Scripts\activate     # On Windows

# Verify activation (you should see (.venv) in your prompt)
which python  # Should point to .venv/bin/python
```

4. **Run demo pipeline**:
```bash
poe run-pipeline --config configs/demo__config.toml
```

5. **Launch web interface**:
```bash
poe build-frontend
# Open http://localhost:8501 in your browser
```

## 📋 Complete Workflow Example

Here's how to process your own camera trap images:

```bash
# Activate virtual environment (if not already active)
source .venv/bin/activate  # On Linux/macOS
# or .venv\Scripts\activate on Windows

# 1. Make new directory and import raw images
# You can also do this by dragging and doping in your code editor
mkdir -p data/my_dataset/raw
cp /path/to/your/images/* data/my_dataset/raw/

# 2. Run MegaDetector
poe run-megadetector --image-dir data/my_dataset/raw/

# 3. Annotate with interface
poe build-frontend
# Open http://localhost:8501 → Annotation tab → Label your images

# 4. Convert annotations to training format
poe run-enrichment --annotation-dirs data/my_dataset/raw --output-path data/my_dataset/enriched_data.parquet

# 5. Create experiment config based on the demo
cp configs/demo__config.toml configs/my_experiment__config.toml
# Edit the config file with your paths and settings

# 6. Run experiments
poe run-pipeline --config configs/my_experiment__config.toml

# 7. Check results
# Open http://localhost:8501 → Results tab → View performance metrics
```

## 📁 Project Structure

```
wildlife-age-sex/
├── configs/                 # Configuration files
│   ├── age/                 # Age classification experiments
│   ├── sex/                 # Sex classification experiments
│   └── demo__config.toml    # Demo configuration
├── data/                    # Data storage
│   ├── demo/                # Demo dataset
│   └── preprocessed/        # Data used for the age/sex classification is not tracked in Github due to high volume
├── models/                  # Trained models
│   ├── age/                 # Age classification models
│   ├── sex/                 # Sex classification models
│   ├── demo_model/          # Demo model
│   └── tensorflow_archive/  # Archive of models trained using legacy Tensorflow pipeline
├── scripts/                 # Main pipeline scripts
│   ├── enrichment.py        # Join annotation files and add metadata
│   ├── megadetector.py      # Bounding box detection with MegaDetector
│   ├── preprocess.py        # Data preprocessing
│   ├── train.py             # Model training
│   └── evaluate.py          # Model evaluation
├── frontend/                # Web interface
│   ├── app.py               # Streamlit main app
│   └── tabs/                # Interface tabs
├── wildlifeml/              # Core package
│   ├── io/                  # Data I/O utilities
│   ├── preprocess/          # Data preprocessing
│   └── train/               # Training and evaluation
├──  pyproject.toml          # Project configuration
└──  SETUP_TUTORIAL.md       # Detialed tutorial
```

## 🔧 Usage

### Configuration

Create a configuration file (see `configs/example__config.toml` for all options):

```toml
[globals]
target_column = "sex" # or "age"
classes = ["male", "female", "unknown"]

[io.data]
raw_data_filepath = "data/your_data/labeled_bbox_data.parquet"
preprocessed_data_filepath = "data/your_data/preprocessed_data.parquet"
train_filepath = "data/your_data/train.parquet"
test_filepath = "data/your_data/test.parquet"

[io.model]
model_dir = "models/your_experiment"

[train]
backbone_model = "resnet50"
batch_size = 32
transfer_epochs = 10
finetune_epochs = 10
```

### Pipeline Commands

**Complete Pipeline** (Preprocess → Train → Evaluate):
```bash
poe run-pipeline --config configs/your_config.toml
```
Or you can use the abbreviated command flag for `--config`: `-c`
```bash
poe run-pipeline -c configs/your_config.toml
```

**Individual Steps**:
```bash
# Preprocessing
poe preprocess -c configs/your_config.toml

# Training
poe train -c configs/your_config.toml

# Evaluation
poe evaluate -c configs/your_config.toml
```

**MegaDetector** (for raw images):
```bash
poe run-megadetector --image-dir /path/to/images
```
Or you can use the abbreviated command flag for `--image-dir`: `-i`
```bash
poe run-megadetector -i /path/to/images
```

**Data Enrichment** (convert annotations to training format):
```bash
# Basic enrichment (demo data)
poe run-enrichment

# Custom enrichment with metadata
poe run-enrichment --annotation-dirs data/dir1 data/dir2 --metadata-file data/metadata.csv --output-path data/enriched.parquet
```
Or use abbreviated flags: `-a` for `--annotation-dirs`, `-m` for `--metadata-file`, `-o` for `--output-path`
To run without metadata, add the `--no-metadata` flag to the end of your command. This will override the `--metadata-file` flag.

**Web Interface**:
```bash
poe build-frontend
```
This command will launch the front end onto you local host 8501 even if you are running the script in a GPU through a remote SSH.

### Data Format
To run the MegaDetector, simply drag and drop all you images into a single folder in the `data/` directory and run the command above. Once the script has run through, you will see a new file `md_unlabeled.json` in your image directory. That file has the following schema:

**MegaDetector Output Format:**
```json
{
  "0__IMG_001": {
    "bbox_id": "0__IMG_001",
    "image_id": "IMG_001", 
    "image_path": "data/raw/my_dataset/IMG_001.JPG",
    "category": 0,
    "bbox": ["x_min", "y_min", "width", "height"],
    "confidence": 0.95
  }
}
```
The category for animals is `0`. You can read more about MegaDetector v6 [here](https://github.com/agentmorris/MegaDetector).

### Next Steps

After running MegaDetector, you'll need to:
1. **Annotate your images** using the web interface to label sex and age
2. **Convert to training format** using the provided scripts
3. **Create experiment configurations** for your specific use case

For detailed instructions on annotation, data conversion, and configuration, see the [Setup Tutorial](SETUP_TUTORIAL.md).

## 🧪 Experiments

The project supports multiple experimental configurations:

### Run one experiment at a time
```bash
# Run all age experiments
poe run-pipeline --config configs/demo__config.toml
```

### Run preprocessing and then a batch of experiments
```bash
# All experiments in this setup should read from the same preprocessed data path
poe preprocess -c configs/sex/experiment1__config.toml
poe run-experiment -c configs/sex
```

### Custom Experiments
Create your own configuration files in the `configs/` directory and run them individually or in batches.

## 📊 Model Architectures

Supported backbone models:
- **ResNet50**: Good balance of speed and accuracy
- **VGG19**: Classic architecture, good for interpretability
- **DenseNet161/201**: Often better accuracy, more memory intensive

## 🎨 Web Interface

The Streamlit frontend provides:

- **Annotation Tab**: Manual image labeling and quality control
- **Results Tab**: Model performance visualization
- **Error Viewing**: Analysis of prediction errors
- **Uncertainty Viewing**: Review of uncertain predictions

Access at `http://localhost:8501` after running `poe build-frontend`.

## 📈 Evaluation Metrics

The system provides comprehensive evaluation including:
- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Per-class metrics
- **Confusion Matrix**: Detailed error analysis
- **Stratified Analysis**: Performance across subgroups
- **Uncertainty Analysis**: Confidence-based filtering

## 🐛 Troubleshooting

For detailed troubleshooting and common issues, see the [Setup Tutorial](SETUP_TUTORIAL.md).

### Quick Fixes

**CUDA/GPU Problems**:
```bash
# Check GPU availability
uv run python -c "import torch; print(torch.cuda.is_available())"
```

**Memory Issues**: Reduce `batch_size` in your configuration file

**Data Loading Errors**: Verify your data format matches the expected schema

## 📚 Documentation

- **[Complete Setup Tutorial](SETUP_TUTORIAL.md)**: Detailed step-by-step guide from installation to evaluation
- **[Configuration Reference](configs/example_complete__config.toml)**: All available parameters
- **[Demo Configuration](configs/demo__config.toml)**: Quick start example config

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **MegaDetector**: Microsoft AI for Earth's wildlife detection model
- **PyTorch**: Deep learning framework
- **Streamlit**: Web interface framework
- **Camera trap data providers**: Wildlife researchers and conservation organizations

## 📞 Support

For questions and support:
- Check the [troubleshooting section](#-troubleshooting)
- Review the [setup tutorial](SETUP_TUTORIAL.md)
- Open an issue on GitHub
- Contact the development team

---

**Built for wildlife researchers, conservationists, and machine learning practitioners working with camera trap data.**