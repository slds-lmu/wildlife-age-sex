# Wildlife Age-Sex Classification: Complete Setup Tutorial

This tutorial walks you through the complete workflow for wildlife age-sex classification using camera trap images. Follow these 7 main steps to get from raw images to trained models.

## 🎯 The 7-Step Workflow

1. **Import Raw Images** - Organize your camera trap images
2. **Run MegaDetector** - Detect wildlife and extract bounding boxes  
3. **Annotate with Interface** - Label images using the web interface
4. **Convert to Training Format** - Use enrichment script to prepare data
5. **Create Experiment Configs** - Set up configuration files for training
6. **Run Experiments** - Train and evaluate models
7. **Check Results** - Analyze performance and iterate

## Table of Contents
- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Step 1: Import Raw Images](#step-1-import-raw-images)
- [Step 2: Run MegaDetector](#step-2-run-megadetector)
- [Step 3: Annotate with Interface](#step-3-annotate-with-interface)
- [Step 4: Convert to Training Format](#step-4-convert-to-training-format)
- [Step 5: Create Experiment Configs](#step-5-create-experiment-configs)
- [Step 6: Run Experiments](#step-6-run-experiments)
- [Step 7: Check Results](#step-7-check-results)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements
- **Python**: 3.10 or higher
- **Operating System**: Linux, macOS, or Windows
- **Memory**: Minimum 8GB RAM (16GB+ recommended for large datasets)
- **Storage**: At least 10GB free space for dependencies and data
- **GPU**: Optional but highly recommended for training (CUDA-compatible)

### Required Software
- **uv**: Modern Python package manager (we'll install this)
- **Git**: For version control (if cloning from repository)

## Environment Setup

### Step 1: Install uv Package Manager

**On Linux/macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**On Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Alternatives:**
```bash
# Using pip
pip install uv
# Using Homebrew
brew install uv
```

### Step 2: Clone or Download the Project

If you have the project in a Git repository:
```bash
git clone <repository-url>
cd wildlife-age-sex
```

If you have the project files locally, navigate to the project directory:
```bash
cd /path/to/wildlife-age-sex
```

### Step 3: Create Virtual Environment and Install Dependencies

```bash
# Create virtual environment and install base dependencies
uv sync

# Install additional dependency groups as needed (automatically done for each preset command)
uv sync --group pipeline    # For ML pipeline (PyTorch, OpenCV)
uv sync --group frontend    # For Streamlit web interface
uv sync --group megadetector # For wildlife detection
```

### Step 4: Activate Virtual Environment

**Important**: You need to activate the virtual environment every time you start a new terminal session.

```bash
# Activate the virtual environment
source .venv/bin/activate  # On Linux/macOS
# or
.venv\Scripts\activate     # On Windows

# Verify activation (you should see (.venv) in your prompt)
python --version  # Should point to .venv/bin/python
```

**Note**: When the virtual environment is active, you'll see `(.venv)` at the beginning of your terminal prompt. This indicates that Python commands will use the project's isolated environment.

### Step 5: Create Required Directories

```bash
# Create all necessary directories
poe configure-dirs
```

This creates the following directory structure:
```
wildlife-age-sex/
├── configs/          # Configuration files
├── logs/            # Training and error logs
├── data/            # All data storage
│   ├── raw/         # Raw images and MegaDetector results
│   ├── preprocessed/ # Processed images for training
│   └── splits/      # Train/test data splits
└── models/          # Trained model files
    ├── age/         # Age classification models
    └── sex/         # Sex classification models
```

### Step 6: Verify Installation

```bash
# Test that everything is working
uv run python -c "import wildlifeml; print('Package imported successfully')"
```

---

## Step 1: Import Raw Images

### Organize Your Images

Create a directory structure for your raw images **inside the data directory**:

```bash
# Activate virtual environment (if not already active)
source .venv/bin/activate  # On Linux/macOS
# or .venv\Scripts\activate on Windows

# Create directory for your dataset
mkdir -p data/raw/my_dataset

# Copy your camera trap images here
# Images can be in any common format: .jpg, .jpeg, .png, .JPG
cp /path/to/your/images/* data/raw/my_dataset/
```

**Image Requirements:**
- Common formats: `.jpg`, `.jpeg`, `.png`, `.JPG`
- No specific size requirements (MegaDetector handles various sizes)
- Organize by location, date, or camera if desired

---

## Step 2: Run MegaDetector

### Detect Wildlife and Extract Bounding Boxes

MegaDetector will automatically find wildlife in your images and create bounding boxes:

```bash
# Run MegaDetector on your image directory
poe run-megadetector --image-dir data/raw/my_dataset
```

**What this does:**
- Scans all images for wildlife
- Creates bounding boxes around detected animals
- Saves results to `data/raw/my_dataset/md_unlabeled.json`

**Output Format:**
```json
{
  "0__IMG_001": {
    "bbox_id": "0__IMG_001",
    "image_id": "IMG_001", 
    "image_path": "data/raw/my_dataset/IMG_001.JPG",
    "category": 0,
    "bbox": [0.1, 0.2, 0.3, 0.4],
    "confidence": 0.95
  }
}
```

**Key Fields:**
- `bbox`: [x, y, width, height] normalized coordinates (0-1)
- `confidence`: Detection confidence (0-1, higher = more certain)
- `category`: 0 = animal, 1 = person, 2 = vehicle

---

## Step 3: Annotate with Interface

### Launch the Web Interface

```bash
# Activate virtual environment (if not already active)
source .venv/bin/activate  # On Linux/macOS
# or .venv\Scripts\activate on Windows

# Start the annotation interface
poe build-frontend
```

Open your browser to `http://localhost:8501`

### Annotate Your Images

1. **Go to Annotation Tab**
2. **Select your dataset**: Choose `data/raw/my_dataset`
3. **Start labeling**: 
   - View detected bounding boxes
   - Assign labels: `male`, `female`, `unknown` for sex
   - Assign labels: `juvenile`, `yearling`, `adult`, `unknown` for age
4. **Save annotations**: Results saved to `data/raw/my_dataset/annotations.json`

**Annotation Interface Features:**
- Zoom and pan images
- Skip uncertain images
- Progress tracking

**Annotation Interface Tips:**
- Think carefully about what classes and labels you want to include. Once you set these, they will stay consistent across all annotations for the image folder.
- If you do need to make changes to your defined classes or delete the label for an individual bounding box, you will need to edit the JSON file directly.

### Understanding Annotation Files

The annotation process creates several files in your dataset directory. The following snippets are examples. You will have complete control over the classes and labels you assign.

**1. `annotations.json`** - Main annotation file created by the web interface:
```json
{
  "0__IMG_001": {
    "image_id": "0__IMG_001",
    "original_image_id": "IMG_001",
    "image_path": "data/my_dataset/raw/IMG_001.JPG",
    "category": 0,
    "bbox": [0.1, 0.2, 0.3, 0.4],
    "confidence": 0.95,
    "source_file": "raw/md_unlabeled.json",
    "sex": "male",
    "age": "adult"
  },
  "1__IMG_001": {
    "image_id": "1__IMG_001", 
    "original_image_id": "IMG_001",
    "image_path": "data/my_dataset/raw/IMG_001.JPG",
    "category": 0,
    "bbox": [0.5, 0.6, 0.2, 0.3],
    "confidence": 0.87,
    "source_file": "raw/md_unlabeled.json",
    "sex": "female",
    "age": "juvenile"
  }
}
```
NOTE: When annotating, you will select a top level folder inside `data/` which will pull in all `md_unlabeled.json` files in any sub folders. The "source_file" key in the JSON output indicates the subfolder where the MegaDetector file associated with the image can be found.

**2. `annotation_config.json`** - Configuration file defining available classes:
```json
{
  "sex_classes": ["male", "female", "unknown"],
  "age_classes": ["juvenile", "yearling", "adult", "unknown"]
}
```
TIP: You can also label images for image-level metadata data at this time, e.g. is_not_alone_in_image: ["true", "false"]

**Key Points:**
- Each bounding box gets a unique `image_id` (format: `{index}__{original_image_id}`)
- Multiple animals can be detected in the same image (different `image_id` values)
- Once the preprocessing step is run each bounding box will be cropped and saved as a unique image
- The `category` field from MegaDetector is preserved (0 = animal)
- Your labels (e.g.,`sex`, `age`) are added during annotation

**Important**: The annotation interface will create these files automatically. You don't need to create them manually - just use the web interface to label your images and the files will be generated for you.

**Troubleshooting Annotation Issues:**

**Problem: Can't see bounding boxes in interface**
- Check that `md_unlabeled.json` exists in your image directory
- Verify the JSON file has valid format (use a JSON validator)
- Ensure image paths in the JSON match actual file locations

**Problem: Can't change class labels after starting**
We make it hard to do this to maintain maximum consistency across sessions and contributors. We _do not recommend_ making these types of changes, but they can be done if absolutely necessary.
- Edit `annotation_config.json` to modify available classes and their values
- If you do this you will need to restart you annotations (recommended) or add any new classes to the existing annotations manually, directly in the `annotations.json` file.
- Restart the web interface after making changes

**Problem: I made a mistake in my labeling**
- If you mislabeled an image and have not labeled any further images, it will be the last entry in the `annotations.json`.
- You can manually delete this entry, making sure to maintain valid JSON syntax in the file. 
- The image will be put back into the batch of unlabeled images once you press "Save & Next" in the frontend.

**Problem: Missing images in interface**
- Verify all images are in the same directory as and are listed in `md_unlabeled.json`
- Check that image file extensions match what's in the JSON
- Ensure images are in supported formats (.jpg, .jpeg, .png, .JPG)

### Understanding the Demo Data Structure

The project includes demo data to help you understand the expected format:

**Demo Files Location:**
- `data/demo/labeled_bbox_data.parquet` - Pre-annotated demo dataset
- `data/demo/md_unlabeled.json` - MegaDetector results for demo images
- `data/demo/preprocessed_demo_data/` - Preprocessed demo images

**Demo Data Schema:**
```python
# Example from demo data
{
    'image_id': 'IMG_001',
    'image': 'data/demo/preprocessed_demo_data/IMG_001.jpg',
    'bbox': [0.1, 0.2, 0.3, 0.4],  # [x, y, width, height] normalized
    'confidence': 0.95,
    'ex_class': 'label_1',
    'camera_id': "VF_014",
    ...
}
```

**Key Points:**
- Demo data shows the exact format your data should follow
- Images are already preprocessed and cropped
- Labels are already assigned for training examples
- You can use this as a template for your own data conversion

**Required Schema:**
```python
{
    'image_id': str,           # Unique identifier
    'image': str,              # Image file path
    'bbox': [float, float, float, float],  # [x, y, width, height]
    'confidence': float,       # Detection confidence
    'sex': str,                # 'male', 'female', 'unknown'
    'age': str,                # 'juvenile', 'yearling', 'adult', 'unknown'
    'metadata': int            # Additional metadata (optional)
}
```

---

## Step 4: Convert to Training Format

This step converts your annotated data into the format required for model training using the productionized enrichment script.

### Convert Annotations to Training Format

After annotation, you need to convert your JSON annotations to the Parquet format required for training. The project includes an enrichment script that handles this conversion, combines different annotation files, and can be used to add additional metadata:

```bash
# Activate virtual environment (if not already active)
source .venv/bin/activate  # On Linux/macOS
# or .venv\Scripts\activate on Windows

# Run enrichment with demo data (default)
poe run-enrichment

# Run enrichment with your own data (with metadata)
poe run-enrichment --annotation-dirs data/my_dataset/raw1 data/my_dataset/raw2 --metadata-file data/my_metadata.csv --output-path data/my_dataset/enriched_data.parquet

# Run enrichment without metadata
poe run-enrichment --annotation-dirs data/my_dataset/raw1 data/my_dataset/raw2 --output-path data/my_dataset/enriched_data.parquet --no-metadata
```

**What the enrichment script does:**
1. **Loads annotations** from one or more directories containing `annotations.json` files
2. **Joins metadata** (optional) from CSV or Parquet files by matching `original_image_id` (annotations) with `image_id` (metadata) - use `--no-metadata` to skip this step
3. **Converts to training format** with the required schema for model training
4. **Saves as Parquet** for efficient loading during training

**Metadata File Format:**
Your metadata file should be a CSV or Parquet with an `image_id` column that matches the `original_image_id` from your annotations:

```csv
image_id,camera_id,location,date,weather,temperature
VF_014_Session_4_20210902_I_00110a,VF_014,North_Field,2021-09-02,Sunny,15.5
VF_014_Session_4_20210902_I_00110b,VF_014,North_Field,2021-09-02,Sunny,15.5
```

**Command Options:**
- `--annotation-dirs` (`-a`): List of directories containing `annotations.json` files
- `--metadata-file` (`-m`): Optional metadata file (CSV or Parquet) to join
- `--output-path` (`-o`): Output path for the enriched Parquet file
- `--no-metadata`: Skip metadata joining and only convert annotations to training format

The script creates an enriched dataset ready for training with all your annotations and any additional metadata you provide (if not using `--no-metadata`).

---

## Step 5: Create Experiment Configs

### Copy and Modify Configuration Template

```bash
# Copy demo config as starting point
cp configs/demo__config.toml configs/new/my_experiment__config.toml
```

Edit your config file (e.g., `configs/example__config.toml`) and set the following fields according to your task.

---

## Step 6: Run Experiments

### Run one experiment at a time
```bash
# Activate virtual environment (if not already active)
source .venv/bin/activate  # On Linux/macOS
# or .venv\Scripts\activate on Windows

# Run all age experiments
poe run-pipeline --config configs/new/my_experiment__config.toml
```

### Run preprocessing and then a batch of experiments
```bash
# Activate virtual environment (if not already active)
source .venv/bin/activate  # On Linux/macOS
# or .venv\Scripts\activate on Windows

# All experiments in this setup should read from the same preprocessed data path
poe preprocess -c configs/new/my_experiment__config.toml
poe run-experiment -c configs/new
```

**What happens during training:**
1. **Preprocessing**: 
   - Loads your labeled data from Parquet file
   - Crops images using bounding box coordinates
   - Filters out low-confidence detections
   - Resizes images to target size (e.g., 224x224)
   - Splits data into train/test sets
2. **Training**: 
   - **Transfer Learning**: Trains only the classifier layers (10 epochs)
   - **Fine-tuning**: Unfreezes some backbone layers and trains further (10 epochs)
   - Uses data augmentation to improve generalization
3. **Evaluation**: 
   - Tests model on held-out test set
   - Generates comprehensive performance metrics
   - Saves results with timestamp for tracking

**Training Output:**
- Model files: `models/sex/my_experiment/model.pt`
- Training specs: `models/sex/my_experiment/tuning_specs.json`
- Evaluation results: `models/sex/my_experiment/TIMESTAMP__eval_results.json`

---

## Step 7: Check Results

### View Results in Web Interface

```bash
# Activate virtual environment (if not already active)
source .venv/bin/activate  # On Linux/macOS
# or .venv\Scripts\activate on Windows

# Launch the interface
poe build-frontend
```

Navigate to `http://localhost:8501` → **Results Tab**

**Results Features:**
- **Performance Metrics**: Accuracy, precision, recall, F1-score
- **Confusion Matrix**: Visual error analysis
- **Stratified Results**: Performance by subgroups
- **Error Analysis**: View misclassified images
- **Uncertainty Analysis**: Review low-confidence predictions

### Analyze Evaluation Files

Check the generated evaluation files:

```bash
# View evaluation results
cat models/new/my_experiment/*__eval_results.json

# Check error analysis
uv run python -c "
import pandas as pd
errors = pd.read_parquet('models/sex/my_experiment/*__eval_errors.parquet')
print(errors.head())
"
```

**Key Metrics to Review:**
- **Overall Accuracy**: How well the model performs
- **Per-Class Performance**: Which classes are hardest to classify
- **Confusion Matrix**: Common misclassification patterns
- **Uncertainty Rate**: How many predictions need manual review

### Iterate and Improve

Based on results, you can:

1. **Adjust Configuration**:
   - Increase training epochs
   - Try different model architectures
   - Adjust confidence thresholds

2. **Improve Data**:
   - Add more annotations for difficult classes
   - Remove low-quality images
   - Balance class distributions

3. **Run New Experiments**:
   - Create new config files with different settings
   - Compare multiple approaches
   - Fine-tune based on error analysis

## Troubleshooting

### Common Issues and Solutions

**1. Virtual Environment Issues:**
```bash
# If you get "command not found" errors, make sure venv is activated
source .venv/bin/activate  # On Linux/macOS
# or .venv\Scripts\activate on Windows

# Check if venv is active (should show (.venv) in prompt)
echo $VIRTUAL_ENV  # Should show path to .venv directory

# If venv doesn't exist, recreate it
rm -rf .venv
uv sync
```

**2. CUDA/GPU Issues:**
```bash
# Check CUDA availability
uv run python -c "import torch; print(torch.cuda.is_available())"

# Force CPU usage in config
device = "cpu"
```

**3. Memory Issues:**
```bash
# Reduce batch size in config
batch_size = 16  # or smaller

# Reduce image size
rescale_to = [128, 128]
```

**4. Dependency Conflicts:**
```bash
# Clean and reinstall
rm -rf .venv
uv sync --reinstall
```

**5. Configuration Errors:**
```bash
# Validate TOML syntax
uv run python -c "import tomli; tomli.load(open('configs/my_config.toml', 'rb'))"
```

**6. Data Loading Issues:**
```bash
# Check data format
uv run python -c "import pandas as pd; df = pd.read_parquet('data/your_data.parquet'); print(df.columns.tolist())"
```

**7. Annotation Interface Issues:**
- **Interface won't start**: Check that port 8501 is available, try `poe build-frontend` again
- **Can't see images**: Verify image paths in `md_unlabeled.json` match actual file locations
- **Annotations not saving**: Check file permissions, ensure you're not editing JSON files while interface is running

### Performance Optimization

**For Large Datasets:**
- Use GPU training (`device = "cuda"`)
- Use smaller image sizes for faster processing
- Consider data sampling for initial experiments

**For Better Accuracy:**
- Increase training epochs
- Use stronger data augmentation
- Try different backbone models (`densenet161`, `densenet201`)

## Next Steps

After successful setup:

1. **Experiment with different configurations** in the `configs/` directory
2. **Analyze results** using the web interface
3. **Fine-tune hyperparameters** based on validation performance
4. **Scale up** to larger datasets once you're satisfied with the approach
5. **Deploy models** for production use

## Getting Help

- Check the logs in the `logs/` directory for detailed error messages
- Review the example configurations in `configs/example_complete__config.toml`
- Use the demo data to verify your setup is working correctly
- Consult the project documentation and code comments for detailed explanations

---

**Congratulations!** You now have a fully functional classification pipeline. The system is designed to be flexible and scalable, allowing you to experiment with different models, data, and configurations to achieve the best results for your specific use case.
