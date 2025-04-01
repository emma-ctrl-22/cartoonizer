# Image Cartoonization Script

This script uses the WhiteBox Cartoonization model to transform photos into cartoon-style images.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Place an image file (jpg, jpeg, or png) in the same directory as the script.

## Usage

Simply run the script:
```bash
python cartoonize.py
```

The script will:
1. Automatically download the pretrained model on first run
2. Process the first image file it finds in the directory
3. Save the cartoonized result as `cartoon_result.jpg`

## Requirements
- Python 3.6+
- TensorFlow 1.15.0 or higher
- OpenCV
- NumPy
- Pillow
- Other dependencies listed in requirements.txt 