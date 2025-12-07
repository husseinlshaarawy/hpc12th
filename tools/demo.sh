#!/bin/bash

# Demo script showing all implemented features
# Creates a sample image and applies various filters

set -e

echo "============================================"
echo "HPC Image Processing - Feature Demo"
echo "============================================"
echo ""

# Create demo directory
DEMO_DIR="/tmp/iproc_demo"
mkdir -p $DEMO_DIR
cd $DEMO_DIR

# Generate a test image using Python
echo "1. Creating test image (256x256)..."

# Check if Python modules are available
python3 -c "import numpy, PIL" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: Required Python modules not found (numpy, PIL)"
    echo "Please install: sudo apt-get install python3-numpy python3-pil"
    exit 1
fi

python3 << 'EOF'
import numpy as np
from PIL import Image, ImageDraw

# Create a test image with patterns
img = np.ones((256, 256, 3), dtype=np.uint8) * 255

# Add some shapes
pil_img = Image.fromarray(img)
draw = ImageDraw.Draw(pil_img)

# Draw circles
draw.ellipse([50, 50, 100, 100], fill=(255, 0, 0))
draw.ellipse([156, 50, 206, 100], fill=(0, 255, 0))
draw.ellipse([103, 156, 153, 206], fill=(0, 0, 255))

# Draw rectangles
draw.rectangle([80, 180, 176, 220], outline=(0, 0, 0), width=3)

# Save
pil_img.save('demo_input.png')
print("  Created: demo_input.png")
EOF

INPUT="demo_input.png"
BUILD_DIR="../../../../build"

if [ ! -d "$BUILD_DIR" ]; then
    echo "Error: Build directory not found. Please build the project first."
    exit 1
fi

echo ""
echo "2. Applying filters (Serial implementation)..."
echo ""

# Point operations
echo "  - Grayscale conversion..."
$BUILD_DIR/bin/iproc_demo grayscale $INPUT demo_grayscale.png

echo "  - Brightness adjustment..."
$BUILD_DIR/bin/iproc_demo brightness $INPUT demo_brightness.png 50

echo "  - Gamma correction..."
$BUILD_DIR/bin/iproc_demo gamma $INPUT demo_gamma.png 0.5

# Smoothing
echo "  - Gaussian blur..."
$BUILD_DIR/bin/iproc_demo gaussian-blur $INPUT demo_gaussian.png 5 1.5

echo "  - Median filter..."
$BUILD_DIR/bin/iproc_demo median $INPUT demo_median.png 5

# Edge detection
echo "  - Sobel edge detection..."
$BUILD_DIR/bin/iproc_demo sobel demo_grayscale.png demo_sobel.png

echo "  - Canny edge detection..."
$BUILD_DIR/bin/iproc_demo canny demo_grayscale.png demo_canny.png 50 150

# Morphological operations (on binary image)
echo "  - Binary threshold..."
$BUILD_DIR/bin/iproc_demo threshold demo_grayscale.png demo_binary.png 128

echo "  - Erosion..."
$BUILD_DIR/bin/iproc_demo erosion demo_binary.png demo_erosion.png 3

echo "  - Dilation..."
$BUILD_DIR/bin/iproc_demo dilation demo_binary.png demo_dilation.png 3

# Geometric transforms
echo "  - Rotation (45°)..."
$BUILD_DIR/bin/iproc_demo rotate $INPUT demo_rotated.png 45

echo "  - Scaling (2x)..."
$BUILD_DIR/bin/iproc_demo scale $INPUT demo_scaled.png 2.0 2.0

echo "  - Horizontal flip..."
$BUILD_DIR/bin/iproc_demo flip-h $INPUT demo_flip_h.png

# Color operations
echo "  - HSV adjustment..."
$BUILD_DIR/bin/iproc_demo adjust-hsv $INPUT demo_hsv.png 30 1.2 1.1

# Bonus features
echo "  - Otsu threshold..."
$BUILD_DIR/bin/iproc_demo otsu demo_grayscale.png demo_otsu.png

echo "  - Sauvola threshold..."
$BUILD_DIR/bin/iproc_demo sauvola demo_grayscale.png demo_sauvola.png 15 0.5

echo ""
echo "3. Testing OpenMP implementation (if available)..."
if [ -f "$BUILD_DIR/bin/iproc_omp_demo" ]; then
    echo "  - Gaussian blur with 4 threads..."
    $BUILD_DIR/bin/iproc_omp_demo gaussian-blur $INPUT demo_omp_gaussian.png 5 1.5 4
    
    echo "  - Median filter with 4 threads..."
    $BUILD_DIR/bin/iproc_omp_demo median $INPUT demo_omp_median.png 5 4
else
    echo "  OpenMP demo not available (build with -DBUILD_OPENMP=ON)"
fi

echo ""
echo "============================================"
echo "Demo complete!"
echo "============================================"
echo ""
echo "Generated files in: $DEMO_DIR"
ls -1 demo_*.png | head -20
echo ""
echo "Total files created: $(ls -1 demo_*.png | wc -l)"
