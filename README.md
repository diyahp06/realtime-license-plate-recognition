# Real-Time Automatic License Plate Recognition (ALPR)

A Computer Vision and Deep Learning system built using Python to detect, segment, and recognize car license plates from real-time video streams. 

## Features
* **Adaptive Noise Reduction & Binarization**: Leverages Gaussian Blurring, Otsu's Thresholding, and Morphological operations to filter dynamic video backgrounds.
* **Structural Area Boundary Filtering**: Isolates exact license plate dimensions based on custom aspect ratio limits and contour constraints.
* **Character Image Segmentation**: Converts plate contours into unique HSV-derived isolated matrices.
* **Tensor-Based Characters Recognition**: Integrates an OCR model inside a TensorFlow graph environment to match segmented letters and digits.

## Project Structure
```text
├── model/
│   ├── binary_128_0.50_ver3.pb        # Pre-trained CNN character graph 
│   └── binary_128_0.50_labels_ver2.txt # Label indexes map
├── plate_finder.py                    # Contour processing & segmentation module
├── main.py                            # Stream loop controller & execution logic
├── requirements.txt                   # Local project environment tracking
└── README.md                          # Project documentation