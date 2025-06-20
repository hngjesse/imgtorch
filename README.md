# ImgTorch: Image Importer and Preprocessor for Classification Tasks
`ImgTorch` is a lightweight, dependency-conscious Python class for importing, preprocessing, and managing image datasets for machine learning projects using PyTorch.

This tool helps automate the process of:
- Collecting image file paths from organized folders
- Supporting raw and standard image formats
- Resizing and padding images uniformly
- Converting images to PyTorch tensors
- Saving the dataset as `.pt` files
- Previewing image samples for verification

---

## Features
- Supports JPEG, PNG, and RAW image formats (e.g., `.cr2`, `.nef`, `.dng`, etc.)
- Directory-based class labeling: each subfolder is treated as a class
- Automatic resizing and padding to ensure uniform dimensions
- Error handling for corrupted or unreadable images
- Live preview of random processed images
- Minimal dependencies: only uses PyTorch, PIL, rawpy, and matplotlib

---

## Folder Structure
Expected structure for input images:

your_dataset/
├── ClassA/
│ ├── img1.jpg
│ └── img2.png
├── ClassB/
│ ├── img3.cr2
│ └── img4.jpeg

---


## Getting Started
### 1. Initialize
```python
from imgtorch import ImgTorch

base_dir = 'your_dataset'
classes = ['ClassA', 'ClassB']
img_size = (128, 128)
imp = ImgTorch(baseDir=base_dir, classDir=classes, imageSize=img_size)
```

### 2. Load and Preprocess Images
```python
imp.collect_images()
imp.shuffle_images()
imp.process_images()
```

### 3. Preview and Export the Data
```python
imp.preview_images(max_images=6)
imp.save_dataset('output_dataset.pt')
imp.process_images()
```

### 4. Use the Dataset
```python
X, Y = imp.get_dataset()
print(X.shape, Y.shape)  # Example: (N, 3, 128, 128), (N,)
```


## Notes
- Corrupted or unreadable files (e.g., truncated TIFFs) are automatically skipped and reported.
- RAW files are processed using rawpy, then converted to RGB with PIL.
- Image resizing is done with thumbnail() to preserve aspect ratio, followed by center-padding.

## Dependencies
- Make sure to install:
- pip install torch torchvision pillow rawpy matplotlib tqdm

## Author
Created by Jesse Hng, 2025
A practical tool for self-managed ML dataset handling.
