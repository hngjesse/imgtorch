# ImgTorch: A Lightweight Image Dataset Loader for PyTorch

**ImgTorch** is a minimal yet powerful image importer and preprocessor tailored for classification tasks in PyTorch. It supports both common and RAW image formats, applies consistent preprocessing, and enables fast dataset creation and visualization — all with minimal dependencies.

---

## Key Features

- **Directory-based class labeling** — each subfolder = one class
- **Supports RAW and standard formats**: `.jpg`, `.png`, `.cr2`, `.nef`, `.dng`, etc.
- **Aspect-preserving resize and padding** to uniform shape
- **Converts to PyTorch tensors** ready for training
- **Save/load dataset** as `.pt` files for fast reuse
- **Live previews** via matplotlib and terminal-friendly ASCII art
- **Graceful handling** of unreadable or corrupted files
- **Minimal dependencies**: Only uses PyTorch, Pillow, rawpy, matplotlib, tqdm

---

## Folder Structure

Your dataset should be organized by class subfolders:

```
your_dataset/
├── ClassA/
│   ├── img1.jpg
│   └── img2.png
├── ClassB/
│   ├── img3.cr2
│   └── img4.jpeg
```

---

## Getting Started

### 1. Initialize

```python
from imgtorch import ImgTorch

imp = ImgTorch(
    baseDir="your_dataset",
    classDir=["ClassA", "ClassB"],
    imageSize=(128, 128)
)
```

### 2. Load and Preprocess

```python
imp.collect_images()     # Scan all images
imp.shuffle_images()     # Optional: randomize order
imp.process_images()     # Load, resize, convert to tensor
```

### 3. Preview

```python
imp.preview_images(max_images=6)         # Matplotlib preview
imp.preview_ASCII(count=3, contrast=1.2) # Terminal-friendly ASCII visualization
```

### 4. Save and Use

```python
imp.save_dataset("dataset.pt")    # Save tensors to disk
X, Y = imp.get_dataset()          # Retrieve processed data
print(X.shape, Y.shape)
```

---

## Additional Notes

- RAW formats are decoded using `rawpy` and converted to RGB using `Pillow`.
- Aspect ratio is preserved using `thumbnail()` and centered padding.
- Corrupted or unreadable files are skipped and listed.

---

## Dependencies

Install with:

```bash
pip install torch torchvision pillow rawpy matplotlib tqdm
```

---

## Author

**Jesse Hng**, 2025  
_A practical tool for quick dataset preparation in terminal or notebook environments._
