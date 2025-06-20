import torch
from PIL import Image
from pathlib import Path
import rawpy
from torchvision import transforms
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
#import plotext as plt_ext


class ImgTorch:
    """
    ImgTorch: A lightweight image import and preprocessing class.

    This class simplifies the task of importing images from folders, converting them
    to RGB or grayscale format, resizing and padding them uniformly, and converting
    them into PyTorch tensors for machine learning tasks.

    Parameters:
    -----------
    baseDir : str
        Path to the dataset root directory (containing class subfolders).
    classDir : list[str]
        A list of subfolder names (each representing a class).
    imageSize : tuple[int, int], optional
        Default image size to which all images will be resized. Can be overridden in process_images().
    nonRawSpecialExt : list[str], optional
        Additional non-RAW image file extensions to support.

    Example:
    --------
    >>> img_loader = ImgTorch("data/", ["Cat", "Dog"])
    >>> img_loader.collect_images()
    >>> img_loader.shuffle_images()
    >>> img_loader.process_images(imageSize=(64, 64), format='L')
    >>> img_loader.preview_images()
    >>> img_loader.save_dataset("dataset.pt")
    """

    __rawExt = ['.cr2', '.cr3', '.cr4', '.nef', '.arw', '.orf', '.rw2', '.dng', '.nrw', '.srf', '.sr2']
    __defExt = ['.jpg', '.jpeg', '.png']

    def __init__(self, baseDir, classDir, imageSize=(128, 128), nonRawSpecialExt=[]):
        """
        Constructor to initialize directory structure and configuration.
        """
        if not isinstance(baseDir, str):
            raise TypeError("baseDir must be a string representing the directory path.")
        if not isinstance(classDir, list) or not all(isinstance(cls, str) for cls in classDir):
            raise TypeError("classDir must be a list of strings representing class names.")
        if not isinstance(imageSize, tuple) or len(imageSize) != 2 or not all(isinstance(dim, int) for dim in imageSize):
            raise TypeError("imageSize must be a tuple of two integers (height, width).")
        if not isinstance(nonRawSpecialExt, list) or not all(isinstance(ext, str) for ext in nonRawSpecialExt):
            raise TypeError("nonRawSpecialExt must be a list of strings representing additional file extensions.")
        
        self._baseDir = Path(baseDir)
        self._classDir = classDir
        self._imageSize = imageSize
        self._extensions = self.__defExt + self.__rawExt + nonRawSpecialExt
        self._filepaths = []
        self._labels = []
        self._X = None
        self._Y = None

    def __repr__(self):
        """
        Print a summary of current state of the ImgTorch object.
        """
        list_dir = [str(self._baseDir / cls) for cls in self._classDir]
        return ("----------------------------------------------------------------------- \n"
                f"<ImgTorch's informations>\n"
                "----------------------------------------------------------------------- \n"
                f"  list of directory          : { '\n                               '.join(list_dir)}\n"
                f"  No. of collected files     : {len(self._filepaths)}\n"
                f"  X shape of collected files : {None if self._X is None else self._X.shape}\n"
                f"  Y shape of collected files : {None if self._Y is None else self._Y.shape}\n"
                "----------------------------------------------------------------------- \n")

    def collect_images(self):
        """
        Scan the class folders and collect all valid image file paths.
        """
        for i, cls in enumerate(self._classDir):
            folder = self._baseDir / cls
            for file in folder.iterdir():
                if file.is_file() and file.suffix.lower() in self._valid_extensions():
                    self._filepaths.append(str(file))
                    self._labels.append(i)
        if not self._filepaths:
            raise ValueError("No image files found. Check directory and class names.")
        print(f"Collected {len(self._filepaths)} files.")

    def shuffle_images(self):
        """
        Shuffle the image file paths and labels to randomize the dataset.
        """
        perm = torch.randperm(len(self._filepaths))
        self._filepaths = [self._filepaths[i] for i in perm]
        self._labels = [self._labels[i] for i in perm]
        print("Shuffling complete.")

    def process_images(self, imageSize=(128,128), format='RGB'):
        """
        Load, convert, resize and pad all images to tensors.

        Parameters:
        -----------
        imageSize : tuple[int, int]
            Desired output image size (height, width).
        format : str
            Color mode. 'RGB' for color, 'L' for grayscale.
        """
        if format not in ['RGB', 'L']:
            raise ValueError("Format must be 'RGB' or 'L'.")

        X, Y = [], []
        error_files = []

        print(f"Processing images as {format}...")
        for i, fp in tqdm(enumerate(self._filepaths), total=len(self._filepaths)):
            try:
                img = self._load_image(fp, format)
                img = self._resize_and_pad(img, *imageSize)
                img_tensor = transforms.ToTensor()(img)
                X.append(img_tensor)
                Y.append(self._labels[i])
            except Exception:
                error_files.append(fp)

        if not X:
            raise RuntimeError("All image processing failed — no valid images found.")

        self._X = torch.stack(X)
        self._Y = torch.tensor(Y, dtype=torch.long)
        print("Image processing complete.")
        
        if error_files:
            print(f"\n⚠️ Skipped {len(error_files)} files due to errors.")
            for bad_file in error_files[:10]:
                print(f" - {bad_file}")
            if len(error_files) > 10:
                print(f" ... and {len(error_files) - 10} more.")

    def save_dataset(self, out_path):
        """
        Save the processed dataset to a .pt file.
        """
        if self._X is None or self._Y is None:
            raise ValueError("Dataset has not been processed yet. Call `process_images()` first.")
        torch.save({'X': self._X, 'Y': self._Y}, out_path)
        print(f"Dataset saved to {out_path}")

    def _valid_extensions(self):
        """
        Return list of valid image extensions.
        """
        return [ext.lower() for ext in self._extensions]

    def _load_image(self, path, format):
        """
        Load an image (including RAW formats) and convert it to desired mode.

        Parameters:
        -----------
        path : str
            Path to the image file.
        format : str
            Desired color format: 'RGB' or 'L'.
        """
        ext = Path(path).suffix.lower()
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=UserWarning)
                if ext in self.__rawExt:
                    with rawpy.imread(path) as raw:
                        rgb = raw.postprocess()
                    img = Image.fromarray(rgb)
                else:
                    img = Image.open(path)
                return img.convert(format)
        except UserWarning as w:
            raise RuntimeError(f"{w}")

    def _resize_and_pad(self, img, height, width):
        """
        Resize and pad image to fixed size without distortion.

        Parameters:
        -----------
        img : PIL.Image
            Image to resize.
        height : int
            Target height.
        width : int
            Target width.
        """
        img.thumbnail((width, height), Image.BILINEAR)
        new_img = Image.new(img.mode, (width, height))  # respect "RGB" or "L"
        paste_pos = ((width - img.width) // 2, (height - img.height) // 2)
        new_img.paste(img, paste_pos)
        return new_img

    def get_dataset(self):
        """
        Returns:
        --------
        X : torch.Tensor
            Image tensors of shape [N, C, H, W].
        Y : torch.Tensor
            Labels as integers.
        """
        return self._X, self._Y

    def preview_images(self, max_images=6):
        """
        Display a preview grid of randomly chosen images.

        Parameters:
        -----------
        max_images : int
            Number of images to preview (max 6 for display clarity).
        """
        if self._X is None or self._Y is None:
            raise ValueError("Dataset not processed. Call `process_images()` first.")

        n = min(max_images, len(self._X))
        indices = torch.randperm(len(self._X))[:n]
        to_pil = transforms.ToPILImage()
        plt.figure(figsize=(n * 2, 2.5))

        for i, idx in enumerate(indices):
            img_tensor = self._X[idx]
            label = self._classDir[self._Y[idx]]

            plt.subplot(1, n, i + 1)

            if img_tensor.shape[0] == 1:
                plt.imshow(img_tensor.squeeze(0), cmap='gray')
            else:
                img = to_pil(img_tensor)
                plt.imshow(img)

            plt.title(label)
            plt.axis('off')

        plt.suptitle("Preview of Random Processed Images", fontsize=14)
        plt.tight_layout()
        plt.show()

    def preview_ASCII(self, count=3, background='dark', contrast=1.0):
        """
        Print ASCII representations of `count` random images from the dataset.

        Parameters:
        -----------
        count : int
            Number of images to display as ASCII.
        background : str
            'dark' for dark terminal background (default),
            'light' for light background (inverts character map).
        aspect_ratio : float
            Height/width aspect correction factor. Characters are taller than wide.
        contrast : float
            Adjusts image contrast (1.0 = default, >1.0 = more contrast, <1.0 = less).
        """
        if self._X is None or self._Y is None:
            raise ValueError("Dataset not processed. Call `process_images()` first.")

        # Clean grayscale-friendly character set
        ascii_chars = "█▇▆▆▅▅▄▃▂▁  "
        ascii_chars = "■■▦▦▥▥◍◍❂❂❆❆❁❁✵✵✭✭▵▵✩✧◦ "
        ascii_chars = "■■▦▦▥▥◍❂❆❁✵✭▵✩✧◦ "

        if background == 'dark':
            ascii_chars = ascii_chars[::-1]

        indices = torch.randperm(len(self._X))[:count]

        for idx in indices:
            img_tensor = self._X[idx]

            # Convert to grayscale if RGB
            if img_tensor.shape[0] == 3:
                tensor = 0.2989 * img_tensor[0] + 0.5870 * img_tensor[1] + 0.1140 * img_tensor[2]
            elif img_tensor.shape[0] == 1:
                tensor = img_tensor[0]
            else:
                raise ValueError("Unsupported image tensor shape.")

            # Resize: correct for character aspect ratio
            aspect_ratio = 2.8
            height, width = tensor.shape
            new_height = int(height / aspect_ratio)
            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0).unsqueeze(0),
                size=(new_height, width),
                mode='bilinear',
                align_corners=False
            ).squeeze()

            # Normalize to [0, 1]
            norm = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-5)

            # Apply contrast adjustment
            norm = 0.5 + contrast * (norm - 0.5)
            norm = norm.clamp(0, 1)

            # Map to ASCII indices
            idx_map = (norm * (len(ascii_chars) - 1)).long()

            print("\n" + "-" * idx_map.shape[1])
            for row in idx_map:
                print(''.join(ascii_chars[i] for i in row))
            print("-" * idx_map.shape[1])

            label_idx = self._Y[idx]
            label_name = self._classDir[label_idx]
            print(f"[Label: {label_name}]")
'''
    def plotext_preview(self, count=3, grayscale=True):
        """
        Display a preview of random images in the terminal using plotext.

        Parameters:
        -----------
        count : int
            Number of random images to display.
        grayscale : bool
            Whether to render the image in grayscale.
        fast : bool
            Use fast rendering mode (faster, less customizable).
        """
        if self._X is None or self._Y is None:
            raise ValueError("Dataset not processed. Call `process_images()` first.")

        indices = torch.randperm(len(self._X))[:count]
        to_pil = transforms.ToPILImage()

        for idx in indices:
            img_tensor = self._X[idx]
            label_idx = self._Y[idx]
            label_name = self._classDir[label_idx]

            # Convert tensor to PIL, save to buffer
            img_pil = to_pil(img_tensor)
            temp_path = "/tmp/imgtorch_preview.png"
            img_pil.save(temp_path)

            # Plot using plotext
            plt_ext.clear_data()
            plt_ext.plotsize(128, 128)
            plt_ext.image_plot(temp_path, fast=False, grayscale=grayscale)
            plt_ext.title(f"[Label: {label_name}]")
            plt_ext.show()
'''

# Example usage:# img_imp = ImgTorch(baseDir='path/to/data', classDir=['class1', 'class2'], imageSize=(128, 128))
# img_imp.collect_images()
# img_imp.shuffle_images()
# img_imp.process_images()
# img_imp.save_dataset('dataset.pth')
# X, Y = img_imp.get_dataset()
# print(img_imp)
# print(f"X shape: {X.shape}, Y shape: {Y.shape}")

