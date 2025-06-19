import torch
from PIL import Image
from pathlib import Path
import rawpy
from torchvision import transforms
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt

class ImgImp:
    __rawExt = ['.cr2', '.cr3', '.cr4', '.nef', '.arw', '.orf', '.rw2', '.dng', '.nrw', '.srf', '.sr2']
    __defExt = ['.jpg', '.jpeg', '.png']

    def __init__(self, baseDir, classDir, imageSize=(128, 128), nonRawSpecialExt=[]):
        # Initialize the attributes
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
        list_dir = []
        for cls in self._classDir:
            list_dir.append(str(self._baseDir/cls))
        return ("----------------------------------------------------------------------- \n"
                f"<ImgImp's informations>\n"
                "----------------------------------------------------------------------- \n"
                f"  list of dir  : { '\n                 '.join(list_dir)}\n"
                f"  image_size   : H = {self._imageSize[0]}, W = {self._imageSize[1]}\n"
                f"  No. of files : {len(self._filepaths)}\n"
                f"  X shape      : {None if self._X is None else self._X.shape}\n"
                f"  Y shape      : {None if self._Y is None else self._Y.shape}\n"
                "----------------------------------------------------------------------- \n")

    def collect_images(self):
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
        perm = torch.randperm(len(self._filepaths))
        self._filepaths = [self._filepaths[i] for i in perm]
        self._labels = [self._labels[i] for i in perm]
        print("Shuffling complete.")



    '''
    def process_images(self):
        X, Y = [], []
        error_files = []

        print("Processing images...")
        for i, fp in tqdm(enumerate(self._filepaths), total=len(self._filepaths)):
            try:
                img = self._load_image(fp)
                img = self._resize_and_pad(img, *self._imageSize)
                img_tensor = transforms.ToTensor()(img)
                X.append(img_tensor)
                Y.append(self._labels[i])
            except Exception as e:
                error_files.append(fp)

        if not X:
            raise RuntimeError("All image processing failed — no valid images found.")

        self._X = torch.stack(X)
        self._Y = torch.tensor(Y, dtype=torch.long)
        print("Image processing complete.")
        
        if error_files:
            print(f"\n⚠️ Skipped {len(error_files)} files due to errors.")
            for bad_file in error_files[:10]:  # Show up to 10 for brevity
                print(f" - {bad_file}")
            if len(error_files) > 10:
                print(f" ... and {len(error_files) - 10} more.")
'''

    def process_images(self, format='RGB'):
        if format not in ['RGB', 'L']:
            raise ValueError("Format must be 'RGB' or 'L'.")

        X, Y = [], []
        error_files = []

        print(f"Processing images as {format}...")
        for i, fp in tqdm(enumerate(self._filepaths), total=len(self._filepaths)):
            try:
                img = self._load_image(fp, format)
                img = self._resize_and_pad(img, *self._imageSize)
                img_tensor = transforms.ToTensor()(img)
                X.append(img_tensor)
                Y.append(self._labels[i])
            except Exception as e:
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
        if self._X is None or self._Y is None:
            raise ValueError("Dataset has not been processed yet. Call `process_images()` first.")
        torch.save({'X': self._X, 'Y': self._Y}, out_path)
        print(f"Dataset saved to {out_path}")

    def _valid_extensions(self):
        return [ext.lower() for ext in self._extensions]
    '''
    def _load_image(self, path):
        ext = Path(path).suffix.lower()

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=UserWarning)

                if ext in self.__rawExt:
                    with rawpy.imread(path) as raw:
                        rgb = raw.postprocess()
                    return Image.fromarray(rgb)
                else:
                    return Image.open(path).convert("RGB")

        except UserWarning as w:
            raise RuntimeError(f" {w}")
    '''

    def _load_image(self, path, format):
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

    def _resize_and_pad(self, img, width, height):
        img.thumbnail((width, height), Image.BILINEAR)
        new_img = Image.new(img.mode, (width, height))  # Respect mode ("L" or "RGB")
        paste_pos = ((width - img.width) // 2, (height - img.height) // 2)
        new_img.paste(img, paste_pos)
        return new_img

    '''
    def _resize_and_pad(self, img, width, height):
        img.thumbnail((width, height), Image.BILINEAR)
        new_img = Image.new("RGB", (width, height))
        paste_pos = ((width - img.width) // 2, (height - img.height) // 2)
        new_img.paste(img, paste_pos)
        return new_img
'''
    def get_dataset(self):
        return self._X, self._Y

    
    def preview_images(self, max_images=6):
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

            # Handle grayscale (1 channel) vs RGB (3 channel)
            if img_tensor.shape[0] == 1:
                # Show grayscale image with correct colormap
                plt.imshow(img_tensor.squeeze(0), cmap='gray')
            else:
                img = to_pil(img_tensor)
                plt.imshow(img)

            plt.title(label)
            plt.axis('off')

        plt.suptitle("Preview of Random Processed Images", fontsize=14)
        plt.tight_layout()
        plt.show()
    '''
    def preview_images(self, max_images=6):
        if self._X is None or self._Y is None:
            raise ValueError("Dataset not processed. Call `process_images()` first.")

        n = min(max_images, len(self._X))
        indices = torch.randperm(len(self._X))[:n]  # Torch-based random shuffle

        to_pil = transforms.ToPILImage()
        plt.figure(figsize=(n * 2, 2.5))

        for i, idx in enumerate(indices):
            img_tensor = self._X[idx]
            label = self._classDir[self._Y[idx]]
            img = to_pil(img_tensor)

            plt.subplot(1, n, i + 1)
            plt.imshow(img)
            plt.title(label)
            plt.axis('off')

        plt.suptitle("Preview of Random Processed Images", fontsize=14)
        plt.tight_layout()
        plt.show()
'''

# Example usage:# img_imp = ImgImp(baseDir='path/to/data', classDir=['class1', 'class2'], imageSize=(128, 128))
# img_imp.collect_images()
# img_imp.shuffle_images()
# img_imp.process_images()
# img_imp.save_dataset('dataset.pth')
# X, Y = img_imp.get_dataset()
# print(img_imp)
# print(f"X shape: {X.shape}, Y shape: {Y.shape}")

