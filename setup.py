from setuptools import setup, find_packages

setup(
    name='ImgTorch',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'Pillow',
        'rawpy',
        'tqdm',
        'matplotlib',
        'torchvision'
    ],
    author='Jesse Hng',
    description='A simple image import and processing library',
)
# This setup script is for the ImgTorch library, which provides functionality for importing and processing images.