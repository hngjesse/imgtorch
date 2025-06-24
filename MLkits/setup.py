from setuptools import setup, find_packages

setup(
    name="MLkits",
    version="0.1.0",
    author="Jesse Hng",
    author_email="hngjesse@gmail.com",
    description="Image preprocessing and dataset organization toolkit for machine learning.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/hngjesse/MLkits",  # update later
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "Pillow",
        "tqdm",
        "rawpy",
        "matplotlib"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License"
    ],
    python_requires=">=3.8",
    include_package_data=True,
)
