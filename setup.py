from setuptools import find_packages, setup

setup(
    name="aerosonicdb",
    version="1.0.0",
    description="Download and get started with the AeroSonicDB-YPAD0523 audio dataset of low-flying aircraft.",
    author="Blake Downward",
    author_email="aerosonicdb@gmail.com",
    url="https://github.com/aerosonicdb/AeroSonicDB-YPAD0523",
    packages=find_packages(),
    install_requires=[
        "click",
        "librosa",
        "tensorflow",
        "scikit-learn",
        "scikeras",
        "matplotlib",
        "pandas",
        "jupyter",
        "notebook",
        "tqdm",
    ],
)
