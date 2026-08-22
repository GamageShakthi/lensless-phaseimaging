from setuptools import setup, find_packages

setup(
    name="lensless-phaseimaging",
    version="0.1.0",
    description="Physics-Informed Deep Inverse Reconstruction for Lensless Phase-Mask Imaging",
    author="Student Researcher",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "opencv-python>=4.7.0",
        "matplotlib>=3.7.0",
        "Pillow>=9.5.0",
        "PyYAML>=6.0",
    ],
)
