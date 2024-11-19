from setuptools import setup, find_packages

setup(
    name="monash_processing",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'h5py',
        'tqdm',
        'astra-toolbox',
        'tifffile',
        'pyqtgraph',
        'pyamg',
        'scipy',
        'matplotlib',
        'scikit-image',
        'rasterio',
    ],
    author="Dominik John",
    author_email="dominik.john@monash.edu",
    description="Phase contrast processing pipeline for Monash",
    long_description=open('README.md').read() if open('README.md') else '',
    long_description_content_type="text/markdown",
)