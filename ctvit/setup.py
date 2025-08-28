from setuptools import setup, find_packages

setup(
  name = 'ctvit',
  install_requires=[
    'accelerate',
    'beartype',
    'einops>=0.6',
    'ema-pytorch>=0.2.2',
    'opencv-python',
    'pillow',
    'numpy',
    'sentencepiece',
    'torch',
    'torchtyping',
    'torchvision',
    'transformers',
    'tqdm',
    'vector-quantize-pytorch==1.1.2',
    'nibabel',
    'openpyxl',
    'pycocoevalcap',
    'pandas',
    'click',
    'appdirs',

  ],
)
