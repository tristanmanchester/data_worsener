from setuptools import setup, find_packages

setup(
    name="data_worsener",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'cupy-cuda11x',
        'h5py',
        'tqdm',
        'scikit-image',
        'opencv-python',
        'matplotlib'
    ],
    scripts=[
        'scripts/worsen_data.py',
        'scripts/segment.py'
    ]
)