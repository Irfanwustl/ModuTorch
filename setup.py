from setuptools import setup, find_packages

setup(
    name='modutorch',  # The name of your package
    version='0.1.0',  # Version number, follow semantic versioning
    description='A modular deep learning framework built using PyTorch',
    author='Irfan Alahi',  # Your name
    author_email='irfanwustl@gmail.com',
    url='https://github.com/Irfanwustl/ModuTorch',  # GitHub repository link
    packages=find_packages(),  # Automatically find and include all packages in modu_torch
    install_requires=[
        'matplotlib>=3.5.0,<3.9',  # Relaxed version constraints for better compatibility
        'numpy>=1.21.0,<2.0.0',
        'Pillow>=10.0.0',  # Allow newer versions if possible
        'scikit-learn>=1.5.0,<1.6',  # Compatible range for scikit-learn
        'seaborn>=0.13.0,<0.14',
        'torch>=2.0.0,<2.5.0',  # Allow more flexibility for PyTorch versions
        'torchvision>=0.17.0,<0.18',
        'tqdm>=4.60.0',  # Allow newer versions if possible
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Operating System :: OS Independent',
        'Framework :: PyTorch',
    ],
    python_requires='>=3.9',
    license='MIT',
)
