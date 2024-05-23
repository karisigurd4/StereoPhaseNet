from setuptools import find_packages, setup

setup(
    name='stereophase-net',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'librosa',
        'torch',
        'transformers',
        'soundfile',
    ],
    entry_points={
        'console_scripts': [
            'preprocess_data=scripts.preprocess_data:main',
            'run_inference=scripts.run_inference:main',
            'train_model=scripts.train_model:main',
        ],
    },
)