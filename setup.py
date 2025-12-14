from setuptools import setup, find_packages

setup(
    name="forest-cover-predictor",
    version="1.0.0",
    author="Forest Analytics Team",
    author_email="forest-analytics@example.com",
    description="A machine learning system for predicting forest cover types",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/forest-analytics/forest-cover-predictor",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
    ],
    entry_points={
        'console_scripts': [
            'forest-predict=forest_cover_predictor:main',
        ],
    },
)