import setuptools


setuptools.setup(
    name="deepprofiler",
    version="0.1.0",
    author="Juan Caicedo",
    author_email="jcaicedo@gmail.com",
    description="Tools for representation learning in high throughput image collections",
    license="BSD",
    keywords="",
    url="https://github.com/jccaicedo/DeepProfiler",
    packages=["deepprofiler"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License"
    ],
    install_requires=[
        "beautifulsoup4>=4.6",
        "click>=6.7",
        "comet_ml>=1.0",
        "gpyopt>=1.2"
        "keras>=2.2.4",
        "keras_resnet>=0.1.0",
        "lxml>=4.2",
        "numpy>=1.13",
        "pandas>=0.23.0",
        "scikit-image>=0.14.0",
        "scikit-learn>=0.19.0",
        "scipy>=1.1"
    ],
    extras_require={
        "cpu": [
            "tensorflow==1.15.0"
        ],
        "gpu": [
            "tensorflow-gpu==1.15.0"
        ],
        "test": [
            "pytest>=3.6",
            "pytest-cov>=2.0",
            "codecov>=2.0"
        ]
    }
)
