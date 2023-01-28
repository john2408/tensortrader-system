#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "Click>=7.0",
    "dnspython==2.1.0",
    "fastparquet==0.8.1" "imbalanced-learn==0.9.1",
    "keras==2.11.0",
    "keras-tcn==3.5.0",
    "matplotlib==3.5.2",
    "numpy==1.22.4",
    "oauthlib==3.2.2",
    "openpyxl==3.0.10",
    "pandas==1.4.2",
    "pandas-ta==0.3.14b0",
    "pyaml==21.10.1",
    "pyarrow==8.0.0",
    "pymongo==4.0",
    "python-binance==1.0.15",
    "python-dateutil==2.8.2",
    "pytz==2022.1",
    "PyWavelets==1.4.1",
    "PyYAML==6.0",
    "scikit-learn==1.1.1",
    "scikit-optimize==0.9.0",
    "scipy==1.8.1",
    "tensorflow==2.11.0",
    "xgboost==1.6.1",
]

test_requirements = ["pytest>=3"]

setup(
    author="John Torres",
    author_email="john.torres.tensor@gmail.com",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Tensor Algorithmic Trader",
    entry_points={
        "console_scripts": [
            "tensortrader=tensortrader.cli:main",
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="tensortrader",
    name="tensortrader",
    packages=find_packages(include=["tensortrader", "tensortrader.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/john2408/tensortrader",
    version="0.1.0",
    zip_safe=False,
)
