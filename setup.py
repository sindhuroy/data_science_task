# setup.py
from setuptools import setup, find_packages

setup(
    name="word2vec_distance",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "gensim",
        "scikit-learn",
    ],
)
