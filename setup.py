import json
import argparse
import click
from setuptools import setup, find_packages

with open("requirements.txt", "r") as _f:
    requirements = [line for line in _f.read().split("\n")]

setup(
    name="vserranolvl",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "vserranolvl=ml-models.scratch.models:sequential_model",
        ],
    },
    version="0.0.001",
    description="MLProjects",
    author="vserranlvl, qirolabs",
)

