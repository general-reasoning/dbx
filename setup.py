import logging
import os
import setuptools


def readlines(fname: str) -> list:
    """Load a local filename."""
    lines = open(os.path.join(os.path.dirname(__file__), fname)).readlines()
    return [line.strip() for line in lines]


def get_requirements():
    requirements = '\n'.join(readlines('requirements.txt'))
    return requirements
    

setuptools.setup(
    name="dbx",
    version="0.0.3",
    author="Dmitry Karpeyev",
    author_email="dmitry.karpeyev@gmail.com",
    description="Data experiment management hub",
    packages=setuptools.find_packages(exclude=("tests",)),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    entry_points={'console_scripts': [
        'dbx=dbx:exec',
        'dbx.exec=dbx:exec',
        'dbx.eval=dbx:eval',
        'dbx.print=dbx:pprint',
        'dbx.pprint=dbx:pprint',
    ]},
    python_requires='>=3.7', 
    install_requires=get_requirements(),
)
