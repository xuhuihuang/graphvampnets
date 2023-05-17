from setuptools import setup, find_packages
import re

install_requires = ["numpy","scipy","torch","torchvision","tqdm","torch-scatter","torch-sparse","torch-spline-conv","torch-cluster","torch-geometric"]
_extras_require = ["matplotlib","scikit-learn","jupyter"]
extras_require = {b: a for a, b in (re.findall(r"^(([^!=<>~]+)(?:[!=<>~].*)?$)", x)[0] for x in _extras_require)}

CLASSIFIERS = """\
Development Status :: 3 - Alpha
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: GPL-3.0 License
Operating System :: OS Independent
Programming Language :: Python :: 3
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8
Programming Language :: Python :: 3.9
Topic :: Scientific/Engineering
Topic :: Artificial Intelligence
"""

setup(
    name ='graphvampnets',
    version = '1.0.0',
    python_requires = '>=3.7.0',
    install_requires = install_requires,
    extras_require = extras_require,
    description = 'GraphVAMPnets for self-assembly kinetics',
    long_description = open("README.md", "r", encoding="utf-8").read(),
    license = 'GPL-3.0 License',
    author = 'Bojun Liu',
    author_email = 'bliu293@wisc.edu',
    url = 'https://github.com/bojunliu0818/graphvampnets',
    packages = find_packages(),
    classifiers = CLASSIFIERS.splitlines()
)

