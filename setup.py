from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.1'
DESCRIPTION = 'indexpaper - library to index papers with vector databases'
LONG_DESCRIPTION = 'indexpaper - library to index papers with vector databases'

# Setting up
setup(
    name="pywhy-llm",
    version=VERSION,
    author="Emre Kıcıman, Rose De Sicilia",
    author_email="<emrek@microsoft.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['python-dotenv', "openai", "guidance", "dowhy", "matplotlib", "networkx", "more-itertools"],
    keywords=['python', 'causal analysis', 'llm'],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
