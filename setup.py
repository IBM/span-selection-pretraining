from setuptools import find_packages, setup

setup(
    name="span-selection-pretraining",
    version="0.1",
    author="Michael Glass",
    author_email="mrglass@us.ibm.com",
    description="Generate training data for extending pretraining of neural language models",
    long_description=open("README.md", "r", encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["*.tests", "*.tests.*",
                                    "tests.*", "tests"]),
    install_requires=['torch>=0.4.1',
                      'numpy',
                      'ujson',
                      'regex'],
    python_requires='>=3.5.0',
    tests_require=['pytest'],
    classifiers=[
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
