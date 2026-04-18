from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="shard-db",
    version="0.1.0",
    description="Scalable Hash-Addressed Retrieval Database — TB-scale storage on minimal RAM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="SHARD Contributors",
    license="Apache-2.0",
    python_requires=">=3.9",
    packages=find_packages(exclude=["tests*", "examples*", "docs*"]),
    install_requires=[
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0"],
    },
    entry_points={
        "console_scripts": [
            "shard=shard.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Database",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
