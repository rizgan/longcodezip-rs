from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="longcodezip",
    version="0.1.0",
    author="Yuling Shi",
    author_email="",
    description="A novel two-stage long code compression method for code language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YerbaPage/LongCodeZip",
    packages=find_packages(exclude=["repo-qa", "long-code-completion", "module-summarization", "assets"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
        ],
    },
    include_package_data=True,
    package_data={
        "longcodezip": ["py.typed"],
    },
)

