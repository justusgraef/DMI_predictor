from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="dmi-predictor",
    version="1.0.0",
    author="Chop Yan Lee, Justus Graef",
    description="A tool to predict domain-motif interfaces (DMI) in protein-protein interactions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/justusgraef/DMI_predictor",
    packages=find_packages(),
    package_data={
        "dmi_predictor": [
            "data/*.tsv",
            "data/*.txt",
            "data/*.joblib",
            "data/*.json",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        'aiupred': [
            'git+https://github.com/doszilab/AIUPred.git@main#egg=aiupred_lib'
        ],
    },
    entry_points={
        "console_scripts": [
            "dmi-predict=dmi_predictor.cli.main:cli",
        ],
    },
)
