import os

# Assuming you are running this script from the project_root directory
# If not, uncomment and adjust the line below:
# os.chdir('/path/to/your/project_root')

# List of directories to create (relative to project_root)
directories = [
    'data/raw',
    'data/processed',
    'data/outputs',
    'notebooks',
    'src',
    'models',
    'docs'
]

# Create directories
for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"Created directory: {directory}")

# Create empty files for requirements.txt and .gitignore
files = ['requirements.txt', '.gitignore']
for file in files:
    with open(file, 'w') as f:
        pass  # Creates an empty file
    print(f"Created empty file: {file}")

# Optionally, add a basic .gitignore content
gitignore_content = """# Data files
data/raw/*
data/processed/*
data/outputs/*

# Models and outputs
models/*
outputs/*

# Environment
__pycache__/
*.pyc
.env

# IDE
.vscode/
.idea/

# Jupyter
.ipynb_checkpoints/
"""
with open('.gitignore', 'w') as f:
    f.write(gitignore_content)
print("Added basic content to .gitignore")

# For the notebooks, you can create empty .ipynb files if desired
# But Jupyter notebooks are JSON, so here's a minimal template
notebook_template = {
    "cells": [],
    "metadata": {"kernelspec": {"name": "python3"}},
    "nbformat": 4,
    "nbformat_minor": 5
}

notebook_files = [
    'notebooks/01_data_collection.ipynb',
    'notebooks/02_preprocessing.ipynb',
    'notebooks/03_modeling.ipynb',
    'notebooks/04_mapping.ipynb'
]

import json
for nb_file in notebook_files:
    os.makedirs(os.path.dirname(nb_file), exist_ok=True)
    with open(nb_file, 'w') as f:
        json.dump(notebook_template, f, indent=2)
    print(f"Created empty notebook: {nb_file}")

print("Project structure setup complete! Run 'jupyter lab' or 'code .' to open.")