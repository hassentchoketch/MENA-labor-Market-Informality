"""
This file is used to make the entire project a package.

By including this file, Python recognizes the directory as a package,
allowing modules within the directory to be imported and used elsewhere
in the project.
"""
from setuptools import setup, find_packages
from typing import List
HYPEN = "-e ."

def get_requirements(file_path: str ) -> list[str]:
    """
    Reads the requirements.txt file and returns a list of required packages.
    """
    requirements = []
    with open(file_path, "r") as f:
        requirements = f.readlines()
        # Remove any leading/trailing whitespace characters from each line and filter out empty lines
        requirements= [req.replace("\n","") for req in requirements]
        
        if HYPEN in requirements:
            requirements.remove(HYPEN)
    
    return requirements
       
setup(
    name="informality_classifier",  # Replace with your project name
    version="0.0.1",  # Replace with your project version
    author="Hacene Tchoketch-kebir",  # Replace with your name
    author_email="hassentchoketch@gmail.com",
    description="A package for classifying labor market informalty in MENA Region.",  # Replace with your project description
    packages=find_packages(),  # Automatically find and include all packages in the directory
    install_requires= get_requirements("requirments.txt"),  # List of dependencies required for the project
)