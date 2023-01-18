"""ojd_daps_skills."""
from pathlib import Path
from setuptools import find_packages
from setuptools import setup

import os
import subprocess


tag_cmd = "echo $(git describe --tags --abbrev=0)"
version = subprocess.check_output(tag_cmd, shell=True).decode('ascii').replace('\n', '')

def read_lines(path):
    """Read lines of `path`."""
    with open(path) as f:
        return f.read().splitlines()


BASE_DIR = Path(__file__).parent


setup(
    name="ojd_daps_skills",
    long_description=open(os.path.join(BASE_DIR, "README.md")).read(),
    long_description_content_type="text/markdown",
    install_requires=read_lines(os.path.join(BASE_DIR, "requirements.txt")),
    packages=find_packages(exclude=["docs"]),
    package_data={
        # If any package contains *.yaml files, include them:
        "": [
            "*.yaml",
        ],
    },
    version=version,
    description="Extract skills from job ads and maps them onto a skills taxonomy of your choice.",
    url="https://github.com/nestauk/ojd_daps_skills",
    author="Nesta",
    author_email="dataanalytics@nesta.org.uk",
    maintainer="Nesta",
    maintainer_email="dataanalytics@nesta.org.uk",
    license="MIT",
)
