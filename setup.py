"""ojd_daps_skills."""
from pathlib import Path
from setuptools import find_packages
from setuptools import setup


def read_lines(path):
    """Read lines of `path`."""
    with open(path) as f:
        return f.read().splitlines()


BASE_DIR = Path(__file__).parent


setup(
    name="ojd_daps_skills",
    long_description=open(BASE_DIR / "README.md").read(),
    install_requires=read_lines(BASE_DIR / "requirements.txt"),
    extras_require={"dev": read_lines(BASE_DIR / "requirements_dev.txt")},
    packages=find_packages(exclude=["docs"]),
    package_data = {
        # If any package contains *.yaml files, include them:
        '': [ '*.yaml', ],
    },
    version="0.1.0",
    description="Improved skills extraction algorithm for OJO",
    author="Nesta",
    license="MIT",
)
