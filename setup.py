"""ojd_daps_skills."""
from pathlib import Path
from setuptools import find_packages
from setuptools import setup
import setuptools_scm


def read_lines(path):
    """Read lines of `path`."""
    with open(path) as f:
        return f.read().splitlines()


BASE_DIR = Path(__file__).parent


setup(
    name="ojd_daps_skills",
    long_description=open(BASE_DIR / "README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    install_requires=read_lines(BASE_DIR / "requirements.txt"),
    extras_require={"dev": read_lines(BASE_DIR / "requirements_dev.txt")},
    packages=find_packages(
        exclude=["docs", "ojd_daps_skills/analysis", "ojd_daps_skills/app"]
    ),
    classifiers=["Development Status :: 5 - Production/Stable"],
    package_data={
        # If any package contains *.yaml files, include them:
        "": [
            "*.yaml",
        ],
    },
    # deal with pypi special characters
    version=".".join(setuptools_scm.get_version().split(".")[:-2]),
    description="Extract skills from job ads and maps them onto a skills taxonomy of your choice.",
    url="https://github.com/nestauk/ojd_daps_skills",
    project_urls={
        "Documentation": "https://nestauk.github.io/ojd_daps_skills/build/html/about.html",
        "Source": "https://github.com/nestauk/ojd_daps_skills",
    },
    author="Nesta",
    author_email="dataanalytics@nesta.org.uk",
    maintainer="Nesta",
    maintainer_email="dataanalytics@nesta.org.uk",
    license="MIT",
)
