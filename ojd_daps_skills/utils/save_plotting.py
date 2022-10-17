"""
utils.save_plotting
Utils for easier exporting of altair charts
"""
from ojd_daps_skills import PROJECT_DIR

from altair_saver import save
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import os
from typing import Iterator
from pathlib import Path


FIGURE_PATH = Path(f"{PROJECT_DIR}/outputs/figures")
DEFAULT_FILETYPES = ["png", "svg", "html"]


def google_chrome_driver_setup():
    """Set up the driver to save figures"""
    driver = webdriver.Chrome(ChromeDriverManager().install())
    return driver


def create_paths(
    path: os.PathLike = FIGURE_PATH, filetypes: Iterator[list] = DEFAULT_FILETYPES
):
    """Checks if the paths exist and if not creates them"""
    for filetype in filetypes:
        os.makedirs(f"{path}/{filetype}", exist_ok=True)


def save_png(fig, path: os.PathLike, name: str, driver):
    """Save altair chart as a  raster png file"""
    save(
        fig,
        f"{path}/png/{name}.png",
        method="selenium",
        webdriver=driver,
        scale_factor=5,
    )


def save_html(fig, path: os.PathLike, name: str):
    """Save altair chart as an html file"""
    fig.save(f"{path}/html/{name}.html")


def save_svg(fig, path: os.PathLike, name: str, driver):
    """Save altair chart as a vector svg file"""
    save(fig, f"{path}/svg/{name}.svg", method="selenium", webdriver=driver)


class AltairSaver:
    """
    Class helping to easily save altair charts
    """

    def __init__(
        self,
        path: os.PathLike = FIGURE_PATH,
        filetypes: Iterator[list] = DEFAULT_FILETYPES,
    ):
        self.driver = google_chrome_driver_setup()
        self.path = path
        self.filetypes = filetypes

    def save(
        self, fig, name: str, path: os.PathLike = None, filetypes: Iterator[list] = None
    ):
        """
        Saves an altair figure in multiple formats (png, html and svg files)
        Args:
            fig: altair chart
            name: name to save the figure
            driver: webdriver
            path: path where to save the figure
            filetype: list of filetypes, eg: ['png', 'svg', 'html']
        """
        # Default values
        path = self.path if path is None else path
        filetypes = self.filetypes if filetypes is None else filetypes
        # Check paths
        create_paths(path, filetypes)
        # Export figures
        if "png" in filetypes:
            save_png(fig, path, name, self.driver)
        if "html" in filetypes:
            save_html(fig, path, name)
        if "svg" in filetypes:
            save_svg(fig, path, name, self.driver)
