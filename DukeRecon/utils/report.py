"""Make reports."""
import os
import pdb
import sys
from typing import Any, Dict
from git.repo import Repo
import numpy as np
import pdfkit

sys.path.append("..")
from utils import constants

PDF_OPTIONS = {
    "page-width": 300,
    "page-height": 150,
    "margin-top": 1,
    "margin-right": 0.1,
    "margin-bottom": 0.1,
    "margin-left": 0.1,
    "dpi": 300,
    "encoding": "UTF-8",
    "enable-local-file-access": None,
}


def get_git_branch() -> str:
    """Get the current git branch.

    Returns:
        str: current git branch, if not in git repo, return "unknown"
    """
    try:
        return Repo("./").active_branch.name
    except:
        return "unknown"


def format_dict(dict_stats: Dict[str, Any]) -> Dict[str, Any]:
    """Format dictionary for report.

    Rounds values to 2 decimal places.
    Args:
        dict_stats (Dict[str, Any]): dictionary of statistics
    Returns:
        Dict[str, Any]: formatted dictionary
    """
    list_round_3 = [constants.StatsIOFields.RBC_M_RATIO]
    for key in dict_stats.keys():
        if isinstance(dict_stats[key], float) and key in list_round_3:
            dict_stats[key] = np.round(dict_stats[key], 3)
        elif isinstance(dict_stats[key], float) and key not in list_round_3:
            dict_stats[key] = np.round(dict_stats[key], 2)
    return dict_stats


def clinical(dict_stats: Dict[str, Any], path: str):
    """Make clinical report with colormap images.

    First converts dictionary to html format. Then saves to path.
    Args:
        dict_stats (Dict[str, Any]): dictionary of statistics
        path (str): path to save report
    """
    dict_stats = format_dict(dict_stats)
    current_path = os.path.dirname(__file__)
    path_clinical = os.path.abspath(
        os.path.join(current_path, os.pardir, "assets", "html", "clinical.html")
    )
    path_html = os.path.join("tmp", "clinical.html")
    # write report to html
    with open(path_clinical, "r") as f:
        file = f.read()
        rendered = file.format(**dict_stats)
    with open(path_html, "w") as o:
        o.write(rendered)
    # write clinical report to pdf
    pdfkit.from_file(path_html, path, options=PDF_OPTIONS)


def grayscale(dict_stats: Dict[str, Any], path: str):
    """Make clinical report with grayscale images.

    First converts dictionary to html format. Then saves to path.
    Args:
        dict_stats (Dict[str, Any]): dictionary of statistics
        path (str): path to save report
    """
    dict_stats = format_dict(dict_stats)
    current_path = os.path.dirname(__file__)
    path_clinical = os.path.abspath(
        os.path.join(current_path, os.pardir, "assets", "html", "grayscale.html")
    )
    path_html = os.path.join("tmp", "grayscale.html")
    # write report to html
    with open(path_clinical, "r") as f:
        file = f.read()
        rendered = file.format(**dict_stats)
    with open(path_html, "w") as o:
        o.write(rendered)
    # write clinical report to pdf
    pdfkit.from_file(path_html, path, options=PDF_OPTIONS)


def intro(dict_info: Dict[str, Any], path: str):
    """Make info report.

    First converts dictionary to html format. Then saves to path.
    Args:
        dict_info (Dict[str, Any]): dictionary of statistics
        path (str): path to save report
    """
    dict_info = format_dict(dict_info)
    current_path = os.path.dirname(__file__)
    path_clinical = os.path.abspath(
        os.path.join(current_path, os.pardir, "assets", "html", "intro.html")
    )
    path_html = os.path.join("tmp", "intro.html")
    # write report to html
    with open(path_clinical, "r") as f:
        file = f.read()
        rendered = file.format(**dict_info)
    with open(path_html, "w") as o:
        o.write(rendered)
    # write clinical report to pdf
    pdfkit.from_file(path_html, path, options=PDF_OPTIONS)


def qa(dict_stats: Dict[str, Any], path: str):
    """Make quality assurance report.

    First converts dictionary to html format. Then saves to path.
    Args:
        dict_info (Dict[str, Any]): dictionary of statistics
        path (str): path to save report
    """
    dict_stats = format_dict(dict_stats)
    current_path = os.path.dirname(__file__)
    path_clinical = os.path.abspath(
        os.path.join(current_path, os.pardir, "assets", "html", "qa.html")
    )
    path_html = os.path.join("tmp", "qa.html")
    # write report to html
    with open(path_clinical, "r") as f:
        file = f.read()
        rendered = file.format(**dict_stats)
    with open(path_html, "w") as o:
        o.write(rendered)
    # write clinical report to pdf
    pdfkit.from_file(path_html, path, options=PDF_OPTIONS)
