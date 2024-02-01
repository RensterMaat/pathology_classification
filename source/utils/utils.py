import os

# import h5py
import yaml
from pathlib import Path
import numpy as np
from openslide import OpenSlide
from pathlib import Path


def get_slide(slide_id: str, slides_dir: os.PathLike) -> OpenSlide:
    """
    Get the slide corresponding to the slide ID.

    Args:
        slide_id (str): Slide ID.
        slides_dir (os.PathLike): Path to the directory containing the slides.

    Returns:
        OpenSlide: Slide corresponding to the slide ID.
    """
    slide_file_path = find_slide_file_path(slide_id, slides_dir)
    return OpenSlide(slide_file_path)


# def get_coordinates(slide_id: str, patch_coordinates_dir: os.PathLike) -> np.array:
#     """
#     Get the coordinates of the patches of the slide corresponding to the slide ID.

#     Args:
#         slide_id (str): Slide ID.
#         patch_coordinates_dir (os.PathLike): Path to the directory containing the patch

#     Returns:
#         np.array: Coordinates of the patches of the slide corresponding to the slide ID.
#     """
#     coordinates_file_path = find_coordinates_file_path(slide_id, patch_coordinates_dir)
#     return np.array(h5py.File(coordinates_file_path, "r")["coords"])


def get_coordinates(*args):
    raise NotImplementedError


def get_dict_of_slide_ids_vs_paths(slides_dir: os.PathLike) -> dict:
    """
    Get a dictionary of slide IDs versus paths to the slides.

    Currently only searches for .ndpi, .svs and .tif files.

    Args:
        slides_dir (os.PathLike): Path to the directory containing the slides.

    Returns:
        dict: Dictionary of slide IDs versus paths to the slides.
    """
    slide_ids_vs_paths = {}
    for root, _, files in os.walk(slides_dir):
        slide_ids_vs_paths.update(
            {
                ".".join(file.split(".")[:-1]): str(Path(root) / file)
                for file in files
                if file.split(".")[-1] in ["ndpi", "svs", "tif"]
            }
        )

    return slide_ids_vs_paths


def list_all_slide_file_paths(slides_dir: os.PathLike) -> list:
    """
    Recursively lists the paths of all histopathology slide files in a given directory.

    Currently only searches for .ndpi, .svs and .tif files.

    Args:
        slides_dir (os.PathLike): directory to be searched for histopathology slide files

    Returns:
        list: list of the paths all histopathology slide files in slides_dir
    """
    all_file_paths = []
    for root, _, files in os.walk(slides_dir):
        slide_file_paths = [
            Path(root) / file
            for file in files
            if file.split(".")[-1] in ["ndpi", "svs", "tif"]
        ]
        all_file_paths.extend(slide_file_paths)

    return all_file_paths


def find_slide_file_path(slide_id: str, slides_dir: os.PathLike) -> str:
    """
    Find the path to the slide corresponding to the slide ID.

    Args:
        slide_id (str): Slide ID.
        slides_dir (os.PathLike): Path to the directory containing the slides.

    Returns:
        str: Path to the slide corresponding to the slide ID.
    """
    for dir, _, files in os.walk(slides_dir):
        for file in files:
            if ".".join(file.split(".")[:-1]) == slide_id:
                return str(Path(dir) / file)


def find_coordinates_file_path(
    slide_id: str, patch_coordinates_dir: os.PathLike
) -> str:
    """
    Find the path to the coordinates of the patches of the slide corresponding to the slide ID.

    Args:
        slide_id (str): Slide ID.
        patch_coordinates_dir (os.PathLike): Path to the directory containing the patch coordinates.

    Returns:
        str: Path to the coordinates of the patches of the slide corresponding to the slide ID.
    """
    root = Path(patch_coordinates_dir)
    return str(root / slide_id / (slide_id + ".h5"))


def scale_coordinates(coordinates: np.array, slide: OpenSlide, level: int) -> np.array:
    """
    Scale the coordinates of the patches from the extraction level to the level of the slide.

    Args:
        coordinates (np.array): Coordinates of the patches at the extraction level.
        slide (OpenSlide): Slide of which the patches are extracted.
        level (int): Level of the slide.

    Returns:
        np.array: Coordinates of the patches at the level of the slide.
    """
    scaling_factor = slide.level_dimensions[0][0] / slide.level_dimensions[level][0]
    return coordinates / scaling_factor


def load_config(config_path: str | os.PathLike) -> dict:
    """
    Load the configuration file.

    Defaults to the default configuration file for keys not present in the configuration file.
    If directories for features, slides, patch coordinates, segmentations or cross-validation splits are not specified,
    they are set to the default directories (e.g. <output_dir>/features).

    Args:
        config_path (str | os.PathLike): Path to the configuration file.

    Returns:
        dict: Configuration file.
    """
    default_config_path = Path(config_path).parent / "default.yaml"

    with open(default_config_path) as f:
        config = yaml.safe_load(f)

    with open(config_path) as f:
        config.update(yaml.safe_load(f))

    Path(config["output_dir"]).mkdir(exist_ok=True)

    return config


def get_patch_coordinates_dir_name(config: dict) -> os.PathLike:
    """
    Get the name of the directory containing the patch coordinates.

    Args:
        config (dict): Configuration file.

    Returns:
        os.PathLike: Name of the directory containing the patch coordinates.
    """
    return (
        Path(config["output_dir"])
        / "patch_coordinates"
        / f"extraction_level={config['extraction_level']}_patch_dimensions={config['patch_dimensions']}"
    )


def get_features_dir_name(config):
    """
    Get the name of the directory containing the features.

    Args:
        config (dict): Configuration file.

    Returns:
        os.PathLike: Name of the directory containing the features.
    """
    return (
        Path(config["output_dir"])
        / "features"
        / f"extraction_level={config['extraction_level']}_extractor={config['extractor_model']}"
    )


def get_cross_val_splits_dir_path(config):
    folder_name = []
    for field in ["target", "seed"]:
        folder_name.append(f"{field}={config[field]}")

    if config["mode"] == "eval":
        folder_name.append(f"n_folds={config['n_folds']}")

    folder_name = "_".join(folder_name)

    return (
        Path(config["output_dir"]) / "cross_val_splits" / config["mode"] / folder_name
    )
