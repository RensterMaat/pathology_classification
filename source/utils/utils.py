import os
import h5py
import yaml
from pathlib import Path
import numpy as np
from openslide import OpenSlide


def get_slide(slide_id: str, slides_dir: os.PathLike) -> OpenSlide:
    slide_file_path = find_slide_file_path(slide_id, slides_dir)
    return OpenSlide(slide_file_path)


def get_coordinates(slide_id: str, patch_coordinates_dir: os.PathLike) -> np.array:
    coordinates_file_path = find_coordinates_file_path(slide_id, patch_coordinates_dir)
    return np.array(h5py.File(coordinates_file_path, "r")["coords"])


def find_slide_file_path(slide_id: str, slides_dir: os.PathLike) -> str:
    for dir, _, files in os.walk(slides_dir):
        for file in files:
            if ".".join(file.split(".")[:-1]) == slide_id:
                return str(Path(dir) / file)


def find_coordinates_file_path(
    slide_id: str, patch_coordinates_dir: os.PathLike
) -> str:
    root = Path(patch_coordinates_dir)
    return str(root / slide_id / (slide_id + ".h5"))


def scale_coordinates(coordinates: np.array, slide: str, level: int) -> np.array:
    scaling_factor = slide.level_dimensions[0][0] / slide.level_dimensions[level][0]
    return coordinates / scaling_factor


def load_config(config_path: str | os.PathLike) -> dict:
    default_config_path = Path(config_path).parent / "default.yaml"

    with open(default_config_path) as f:
        config = yaml.safe_load(f)

    with open(config_path) as f:
        config.update(yaml.safe_load(f))

    for directory in [
        "features",
        "slides",
        "patch_coordinates",
        "segmentations",
        "cross_val_splits",
        "output",
    ]:
        if not config[f"{directory}_dir"]:
            config[f"{directory}_dir"] = Path(config["dataset_dir"]) / directory

    return config

def get_patch_coordinates_dir_name(config):
    return f"extraction_level={config['extraction_level']}_patch_dimensions={config['patch_dimensions']}"