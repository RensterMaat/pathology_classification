import os
import h5py
from pathlib import Path
import numpy as np
from openslide import OpenSlide


def get_slide(slide_id: str, slide_dir: os.PathLike) -> OpenSlide:
    slide_file_path = find_slide_file_path(slide_id, slide_dir)
    return OpenSlide(slide_file_path)


def get_coordinates(slide_id: str, patch_coordinate_dir: os.PathLike) -> np.array:
    coordinates_file_path = find_coordinates_file_path(slide_id, patch_coordinate_dir)
    return np.array(h5py.File(coordinates_file_path, "r")["coords"])


def find_slide_file_path(slide_id: str, slide_dir: os.PathLike) -> str:
    for dir, _, files in os.walk(slide_dir):
        for file in files:
            if ".".join(file.split(".")[:-1]) == slide_id:
                return str(Path(dir) / file)


def find_coordinates_file_path(slide_id: str, patch_coordinate_dir: os.PathLike) -> str:
    root = Path(patch_coordinate_dir)
    return str(root / slide_id / (slide_id + ".h5"))
