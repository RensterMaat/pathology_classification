import os
import cv2
import json
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.filters import threshold_otsu
from skimage.measure import find_contours
from skimage.morphology import remove_small_holes
from matplotlib.colors import rgb_to_hsv
from scipy.ndimage import median_filter
from openslide import OpenSlide
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
from PIL import Image

from source.utils.utils import (
    get_patch_coordinates_dir_name,
)
from source.tessellate.tessellate_utils import tessellate


class Preprocessor:
    """
    Preprocessor class for extracting tiles from whole-slide images.

    Args:
        config: dictionary containing the configuration parameters.
    """

    def __init__(self, config: dict) -> None:
        self.config = config

        self.manifest = pd.read_csv(config["manifest_file_path"]).set_index("slide_id")

        self.patch_coordinates_save_dir_path = get_patch_coordinates_dir_name(config)
        self.patch_coordinates_save_dir_path.mkdir(parents=True, exist_ok=True)

        self.tile_images_save_dir_path = Path(
            config["output_dir"], "tiles", self.patch_coordinates_save_dir_path.name
        )
        self.tile_images_save_dir_path.mkdir(parents=True, exist_ok=True)

        self.segmentation_visualization_save_dir_path = Path(
            config["output_dir"],
            "segmentation_visualization",
            self.patch_coordinates_save_dir_path.name,
        )
        self.segmentation_visualization_save_dir_path.mkdir(parents=True, exist_ok=True)

    def __call__(
        self, slide_path: str | os.PathLike
    ) -> dict[int, list[tuple[tuple[int, int], tuple[int, int]]]]:
        """
        Extracts tiles from a whole-slide image.

        Defaults to using the Otsu thresholding method to create a segmentation if no
        segmentation is provided.

        Args:
            slide_path: path to the whole-slide image.
            segmentation_path: path to the segmentation of the whole-slide image.

        Returns:
            tiles: dictionary containing the tiles for each cross-section.
        """

        slide = OpenSlide(str(slide_path))

        # The preprocessing level is the level at which the tiles are extracted. It cannot be higher than the highest level of the slide.
        preprocessing_level = min(
            self.config["preprocessing_level"], len(slide.level_dimensions) - 1
        )

        # Load the whole-slide image at the preprocessing level
        img = np.array(
            slide.read_region(
                (0, 0),
                preprocessing_level,
                slide.level_dimensions[preprocessing_level],
            )
        )[:, :, :3]

        # create automatic segmentation and load all manual segmentations
        segmentations = {}
        segmentations["automatic"] = self.create_segmentation(img)
        for segmentation_path_column_name in self.config[
            "segmentation_path_column_name"
        ]:
            segmentation_path = self.manifest.loc[
                Path(slide_path).stem, segmentation_path_column_name
            ]
            segmentations[segmentation_path_column_name] = self.load_segmentation(
                segmentation_path
            )

        # create union of all segmentations. This union is used for tessalation. Subsets of the extracted tiles are used later during classification.
        union_of_all_segmentations = np.zeros(
            (
                slide.level_dimensions[self.config["preprocessing_level"]][1],
                slide.level_dimensions[self.config["preprocessing_level"]][0],
            ),
            dtype=np.uint8,
        )
        for segmentation in segmentations.values():
            union_of_all_segmentations = np.logical_or(
                union_of_all_segmentations, segmentation[:, :, 0]
            )

        # Calculate the scaling factor between the preprocessing level and the extraction level
        scaling_factor = (
            slide.level_downsamples[preprocessing_level]
            / slide.level_downsamples[self.config["extraction_level"]]
        )

        # Extract tiles from the whole-slide image based on the union of all segmentations
        tile_coordinates = tessellate(
            union_of_all_segmentations, self.config, scaling_factor
        )

        # Scale the tile coordiantes to the highest magnification level
        scaled_tile_coordinates = self.scale_tiles(
            tile_coordinates, slide.level_downsamples[preprocessing_level]
        )

        if not tile_coordinates:
            raise ValueError("No tiles were found.")

        # Save coordinates, extracted images and visualization of the segmentations (automatic and manual)
        self.save_tile_coordinates(scaled_tile_coordinates, slide_path)
        self.save_tile_images(scaled_tile_coordinates, slide, slide_path)
        self.save_segmentation_visualization(
            img,
            union_of_all_segmentations,
            tile_coordinates,
            scaling_factor,
            slide_path,
        )

        return scaled_tile_coordinates

    def load_segmentation(
        self, slide: OpenSlide, segmentation_path: str | os.PathLike
    ) -> np.ndarray:
        with open(segmentation_path, "r") as f:
            annotation = json.load(f)

        segmentation = np.zeros(
            (
                slide.level_dimensions[self.config["preprocessing_level"]][1],
                slide.level_dimensions[self.config["preprocessing_level"]][0],
            ),
            dtype=np.uint8,
        )

        for annotation_part in annotation["features"]:
            if annotation_part["geometry"]["type"] == "Polygon":
                coordinates = (
                    np.array(annotation_part["geometry"]["coordinates"][0])
                    / slide.level_downsamples[self.config["preprocessing_level"]]
                ).astype(np.int32)

                cv2.fillPoly(
                    segmentation,
                    [coordinates],
                    1,
                )

            return segmentation[:, :, np.newaxis]

    def save_tile_coordinates(
        self,
        tile_coordinates: dict[int, list[tuple[tuple[int, int], tuple[int, int]]]],
        slide_path: str | os.PathLike,
    ) -> None:
        """
        Saves the tile coordinates to a JSON file.

        Args:
            tiles: dictionary containing the tiles for each cross-section.
            slide_path: path to the whole-slide image.

        """
        for cross_section in tile_coordinates:
            slide_name = Path(slide_path).stem
            cross_section_name = f"{slide_name}_cross_section_{cross_section}.json"

            cross_section_save_path = (
                self.patch_coordinates_save_dir_path / cross_section_name
            )

            with open(cross_section_save_path, "w") as f:
                json.dump(tile_coordinates[cross_section], f)

    def save_tile_images(
        self,
        tile_coordinates: dict[int, list[tuple[tuple[int, int], tuple[int, int]]]],
        slide: OpenSlide,
        slide_path: str | os.PathLike,
    ) -> None:
        """
        Saves the tiles to a directory.

        Args:
            tile_coordinates: dictionary containing the tiles for each cross-section.
            slide: whole-slide image.
            slide_path: path to the whole-slide image.
        """
        for cross_section, coordinates in tile_coordinates.items():
            slide_name = Path(slide_path).stem
            cross_section_name = f"{slide_name}_cross_section_{cross_section}"

            cross_section_save_dir = self.tile_images_save_dir_path / cross_section_name
            cross_section_save_dir.mkdir(exist_ok=True)

            if self.config["num_workers"] == 1:
                coordinates = tqdm(
                    coordinates,
                    desc=f"Saving patches for {slide_name}",
                    unit="patches",
                    leave=False,
                )

            for (_, _), (x, y), (width, height) in coordinates:
                tile = slide.read_region(
                    (x, y),
                    self.config["extraction_level"],
                    (width, height),
                )
                tile_rgb = Image.fromarray(np.array(tile)[:, :, :3])
                tile_rgb.save(cross_section_save_dir / f"{x}_{y}.jpg")

    def save_segmentation_visualization(
        self,
        image: np.ndarray,
        segmentation: np.ndarray,
        tile_coordinates: dict[int, list[tuple[tuple[int, int], tuple[int, int]]]],
        scaling_factor: float,
        slide_path: str | os.PathLike,
    ) -> None:
        """
        Saves the segmentation visualization to a directory.

        Args:
            segmentation: segmentation of the whole-slide image.
            tile_coordinates: dictionary containing the tiles for each cross-section.
            slide_path: path to the whole-slide image.
        """
        slide_name = Path(slide_path).stem

        contours = find_contours(segmentation[:, :, 0], 0.5)

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(image)
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
        ax.axis("off")

        visualization_save_path = (
            self.segmentation_visualization_save_dir_path / f"{slide_name}.jpg"
        )
        fig.savefig(visualization_save_path)

    def scale_tiles(
        self,
        tiles: dict[int, list[tuple[tuple[int, int], tuple[int, int]]]],
        scaling_factor: float,
    ) -> dict[int, list[tuple[tuple[int, int], tuple[int, int]]]]:
        """
        Scales the tiles from the extraction level to the highest magnification level.

        Args:
            tiles: dictionary containing the tiles for each cross-section.
            scaling_factor: scaling factor between the extraction level and the highest magnification level.

        Returns:
            scaled_tiles: dictionary containing the scaled tiles for each cross-section.
        """

        scaled_tiles = {}
        for cross_section in tiles:
            scaled_tiles[cross_section] = []
            for pos, loc, shape in tiles[cross_section]:
                scaled_tiles[cross_section].append(
                    (
                        pos,
                        (int(loc[0] * scaling_factor), int(loc[1] * scaling_factor)),
                        shape,
                    ),
                )

        return scaled_tiles

    def create_segmentation(self, img):
        """
        Creates a segmentation of the whole-slide image using the Otsu thresholding method.

        Args:
            img: whole-slide image.

        Returns:
            segmentation: segmentation of the whole-slide image.
        """
        img_hsv = rgb_to_hsv(img)
        img_hsv_blurred = median_filter(
            img_hsv, size=self.config["median_filter_size"], axes=[0, 1]
        )

        otsu_threshold = threshold_otsu(img_hsv_blurred[:, :, 1])
        segmentation = img_hsv_blurred[:, :, 1] > otsu_threshold
        segmentation = remove_small_holes(
            segmentation, area_threshold=self.config["hole_area_threshold"]
        )

        segmentation = segmentation[:, :, np.newaxis]

        return segmentation


def find_partially_processed_cross_sections(
    slide_paths, patch_coordinates_save_dir_path, tile_images_save_dir_path
):
    """
    Find cross-sections that have been partially processed.

    Args:
        patch_coordinates_save_dir_path: path to the directory containing the patch coordinates.
        tile_images_save_dir_path: path to the directory containing the tile images.

    Returns:
        partially_processed: list of cross-sections that have been partially processed.
    """
    partially_processed = []
    for slide in tqdm(
        slide_paths, leave=False, desc="Checking for partially processed slides"
    ):
        patch_coordinates_file_path = patch_coordinates_save_dir_path / (
            Path(slide).stem + "_cross_section_0.json"
        )
        with open(patch_coordinates_file_path) as f:
            patch_coordinates = json.load(f)
        expected_number_of_patches = len(patch_coordinates)
        extracted_number_of_patches = len(
            list(
                (tile_images_save_dir_path / patch_coordinates_file_path.stem).iterdir()
            )
        )
        if expected_number_of_patches != extracted_number_of_patches:
            partially_processed.append(slide)

    return partially_processed


def main(config):
    preprocessor = Preprocessor(config)

    manifest = pd.read_csv(config["manifest_file_path"])
    slide_paths = manifest["slide_path"].values

    preprocessor.patch_coordinates_save_dir_path.mkdir(parents=True, exist_ok=True)
    preprocessor.tile_images_save_dir_path.mkdir(parents=True, exist_ok=True)

    # only process slides that have not yet been processed
    not_yet_processed = [
        slide_path
        for slide_path in slide_paths
        if not (
            preprocessor.patch_coordinates_save_dir_path
            / f"{Path(slide_path).stem}_cross_section_0.json"
        ).exists()
    ]

    # add to this the slides which have been only partially processed
    already_processed = [
        slide_path for slide_path in slide_paths if not slide_path in not_yet_processed
    ]
    only_partially_processed = find_partially_processed_cross_sections(
        already_processed,
        preprocessor.patch_coordinates_save_dir_path,
        preprocessor.tile_images_save_dir_path,
    )

    todo = not_yet_processed + only_partially_processed

    Parallel(
        n_jobs=(
            config["num_workers"]
            if config["num_workers"] is not None
            else mp.cpu_count()
        )
    )(
        delayed(preprocessor)(slide)
        for slide in tqdm(todo, desc="Preprocessing slides", unit="slides", leave=False)
    )
