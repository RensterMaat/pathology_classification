import os
import argparse
import torch
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from torch.nn.functional import conv2d
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_holes
from matplotlib.colors import rgb_to_hsv
from scipy.ndimage import median_filter
from openslide import OpenSlide
from typing import Optional
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed

from source.utils.utils import (
    load_config,
    get_patch_coordinates_dir_name,
)


class Preprocessor:
    """
    Preprocessor class for extracting tiles from whole-slide images.

    Args:
        config: dictionary containing the configuration parameters.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.patch_coordinates_save_dir_path = get_patch_coordinates_dir_name(config)

    def __call__(
        self, slide_path: str | os.PathLike, segmentation_path: str | os.PathLike = None
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

        preprocessing_level = min(
            self.config["preprocessing_level"], len(slide.level_dimensions) - 1
        )

        if segmentation_path is None:  # default to Otsu thresholding
            img = np.array(
                slide.read_region(
                    (0, 0),
                    preprocessing_level,
                    slide.level_dimensions[preprocessing_level],
                )
            )[:, :, :3]
            segmentation = self.create_segmentation(img)
        else:  # use user-supplied segmentation
            raise NotImplementedError(
                "Preprocessing based on user-supplied segmentation is not yet implemented."
            )

        scaling_factor = (
            slide.level_downsamples[preprocessing_level]
            / slide.level_downsamples[self.config["extraction_level"]]
        )
        tile_coordinates = self.tessellate(segmentation, scaling_factor)
        scaled_tile_coordinates = self.scale_tiles(
            tile_coordinates, slide.level_downsamples[preprocessing_level]
        )

        if tile_coordinates:
            self.save_tile_coordinates(scaled_tile_coordinates, slide_path)

        return scaled_tile_coordinates

    def save_tile_coordinates(
        self,
        tile_coordinates: dict[int, list[tuple[tuple[int, int], tuple[int, int]]]],
        slide_path: str | os.PathLike,
    ) -> None:
        """
        Saves the tiles to a JSON file.

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

        otsu_threshold = threshold_otsu(img_hsv_blurred[:, :, 2])
        segmentation = img_hsv_blurred[:, :, 2] < otsu_threshold
        segmentation = remove_small_holes(
            segmentation, area_threshold=self.config["hole_area_threshold"]
        )

        segmentation = segmentation[:, :, np.newaxis]

        return segmentation

    def tessellate(
        self, segmentation: np.ndarray, scaling_factor: float = 1.0
    ) -> dict[int, list[tuple[tuple[int, int], tuple[int, int]]]]:
        """
        Extracts tiles from a segmentation.

        Args:
            segmentation: segmentation of the whole-slide image as (height, width, cross-section).
            scaling_factor: scaling factor between the extraction level and the highest magnification level.

        Returns:
            tile_information: dictionary containing the location and shape of the tiles for each cross-section.
        """
        segmentation_width, segmentation_height = (
            segmentation.shape[1],
            segmentation.shape[0],
        )

        patch_size = [
            int(dim / scaling_factor) for dim in self.config["patch_dimensions"]
        ]
        patch_width, patch_height = patch_size[1], patch_size[0]

        if isinstance(segmentation, torch.Tensor):
            segmentation = segmentation.numpy()
        # check if the segmentation shape has three axes
        if len(segmentation.shape) != 3:
            raise ValueError(
                "Argument for `segmentation` has an invalid shape (expected 3 axes).",
            )
        # check if the shape is not larger than the segmentation
        if patch_width > segmentation_width or patch_height > segmentation_height:
            raise ValueError(
                "Argument for `shape` exceeds the height and/or width of the segmentation.",
            )

        # set the stride equal to the shape if it is None
        stride = (
            self.config["stride"]
            if "stride" in self.config.keys() and self.config["stride"] is not None
            else patch_size
        )
        horizontal_stride, vertical_stride = stride[1], stride[0]

        # calulate the maximum number of patches that fit inside the image
        max_horizontal_patches = (
            segmentation_width - patch_width
        ) // horizontal_stride + 1
        max_vertical_patches = (
            segmentation_height - patch_height
        ) // vertical_stride + 1

        # initialize dictionary to store tile location and shapes
        tile_information = {}

        for i, cross_section in enumerate(segmentation.transpose(2, 0, 1)):
            bounding_box = self.get_bounding_box(cross_section)

            if bounding_box is None:
                continue

            # create a boundix box and calculate the width and height
            (
                top,
                bottom,
                left,
                right,
            ) = bounding_box

            bounding_box_width = right - left
            bounding_box_height = bottom - top

            # determine how many patches are needed in both dimensions to completely cover the bounding box
            # two cases are possible:
            # 1. the bounding box is smaller than the patch size -> default to one patch
            # 2. the bounding box is larger than the patch size -> calculate the number of patches needed
            horizontal_patches_needed_to_cover_bounding_box = (
                max(
                    math.ceil((bounding_box_width - patch_width) / horizontal_stride), 0
                )
                + 1
            )
            vertical_patches_needed_to_cover_bounding_box = (
                max(
                    math.ceil((bounding_box_height - patch_height) / vertical_stride), 0
                )
                + 1
            )

            # the number of patches should never be more than the maximum number of patches
            # that fit inside the image
            horizontal_patches = min(
                horizontal_patches_needed_to_cover_bounding_box, max_horizontal_patches
            )
            vertical_patches = min(
                vertical_patches_needed_to_cover_bounding_box, max_vertical_patches
            )

            # calculate the width and height spanned by the patches
            width_spanned_by_horizontal_patches = (
                horizontal_patches - 1
            ) * horizontal_stride + patch_width
            height_spanned_by_vertical_patches = (
                vertical_patches - 1
            ) * vertical_stride + patch_height

            # Align the center of the patches with the center of the bounding box
            horizontal_shift = (
                bounding_box_width - width_spanned_by_horizontal_patches
            ) // 2
            left_margin = left + horizontal_shift

            vertical_shift = (
                bounding_box_height - height_spanned_by_vertical_patches
            ) // 2
            top_margin = top + vertical_shift

            # Ensure that no patches fall outside the segmentation
            if left_margin < 0:
                left_margin = 0
            if left_margin + width_spanned_by_horizontal_patches > segmentation_width:
                horizontal_shift = (
                    left_margin
                    + width_spanned_by_horizontal_patches
                    - segmentation_width
                )
                left_margin -= horizontal_shift
            if top_margin < 0:
                top_margin = 0
            if top_margin + height_spanned_by_vertical_patches > segmentation_height:
                vertical_shift = (
                    top_margin
                    + height_spanned_by_vertical_patches
                    - segmentation_height
                )
                top_margin -= vertical_shift

            # crop the cross-section and prepare for convolution
            crop = cross_section[
                top_margin : top_margin + height_spanned_by_vertical_patches,
                left_margin : left_margin + width_spanned_by_horizontal_patches,
            ][None, None, ...].astype(np.float32)
            # define average filter
            filter = (torch.ones(patch_size) / np.prod(patch_size))[None, None, ...]
            # convolve crop with filter
            filtered_crop = conv2d(
                torch.from_numpy(crop),
                weight=filter,
                bias=None,
                stride=stride,
                padding="valid",
            )[0, 0, ...].numpy()
            # find the region to extract tiles from

            extraction_region = np.where(
                filtered_crop >= self.config["min_tissue_fraction"], 1, 0
            )

            # find the indices of the tiles that exceed the minimum faction of tissue
            indices = np.nonzero(extraction_region)
            # loop over indices to get all (top left) tile locations
            positions = []
            locations = []
            for x, y in zip(indices[1], indices[0]):
                positions.append((int(x), int(y)))
                locations.append(
                    (
                        int(left_margin + x * horizontal_stride),
                        int(top_margin + y * vertical_stride),
                    )
                )
            # add information about the location and shape of the tiles
            # to the dictionary
            tile_information[str(i)] = [
                (pos, loc, self.config["patch_dimensions"])
                for pos, loc in zip(positions, locations)
            ]

        return tile_information

    def get_bounding_box(
        self, array: np.ndarray
    ) -> Optional[tuple[int, int, int, int]]:
        """
        Returns minimum and maximum row and column index for box around
        non-zero elements.

        Args:
            array: array for which the bounding box enclosing all non-zero elements
                should be found.
        Returns:
            rmin: smallest row index with non-zero element.
            rmax: largest row index with non-zero element.
            cmin: smallest column index with non-zero element.
            cmax: largest column index with non-zero element.
        """
        rows = np.any(array, axis=1)
        cols = np.any(array, axis=0)
        # check if there are any non-zero elements
        if sum(rows) == 0 or sum(cols) == 0:
            return None

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        return rmin, rmax, cmin, cmax


def main(config):
    preprocessor = Preprocessor(config)

    with open(config["slide_paths_txt_file"], "r") as f:
        slide_paths = f.read().splitlines()

    preprocessor.patch_coordinates_save_dir_path.mkdir(parents=True, exist_ok=True)

    Parallel(
        n_jobs=config["num_workers"]
        if config["num_workers"] is not None
        else mp.cpu_count()
    )(
        delayed(preprocessor)(slide)
        for slide in tqdm(slide_paths, desc="Preprocessing slides", unit="slides")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extracts patches from a whole-slide image."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/preprocess/default.yaml",
        help="Path to the config file.",
    )
    args = parser.parse_args()

    print(args.config)
    config = load_config(args.config)

    main(config)
