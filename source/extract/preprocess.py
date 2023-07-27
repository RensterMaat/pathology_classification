import os
import argparse
import torch
import json
import numpy as np
from torch.nn.functional import conv2d
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_holes
from matplotlib.colors import rgb_to_hsv
from scipy.ndimage import median_filter
from openslide import OpenSlide
from math import ceil, floor
from typing import Optional
from pathlib import Path
from tqdm import tqdm

from source.utils.utils import load_config, get_patch_coordinates_dir_name


class Preprocessor:
    """
    Preprocessor class for extracting tiles from whole-slide images.

    Args:
        config: dictionary containing the configuration parameters.
    """

    def __init__(self, config: dict) -> None:
        self.config = config

        self.save_dir_name = get_patch_coordinates_dir_name(config)

    def __call__(
        self, slide_path: str | os.PathLike, segmentation_path: str | os.PathLike = None
    ) -> dict[int, list[tuple[tuple[int, int], tuple[int, int]]]]:
        """
        Extracts tiles from a whole-slide image. Defaults to using the Otsu thresholding method
        to create a segmentation if no segmentation is provided.

        Args:
            slide_path: path to the whole-slide image.
            segmentation_path: path to the segmentation of the whole-slide image.

        Returns:
            tiles: dictionary containing the tiles for each cross-section.
        """

        slide = OpenSlide(str(slide_path))

        if segmentation_path is None:  # default to Otsu thresholding
            img = np.array(
                slide.read_region(
                    (0, 0),
                    self.config["preprocessing_level"],
                    slide.level_dimensions[self.config["preprocessing_level"]],
                )
            )[:, :, :3]
            segmentation = self.create_segmentation(img)
        else:  # use user-supplied segmentation
            raise NotImplementedError(
                "Preprocessing based on user-supplied segmentation is not yet implemented."
            )

        scaling_factor = (
            slide.level_downsamples[self.config["preprocessing_level"]]
            / slide.level_downsamples[self.config["extraction_level"]]
        )
        tiles = self.tessellate(segmentation, scaling_factor)
        scaled_tiles = self.scale_tiles(tiles, scaling_factor)

        self.save_tiles(scaled_tiles, slide_path)

        return scaled_tiles

    def save_tiles(
        self,
        tiles: dict[int, list[tuple[tuple[int, int], tuple[int, int]]]],
        slide_path: str | os.PathLike,
    ) -> None:
        """
        Saves the tiles to a JSON file.

        Args:
            tiles: dictionary containing the tiles for each cross-section.
            slide_path: path to the whole-slide image.

        """
        for cross_section in tiles:
            slide_name = Path(slide_path).stem
            cross_section_name = f"{slide_name}_cross_section_{cross_section}.json"

            save_dir_path = Path(self.config["patch_coordinates_dir"]) / self.save_dir_name
            if not save_dir_path.exists():
                save_dir_path.mkdir()

            cross_section_save_path = (
                save_dir_path / cross_section_name
            )

            with open(cross_section_save_path, "w") as f:
                json.dump(tiles[cross_section], f)

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
                        (
                            int(shape[0] * scaling_factor),
                            int(shape[1] * scaling_factor),
                        ),
                    )
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

        Returns:
            tile_information: dictionary containing the location and shape of the tiles for each cross-section.
        """
        shape = [int(dim / scaling_factor) for dim in self.config["patch_dimensions"]]

        if isinstance(segmentation, torch.Tensor):
            segmentation = segmentation.numpy()
        # check if the segmentation shape has three axes
        if len(segmentation.shape) != 3:
            raise ValueError(
                "Argument for `segmentation` has an invalid shape (expected 3 axes).",
            )
        # check if the shape is not larger than the segmentation
        if shape[0] > segmentation.shape[0] or shape[1] > segmentation.shape[1]:
            raise ValueError(
                "Argument for `shape` exceeds the height and/or width of the segmentation.",
            )
        # set the stride equal to the shape if it is None
        stride = self.config["stride"] if "stride" in self.config.keys() else shape

        # initialize dictionary to store tile location and shapes
        tile_information = {}

        # loop over the axes:
        for i in range(segmentation.shape[-1]):
            # get the cross-section
            cross_section = segmentation[..., i]

            # get bounding box for cross-section
            bounding_box = self.get_bounding_box(cross_section)
            if bounding_box is not None:
                # get the bounding box coordinates
                top, bottom, left, right = bounding_box
                # calculate the height and width
                height = bottom - top
                width = right - left
                # calculate the added height and width to make it divisible
                added_height = (ceil(height / shape[0]) * shape[0]) - height
                added_width = (ceil(width / shape[1]) * shape[1]) - width

                # determine the correction by splitting the height and width to be added
                height_correction = (-floor(added_height / 2), ceil(added_height / 2))
                width_correction = (-floor(added_width / 2), ceil(added_width / 2))

                # determine how far outside the corrected height and width would be
                outside_top = max(0, -(top + height_correction[0]))
                outside_bottom = max(
                    0, (bottom + height_correction[1]) - segmentation.shape[0]
                )
                outside_left = max(0, -(left + width_correction[0]))
                outside_right = max(
                    0, (right + width_correction[1]) - segmentation.shape[1]
                )

                # address the corrections for which there is overlap at the top and/or bottom
                if (outside_top and not outside_bottom) or (
                    outside_bottom and not outside_top
                ):
                    height_correction = (
                        height_correction[0] + outside_top - outside_bottom,
                        height_correction[1] + outside_top - outside_bottom,
                    )
                elif outside_top and outside_bottom:
                    height_correction = (
                        floor((shape[0] - added_height) / 2),
                        -ceil((shape[0] - added_height) / 2),
                    )
                # address the corrections for which there is overlap on the left and/or right
                if (outside_left and not outside_right) or (
                    outside_right and not outside_left
                ):
                    width_correction = (
                        width_correction[0] + outside_left - outside_right,
                        width_correction[1] + outside_left - outside_right,
                    )
                elif outside_left and outside_right:
                    width_correction = (
                        floor((shape[1] - added_width) / 2),
                        -ceil((shape[1] - added_width) / 2),
                    )
                # determine the top left coordinate as (x, y)
                top_left = (left + width_correction[0], top + height_correction[0])
                height = height - height_correction[0] + height_correction[1]
                width = width - width_correction[0] + width_correction[1]

                # crop the cross-section and prepare for convolution
                crop = cross_section[
                    top_left[1] : top_left[1] + height,
                    top_left[0] : top_left[0] + width,
                ][None, None, ...].astype(np.float32)
                # define average filter
                filter = (torch.ones(shape) / np.prod(shape))[None, None, ...]
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
                            int(top_left[0] + x * stride[1]),
                            int(top_left[1] + y * stride[0]),
                        )
                    )
                # add information about the location and shape of the tiles
                # to the dictionary
                tile_information[str(i)] = [
                    (pos, loc, tuple(shape)) for pos, loc in zip(positions, locations)
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

    for slide in tqdm(list(Path(config["slides_dir"]).glob("*.ndpi"))):
        preprocessor(slide)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extracts tiles from a whole-slide image."
    )
    parser.add_argument(
        "config_path",
    )
    args = parser.parse_args()

    config = load_config(args.config_path)

    main(config)