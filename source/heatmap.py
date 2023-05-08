import os
import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
from openslide import OpenSlide


class HeatmapGenerator:
    def __init__(self, config: dict) -> None:
        self.config = config

    def __call__(self, heatmap_vector: np.array, slide_id: str) -> matplotlib.figure.Figure:
        slide = self.get_slide(slide_id)
        coordinates = self.get_coordinates(slide_id)

        heatmap = self.generate_heatmap(slide, heatmap_vector, coordinates)

        return heatmap

    def get_slide(self, slide_id: str) -> OpenSlide:
        slide_file_path = self.find_slide_file_path(slide_id)
        return OpenSlide(slide_file_path)

    def get_coordinates(self, slide_id: str) -> np.array:
        coordinates_file_path = self.find_coordinates_file_path(slide_id)
        return np.array(h5py.File(coordinates_file_path, "r")["coords"])

    def scale_coordinates(self, coordinates: np.array, slide: str) -> np.array:
        scaling_factor = (
            slide.level_dimensions[0][0]
            / slide.level_dimensions[self.config["level_for_visualizing_heatmap"]][0]
        )
        return coordinates / scaling_factor

    def get_image(self, slide: str) -> np.array:
        return np.array(
            slide.read_region(
                (0, 0),
                self.config["level_for_visualizing_heatmap"],
                slide.level_dimensions[self.config["level_for_visualizing_heatmap"]],
            )
        )

    def generate_heatmap(self, slide: str, heatmap_vector: np.array, coordinates: np.array) -> matplotlib.figure.Figure:
        img = self.get_image(slide)
        patch_size = self.get_patch_size_for_plotting(slide)
        scaled_coordinates = self.scale_coordinates(coordinates, slide)

        cmap = matplotlib.colormaps[
            "seismic" if self.config["model"] == "NaivePoolingClassifier" else "Reds"
        ]
        class_ix = 1

        fig, ax = plt.subplots(figsize=np.array(img.shape[:-1]) / self.config["dpi"])
        ax.imshow(img)
        ax.axis("off")

        for ix, coordinate in enumerate(scaled_coordinates):
            prediction = float(heatmap_vector[ix, class_ix])
            ax.add_patch(
                Rectangle(
                    coordinate,
                    patch_size,
                    patch_size,
                    alpha=0.3,
                    color=cmap(prediction),
                )
            )

        return fig

    def get_patch_size_for_plotting(self, slide: str) -> int:
        scaling_factor = (
            slide.level_dimensions[self.config["level_during_feature_extraction"]][0]
            / slide.level_dimensions[self.config["level_for_visualizing_heatmap"]][0]
        )
        return self.config["patch_size_during_feature_extraction"] / scaling_factor

    def find_slide_file_path(self, slide_id:str) -> str:
        for dir, _, files in os.walk(self.config["slide_dir"]):
            for file in files:
                if ".".join(file.split(".")[:-1]) == slide_id:
                    return str(Path(dir) / file)

    def find_coordinates_file_path(self, slide_id: str) -> str:
        # for dir, _, files in os.walk(self.config["patch_coordinate_dir"]):
        #     for file in files:
        #         if file == slide_id + ".h5":
        #             return str(Path(dir) / file)
        root = Path(self.config["patch_coordinate_dir"])
        return str(root / slide_id / (slide_id + '.h5'))
