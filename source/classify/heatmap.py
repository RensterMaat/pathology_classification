import json
import torch
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
from openslide import OpenSlide
from source.utils.utils import (
    get_patch_coordinates_dir_name,
    scale_coordinates,
)


class HeatmapGenerator:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.manifest = pd.read_csv(config["manifest_file_path"]).set_index("slide_id")

    def __call__(self, heatmap_vector_file_path: str) -> matplotlib.figure.Figure:
        slide_id = Path(heatmap_vector_file_path).stem.replace("_cross_section_0", "")

        slide_path = self.manifest.loc[slide_id, "slide_path"]
        slide = OpenSlide(slide_path)

        coordinates_path = get_patch_coordinates_dir_name(self.config) / (
            slide_id + "_cross_section_0.json"
        )
        with open(coordinates_path, "r") as f:
            coordinates = np.array(json.load(f))[:, 1]

        heatmap_vector = torch.load(heatmap_vector_file_path, map_location="cpu")

        heatmap = self.generate_heatmap(slide, heatmap_vector, coordinates)

        return heatmap

    def get_image(self, slide: OpenSlide) -> np.array:
        return np.array(
            slide.read_region(
                (0, 0),
                self.config["level_for_visualizing_heatmap"],
                slide.level_dimensions[self.config["level_for_visualizing_heatmap"]],
            )
        )

    def generate_heatmap(
        self, slide: OpenSlide, heatmap_vector: np.array, coordinates: np.array
    ) -> matplotlib.figure.Figure:
        img = self.get_image(slide)
        patch_size = self.get_patch_size_for_plotting(slide)
        scaled_coordinates = scale_coordinates(
            coordinates, slide, self.config["level_for_visualizing_heatmap"]
        )

        # if self.config["classifier"] == "NaivePoolingClassifier":
        cmap = matplotlib.colormaps["seismic"]
        # else:
        #     cmap = matplotlib.colormaps["Reds"]
        #     heatmap_vector = (heatmap_vector - heatmap_vector.min(axis=0).values) / (
        #         heatmap_vector.max(axis=0).values - heatmap_vector.min(axis=0).values
        #     )

        if self.config["classifier"] == "TransformerClassifier":
            class_ix = 0
        else:
            class_ix = 1

        fig, ax = plt.subplots(
            figsize=np.array(list(reversed(img.shape[:-1]))) / self.config["dpi"]
        )
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
            slide.level_dimensions[self.config["extraction_level"]][0]
            / slide.level_dimensions[self.config["level_for_visualizing_heatmap"]][0]
        )

        if self.config["extractor_model"] == "region_hipt":
            patch_size = 4096
        else:
            patch_size = 256

        return patch_size / scaling_factor
