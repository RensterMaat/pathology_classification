import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from source.utils.utils import get_slide, get_coordinates, scale_coordinates


class HeatmapGenerator:
    def __init__(self, config: dict) -> None:
        self.config = config

    def __call__(
        self, heatmap_vector: np.array, slide_id: str
    ) -> matplotlib.figure.Figure:
        slide = get_slide(slide_id, self.config["slide_dir"])
        coordinates = get_coordinates(slide_id, self.config["patch_coordinate_dir"])

        heatmap = self.generate_heatmap(slide, heatmap_vector, coordinates)

        return heatmap

    def get_image(self, slide: str) -> np.array:
        return np.array(
            slide.read_region(
                (0, 0),
                self.config["level_for_visualizing_heatmap"],
                slide.level_dimensions[self.config["level_for_visualizing_heatmap"]],
            )
        )

    def generate_heatmap(
        self, slide: str, heatmap_vector: np.array, coordinates: np.array
    ) -> matplotlib.figure.Figure:
        img = self.get_image(slide)
        patch_size = self.get_patch_size_for_plotting(slide)
        scaled_coordinates = scale_coordinates(
            coordinates, slide, self.config["level_for_visualizing_heatmap"]
        )

        if self.config["classifier"] == "NaivePoolingClassifier":
            cmap = matplotlib.colormaps["seismic"]
        else:
            cmap = matplotlib.colormaps["Reds"]
            heatmap_vector = (heatmap_vector - heatmap_vector.min(axis=0).values) / (
                heatmap_vector.max(axis=0).values - heatmap_vector.min(axis=0).values
            )

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
            slide.level_dimensions[self.config["level_during_feature_extraction"]][0]
            / slide.level_dimensions[self.config["level_for_visualizing_heatmap"]][0]
        )
        return self.config["patch_size_during_feature_extraction"] / scaling_factor
