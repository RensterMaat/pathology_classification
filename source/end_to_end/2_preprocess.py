import argparse
import source.extract.preprocess as preprocess
from source.utils.utils import load_config
from tqdm import tqdm
from collections import defaultdict


def main(config):
    # Activate the settings for preprocess from the config
    config.update(config["preprocess"])

    # Format requested patch sizes and corresponding magnification levels in a dict
    patch_sizes_vs_magnification_levels = defaultdict(set)
    for model, settings in config["extractor_models"].items():
        patch_sizes_vs_magnification_levels[tuple(settings["patch_dimensions"])].update(
            settings["extraction_levels"]
        )

    # Loop through all requested patch sizes
    for patch_dimensions, magnification_levels in tqdm(
        patch_sizes_vs_magnification_levels.items(),
        desc="Extracting at different patch sizes",
        unit="sizes",
        leave=False,
    ):
        config["patch_dimensions"] = list(patch_dimensions)

        # Loop through all requested levels for this patch size
        for level in tqdm(
            magnification_levels,
            desc="Extracting at different magnification levels",
            unit="levels",
            leave=False,
        ):
            config["extraction_level"] = level

            # Extract patches
            preprocess.main(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tessalate slides at all requested levels and patch sizes"
    )
    parser.add_argument("--config", default="config/end_to_end/umcu.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    main(config)
