import argparse
import source.extract.preprocess as preprocess
from source.utils.utils import load_config
from tqdm import tqdm


def main(config):
    # Loop through all requested patch sizes
    patch_dimensions_to_extract = config["patch_dimensions_options"]
    for patch_dimensions in tqdm(
        patch_dimensions_to_extract,
        desc="Extracting at different patch sizes",
        unit="sizes",
        leave=False,
    ):
        config["patch_dimensions"] = patch_dimensions

        # Loop through all requested levels
        levels_to_extract = config[f"patch_dimension_{patch_dimensions[0]}_options"][
            "extraction_levels"
        ]
        for level in tqdm(
            levels_to_extract,
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
