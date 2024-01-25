import argparse
import source.extract.preprocess as preprocess
from source.utils.utils import load_config


def main(config):
    # Loop through all requested patch sizes
    patch_dimensions_to_extract = config["patch_dimensions_options"]
    for patch_dimensions in patch_dimensions_to_extract:
        config["patch_dimensions"] = patch_dimensions

        # Loop through all requested levels
        levels_to_extract = config[f"patch_dimension_{patch_dimensions[0]}_options"][
            "extraction_levels"
        ]
        for level in levels_to_extract:
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
