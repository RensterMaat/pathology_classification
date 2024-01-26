import argparse
import source.extract.extract as extract
from source.utils.utils import load_config
from tqdm import tqdm


def main(config):
    # Activate the settings for extract from the config
    config.update(config["extract"])

    # Loop through all extractor models
    for model, settings in tqdm(
        config["extractor_models"].items(),
        desc="Extracting with different extractor models",
        unit="models",
        leave=False,
    ):
        config["extractor_model"] = model

        # Loop through all requested levels for this extractor model
        for level in tqdm(
            settings["extraction_levels"],
            desc="Extracting at different magnification levels",
            unit="levels",
            leave=False,
        ):
            config["extraction_level"] = level

            # Extract features
            extract.main(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract features at all requested levels and extractor models"
    )
    parser.add_argument("--config", default="config/end_to_end/umcu.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    main(config)
