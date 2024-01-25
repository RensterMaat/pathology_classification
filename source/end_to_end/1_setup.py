import os
import yaml
import pandas as pd
from pathlib import Path


def check_manifest(manifest_path):
    """
    Performs quality checks on the provided manifest.

    Checks for:
    - existence of manifest
    - existence of slide_id column
    - existence of slide_path column
    - existence of slides

    Parameters
    ----------
    manifest_path : Path
        Path to the manifest

    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Could not find {manifest_path}")

    manifest = pd.read_csv(manifest_path)

    if not "slide_id" in manifest.columns:
        raise KeyError(
            "Could not find slide_id column in manifest. Please provide a manifest with a slide_id column"
        )

    if not "slide_path" in manifest.columns:
        raise KeyError(
            "Could not find slide_path column in manifest. Please provide a manifest with a slide_path column"
        )

    slide_paths_that_do_not_exist = manifest.loc[
        ~manifest.slide_path.apply(lambda x: Path(x).exists())
    ]
    if len(slide_paths_that_do_not_exist) > 0:
        raise FileNotFoundError(
            f"The following slides do not exist: {slide_paths_that_do_not_exist}"
        )

    print("Manifest checks passed")
    print(f"You provided a manifest with {len(manifest)} slides\n")


if __name__ == "__main__":
    print("Setting up the end-to-end pipeline\n")

    print(
        "Please provide the path to the manifest, containing the paths to the slides and the corresponding labels"
    )
    manifest_path = Path(input("Manifest path: "))
    check_manifest(manifest_path)

    print(
        "Please provide the path to the folder where all pipeline output will be stored"
    )
    output_path = Path(input("Output path: "))
    output_path.mkdir(exist_ok=True, parents=True)

    print(
        "\nPlease provide the name for the config file. This config file will store the path to the manifest file and output folder, and will be used in the following steps of the pipeline. "
    )
    config_name = input("Config name: ")
    config_path = (
        Path(os.path.abspath(__file__)).parent.parent.parent
        / "config"
        / "end_to_end"
        / (config_name + ".yaml")
    )
    print(
        f"Config file will be stored at {config_path}. Pass this path as an argument to the following steps of the pipeline."
    )

    config = {
        "manifest_file_path": str(manifest_path),
        "output_dir": str(output_path),
    }

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    print("\nSetup complete")
