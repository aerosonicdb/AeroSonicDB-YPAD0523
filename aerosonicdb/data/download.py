#!/usr/bin/env python
"""Logic and entrypoint for downloading the dataset files to local file system."""
import os
import urllib.request
import zipfile

import click

from aerosonicdb.utils import get_project_root

PROJECT_ROOT_PATH = get_project_root()
TARGET_PATH = os.path.join(PROJECT_ROOT_PATH, "data/raw")
BASE_URL = "https://zenodo.org/record/10215080/files"


def download(base_url=BASE_URL, target_path=TARGET_PATH, force=False):
    # Target URLs for the dataset files on Zenodo
    audio_zip = f"{base_url}/audio.zip"
    env_audio_zip = "{base_url}/env_audio.zip"
    sample_meta_csv = "{base_url}/sample_meta.csv"
    aircraft_meta_csv = "{base_url}/aircraft_meta.csv"
    aircraft_meta_json = "{base_url}/aircraft_meta.json"
    locations_json = "{base_url}/locations.json"
    env_map_csv = "{base_url}/environment_class_mappings.csv"
    raw_env_map = "{base_url}/environment_mappings_raw.csv"
    license_txt = "{base_url}/LICENSE.txt"
    readme_md = "{base_url}/README.md"

    # Specify and create the target directory for the dataset

    if os.path.exists(target_path) and not force:
        print(f'Dataset downloaded - see "data/raw" directory')
        return

    os.makedirs(target_path)

    # Download the audio zip file
    urllib.request.urlretrieve(audio_zip, os.path.join(target_path, "audio.zip"))

    # Download the environment audio zip file
    urllib.request.urlretrieve(
        env_audio_zip, os.path.join(target_path, "env_audio.zip")
    )

    # Download the meta files
    urllib.request.urlretrieve(
        sample_meta_csv, os.path.join(target_path, "sample_meta.csv")
    )
    urllib.request.urlretrieve(
        env_map_csv, os.path.join(target_path, "environment_class_mappings.csv")
    )
    urllib.request.urlretrieve(
        raw_env_map, os.path.join(target_path, "environment_mappings_raw.csv")
    )
    urllib.request.urlretrieve(
        aircraft_meta_csv, os.path.join(target_path, "aircraft_meta.csv")
    )
    urllib.request.urlretrieve(
        aircraft_meta_json, os.path.join(target_path, "aircraft_meta.json")
    )
    urllib.request.urlretrieve(
        locations_json, os.path.join(target_path, "locations.json")
    )
    urllib.request.urlretrieve(license_txt, os.path.join(target_path, "LICENSE.txt"))
    urllib.request.urlretrieve(readme_md, os.path.join(target_path, "README.md"))

    # Extract the dataset
    with zipfile.ZipFile(os.path.join(target_path, "audio.zip"), "r") as zip_ref:
        zip_ref.extractall(target_path)

    # Clean up the zip file
    os.remove(os.path.join(target_path, "audio.zip"))

    # Extract the environment dataset
    with zipfile.ZipFile(os.path.join(target_path, "env_audio.zip"), "r") as zip_ref:
        zip_ref.extractall(target_path)

    # Clean up the zip file
    os.remove(os.path.join(target_path, "env_audio.zip"))


@click.command()
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Force re-generation of features even if features JSON descriptor file already exists.",
)
def download_entrypoint(force):
    """Download the dataset."""
    click.echo("Downloading dataset...")
    download(force=force)


if __name__ == "__main__":
    download_entrypoint()
