import os
import urllib.request
import zipfile
from aerosonicdb.utils import get_project_root

root_path = get_project_root()
download_path = os.path.join(root_path, 'data/raw')


def download(target_path=download_path):
    # Target URLs for the dataset files on Zenodo
    audio_zip = 'https://zenodo.org/record/8371595/files/audio.zip'
    env_audio_zip = 'https://zenodo.org/record/8371595/files/env_audio.zip'
    sample_meta_csv = 'https://zenodo.org/record/8371595/files/sample_meta.csv'
    aircraft_meta_csv = 'https://zenodo.org/record/8371595/files/aircraft_meta.csv'
    aircraft_meta_json = 'https://zenodo.org/record/8371595/files/aircraft_meta.json'
    locations_json = 'https://zenodo.org/record/8371595/files/locations.json'
    env_map_csv = 'https://zenodo.org/record/8371595/files/environment_class_mappings.csv'
    license_txt = 'https://zenodo.org/record/8371595/files/LICENSE.txt'
    readme_md = 'https://zenodo.org/record/8371595/files/README.md'

    # Specify and create the target directory for the dataset

    if not os.path.exists(target_path):
        os.makedirs(target_path)

        # Download the audio zip file
        urllib.request.urlretrieve(audio_zip, os.path.join(target_path, 'audio.zip'))

        # Download the environment audio zip file
        urllib.request.urlretrieve(env_audio_zip, os.path.join(target_path, 'env_audio.zip'))

        # Download the meta files
        urllib.request.urlretrieve(sample_meta_csv, os.path.join(target_path, 'sample_meta.csv'))
        urllib.request.urlretrieve(env_map_csv, os.path.join(target_path, 'environment_class_mappings.csv'))
        urllib.request.urlretrieve(aircraft_meta_csv, os.path.join(target_path, 'aircraft_meta.csv'))
        urllib.request.urlretrieve(aircraft_meta_json, os.path.join(target_path, 'aircraft_meta.json'))
        urllib.request.urlretrieve(locations_json, os.path.join(target_path, 'locations.json'))
        urllib.request.urlretrieve(license_txt, os.path.join(target_path, 'LICENSE.txt'))
        urllib.request.urlretrieve(readme_md, os.path.join(target_path, 'README.md'))

        # Extract the dataset
        with zipfile.ZipFile(os.path.join(target_path, 'audio.zip'), 'r') as zip_ref:
            zip_ref.extractall(target_path)

        # Clean up the zip file
        os.remove(os.path.join(target_path, 'audio.zip'))

        # Extract the environment dataset
        with zipfile.ZipFile(os.path.join(target_path, 'env_audio.zip'), 'r') as zip_ref:
            zip_ref.extractall(target_path)

        # Clean up the zip file
        os.remove(os.path.join(target_path, 'env_audio.zip'))

    else:
        print(f'Dataset downloaded - see "data/raw" directory')


if __name__ == '__main__':
    download()
