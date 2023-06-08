import urllib.request
import zipfile
import os

# Target URLs for the dataset files on Zenodo
audio_zip = 'https://zenodo.org/record/8004108/files/audio.zip'
sample_meta_csv = 'https://zenodo.org/record/8004108/files/sample_meta.csv'
aircraft_meta_csv = 'https://zenodo.org/record/8004108/files/aircraft_meta.csv'
aircraft_meta_json = 'https://zenodo.org/record/8004108/files/aircraft_meta.json'
locations_json = 'https://zenodo.org/record/8004108/files/locations.json'
license_txt = 'https://zenodo.org/record/8004108/files/LICENSE.txt'
readme_md = 'https://zenodo.org/record/8004108/files/README.md'

# Specify and create the target directory for the dataset
target = 'dataset'
cwd = os.getcwd()
target_directory = os.path.join(cwd, target)


if not os.path.exists(target_directory):
    os.makedirs(target_directory)


# Download the audio zip file
urllib.request.urlretrieve(audio_zip, os.path.join(target_directory, 'audio.zip'))

# Download the meta files
urllib.request.urlretrieve(sample_meta_csv, os.path.join(target_directory, 'sample_meta.csv'))
urllib.request.urlretrieve(aircraft_meta_csv, os.path.join(target_directory, 'aircraft_meta.csv'))
urllib.request.urlretrieve(aircraft_meta_json, os.path.join(target_directory, 'aircraft_meta.json'))
urllib.request.urlretrieve(locations_json, os.path.join(target_directory, 'locations.json'))
urllib.request.urlretrieve(license_txt, os.path.join(target_directory, 'LICENSE.txt'))
urllib.request.urlretrieve(readme_md, os.path.join(target_directory, 'README.md'))


# Extract the dataset
with zipfile.ZipFile(os.path.join(target_directory, 'audio.zip'), 'r') as zip_ref:
    zip_ref.extractall(target_directory)

# Clean up the zip file
os.remove(os.path.join(target_directory, 'audio.zip'))
