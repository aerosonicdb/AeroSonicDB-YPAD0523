import subprocess
import sys
import venv
import os
from download_dataset import download_ds


def dataset_setup():

    # Target directory for the virtual environment
    venv_dir = 'venv'

    # Python requirements file
    requirements_file = 'requirements.txt'

    # "Get Started" jupyter notebook
    notebook_path = 'AeroSonicDB_GetStarted.ipynb'

    # check if the dataset has been downloaded
    if not os.path.exists('dataset'):
        download_ds()
        print('Dataset downloaded')

    # Check if the virtual environment already exists
    if not os.path.exists(venv_dir):
        # Create the virtual environment
        venv.create(venv_dir, system_site_packages=False, clear=True)
        print("Virtual environment created.")
    else:
        print("Virtual environment already exists.")

    # Activate the virtual environment
    venv_activation_script = os.path.join(venv_dir, 'Scripts', 'activate.bat')
    subprocess.run(venv_activation_script, shell=True)
    print("Virtual environment activated")

    # Install required packages in the virtual environment
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', requirements_file])
    print("Dependencies installed.")

    # Launch Jupyter Notebook
    subprocess.run([sys.executable, '-m', 'jupyter', 'notebook', notebook_path])


if __name__ == '__main__':
    dataset_setup()
