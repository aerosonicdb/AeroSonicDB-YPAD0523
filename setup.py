from setuptools import setup, find_packages

setup(
    name='AeroSonicDB',
    version='0.5.0',
    description='Download and get started with the AeroSonicDB-YPAD0523 audio dataset of low-flying aircraft.',
    author='Blake Downward',
    author_email='aerosonicdb@gmail.com',
    url='https://github.com/aerosonicdb/AeroSonicDB-YPAD0523',
    packages=find_packages(where='aerosonicdb'),
    install_requires=['librosa', 'tensorflow', 'sklearn', 'scikeras', 'matplotlib', 'pandas', 'jupyter', 'notebook'],
    package_dir={'': 'aerosonicdb'}
)
