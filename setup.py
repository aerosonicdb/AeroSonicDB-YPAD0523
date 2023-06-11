from setuptools import setup, find_packages

setup(
    name='aerosonicdb',
    version='0.2.2',
    description='Download and get started with the AeroSonicDB-YPAD0523 audio dataset of low-flying aircraft.',
    author='Blake Downward',
    author_email='aerosonicdb@gmail.com',
    url='https://github.com/aerosonicdb/AeroSonicDB-YPAD0523',
    packages=find_packages(),
    install_requires=['pandas', 'jupyter', 'notebook']
)
