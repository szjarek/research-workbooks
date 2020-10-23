from setuptools import find_packages, setup

setup(
    name='sample_aml_steps',
    packages=find_packages(),
    version='0.1.0',
    description='Sample shared compute logic',
    author='Objectivity',
    license='',
)

# SETUP proj_compute module using [pip install -e .] from the app folder
