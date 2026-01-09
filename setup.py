from setuptools import setup, find_packages

setup(
    name="master_thesis_tennis",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "numpy",
        "tqdm",
        "scipy",
        "matplotlib",
    ],
)