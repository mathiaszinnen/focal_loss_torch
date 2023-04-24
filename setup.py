import pathlib
from setuptools import setup

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()

setup(
    setup_requires=['wheel'],
    name="focal_loss_torch",
    version="0.1.2",
    description="Simple pytorch implementation of focal loss",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/mathiaszinnen/focal_loss_torch",
    author="Mathias Zinnen",
    author_email="mathias.zinnen@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    packages=["focal_loss"],
    install_requires=["torch", "numpy"],
)
