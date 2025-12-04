"""Build image-analyser as a module."""

from pathlib import Path
from typing import List

import setuptools

requires: List[str] = [
    "setuptools==80.9.0",
    "opencv-python==4.12.0.88",
    "numpy==2.2.6",  # Actually supported numpy version for installed opencv
]
test: List[str] = ["pytest==8.3.3"]
extra: List[str] = ["black==24.10.0", "isort==5.13.2"]
all_modules: List[str] = requires + test + extra

source_folder = Path("analyser")
test_folder = Path("tests")

setuptools.setup(
    name="analyser",
    version="1.0",
    author="Miłosz Martynow",
    author_email="miloszmartynow@gmail.com",
    description="Analyser.",
    packages=["analyser"],
    install_requires=requires,
    test_require=test,
    extras_require={"development": extra, "all": all_modules},
    scripts=[
        str(Path("main.py")),
        str(Path.joinpath(source_folder, "environment.py")),
        str(Path.joinpath(source_folder, "line_spread_function.py")),
        str(Path.joinpath(source_folder, "modulation_transfer_function.py")),
        str(Path.joinpath(test_folder, "test_line_spread_function.py")),
        str(
            Path.joinpath(test_folder, "test_modulation_transfer_function.py")
        ),
    ],
    python_requires="==3.12.*",
)
