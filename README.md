# Description
## General
This project provides a complete workflow for computing Line Spread Functions (LSF) 
and Modulation Transfer Functions (MTF) from image data.
The core goal is to automatically estimate such features like:
Fourier-domain computation of MTF from LSF derivatives.
- Nyquist and half-Nyquist contrast evaluation.
- Full-Width at Half Maximum (FWHM) computation.
- Ground Resolved Distance (GRD) via threshold-crossing.

## Workflow
The toolkit extracts intensity profiles along edge points pairs across calibration targets images.
LSF is calculated as derivative of intensity across such line with respect to pixelwise relative distance.
MTF is calculated as fourier transform of LSF, further calculations are performed on this frequency domain spectrum.
Calibration targets are embedded in aerial imagery, which were manually extracted.
Aggregated results metrics are treated as statistical measures of image quality.

## Run
To run the MTF analysis just build and activate virtual environment and install module (information below) 
and after first installation step run main script:

### first run:
```commandline
py.exe -m venv .venv
.venv\Scripts\activate
pip.exe install -e ".[all]"
python.exe main.py
```

### later runs:
```commandline
python.exe main.py
```
Rest of maintenance information are stored in this README.md file, in Dependencies section.

## Results
Unfortunetley for aerial test images statistical results have enormous standard deviation.
This means that input aerial tests images are poor overall quality. 
But for sure algorithm presented here requires further development as well 

### \analyser\images\target-bar.png
| Metric                | Mean  | Std Dev     |
|----------------------|-------|--------------|
| **MTF @ Nyquist**     | 0.116 | 0.126        |
| **MTF @ Nyquist / 2** | 1.360 | 0.832        |
| **FWHM of MTF**       | 0.185 | 0.069        |
| **GRD**               | 0.000 | 0.000    |

---

### analyser/images/target-chess.png

| Metric                | Mean       | Std Dev     |
|----------------------|------------|--------------|
| **MTF @ Nyquist**     | 0.158      | 0.180        |
| **MTF @ Nyquist / 2** | 1.319      | 1.052        |
| **FWHM of MTF**       | 0.243      | 0.080        |
| **GRD**               | 16.485     | 34.742       |

---

### analyser/images/target-star.png

| Metric                | Mean       | Std Dev     |
|----------------------|------------|--------------|
| **MTF @ Nyquist**     | 0.076      | 0.082        |
| **MTF @ Nyquist / 2** | 1.452      | 1.203        |
| **FWHM of MTF**       | 0.221      | 0.084        |
| **GRD**               | 16.473     | 23.009       |

# Dependencies

## Operational system
Base Operating system is Windows 11 Pro.

## Python 
Python in version [3.12.7](https://peps.python.org/pep-0693/). 

## Virtual Environment
For better maintenance of Python code it is worth to use 
[Python Virtual Environment](https://docs.python.org/3/library/venv.html). If Python Virtual Environment is 
installed, then project specific virtual environment might be created by typing in terminal of project root folder:
```
py.exe -m venv .venv
.venv\Scripts\activate
```
Note - `'.venv'` name is included in `.gitignore` file.

## PIP
Upgrading pip will be useful, when issues with requirements libraries araises:
```
py.exe -m pip install --upgrade pip
```

## Install packages
To install `image-analyser` with all project dependencies:
```
pip.exe install -e ".[all]"
```
To install `image-analyser` only with required dependencies:
```
pip.exe install -e .
```
To install `image-analyser` with test dependencies:
```
pip.exe install -e ".[test]"
```
To install `image-analyser` with extra dependencies for formatting:
```
pip.exe install -e ".[extra]"
```
Sometimes, python does not come with `setuptools`. If so - above will not work until
`setuptools` will be installed virtual environment:
```
pip.exe install setuptools==75.6.0
```

## Run tests
If `[test]` dependencies are installed, code testing with automatic tests is possible via:
```pytest.exe .```

## Run formatting
If `[extra]` dependencies are installed, code maintenance by automatized and standardized formatting method 
is possible via:
```
isort.exe .
black.exe --config=.blackrc .\analyser\ .\tests\ setup.py
```



https://www.researchgate.net/publication/252133214_Retroreflection_reduction_by_masking_apertures
https://ntrs.nasa.gov/api/citations/20240003379/downloads/2024-ss-jacie_Gary_Lin_Spatial_Resolution_STI.pdf