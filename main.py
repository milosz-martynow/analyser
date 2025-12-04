from pathlib import Path

import cv2

from analyser.environment import IMAGES

from analyser.line_spread_function import lsf
from analyser.modulation_transfer_function import mtf_analysis

for image in list(Path(IMAGES).glob("*.*")):
    print(image)
    image = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
    lsf_derivatives, lsf_distances = lsf(image=image)
    mtf_analysis(lsf_distances=lsf_distances, lsf_derivatives=lsf_derivatives)
    print()
