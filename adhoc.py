# %%
from pathlib import Path

import cv2
import numpy as np
from pdf2image import convert_from_path
import pytesseract  # type: ignore

# %%
# If you don't have tesseract executable in your PATH, include the following:
# pytesseract.pytesseract.tesseract_cmd = r'<full_path_to_your_tesseract_executable>'
# Example tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'


# %%
def enhance_image(image):
    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]
    de_noised = cv2.fastNlMeansDenoising(thresh, None, 30, 7, 21)
    return de_noised


# %%
def extract_text(image):
    return pytesseract.image_to_string(image, config="--oem 3 --psm 6")


# %%
file_dir = Path(__file__).parent
data_dir = file_dir.joinpath("data")


# %%
inpath = data_dir.joinpath("cms_1.pdf")
images = convert_from_path(inpath)

# %%
with open(file_dir.joinpath("output", "cms_1.txt"), "w") as fw:
    for image in images:
        enhanced = enhance_image(image)
        text = extract_text(enhanced)
        fw.write("\n")
        fw.write(text)

# %%
