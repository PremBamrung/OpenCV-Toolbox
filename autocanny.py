import cv2
import numpy as np
from matplotlib import pyplot as plt

def load_image():
    img=cv2.imread("Scanned2.jpg", 0) # high res
    return img

def auto_canny(sigma=0.50):
    img = load_image()
    # compute the median of the single channel pixel intensities
    v = np.median(img)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(img, lower, upper)
    plt.imshow(edged, 'gray')
    plt.title("Auto threshold")
    plt.show()


auto_canny()


def auto_canny_otsu():
    img =  load_image()
    otsu_thresh_val, _ = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    canny = cv2.Canny(img, otsu_thresh_val, otsu_thresh_val*0.5)
    plt.imshow(canny, 'gray')
    plt.title("Auto threshold with Otsu")
    plt.show()


auto_canny_otsu()


def normal_canny():
    img =  load_image()
    canny = cv2.Canny(img, 50, 100)
    plt.imshow(canny, 'gray')
    plt.title("Normal threshold (min=50, max=100)")
    plt.show()
normal_canny()