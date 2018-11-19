# from tkinter import *
from PIL import Image, ImageTk
import cv2
import fitz
import numpy as np


def resize_image(image, size=(700, 700)):
    """
    Resizes given image to given size.
    :param image: image as binary numpy array from 'planData'
    :param size: new size in px (width, heigth).
    :return: tkPhotoImage with specified size
    """
    if len(image.shape) == 3:
        image = image[:, :, 0]

    h_image, w_image = image.shape
    koeff1 = size[0] / w_image
    koeff2 = size[1] / h_image

    # --> Finding the correct coefficient of Width to Height
    if koeff1 <= koeff2:
        koeff = koeff1
    else:
        koeff = koeff2

    # --> Resizing the image
    w_resized = round(w_image * koeff)
    h_resized = round(h_image * koeff)

    resized_image = cv2.resize(image, (w_resized, h_resized))
    pil_image = Image.fromarray(resized_image)
    photo_cut = ImageTk.PhotoImage(pil_image)

    return photo_cut


def read_file(file):
    """
    Reads single file (image data) and turns it into a binary image array with 200 dpi.
    :param file: Location of image data as string.
    :return: binary image array (rotated if neccessary), rotated flag (true or false)
    """
    extensions = [".tif", "tiff", ".jpg", "jpeg", ".png"]
    rotated = 0
    if file[-3:].lower() == "pdf":
        # noinspection PyUnresolvedReferences
        doc = fitz.open(file)
        page = doc.loadPage(0)
        dpi = 200.0
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.getPixmap(matrix=mat, colorspace=fitz.csGRAY, alpha=0)
        pic = Image.frombytes("L", [pix.width, pix.height], pix.samples)
        _, img = cv2.threshold(np.array(pic), 140, 255, cv2.THRESH_BINARY)
    elif file[-4:].lower() in extensions:
        imgT = Image.open(file)
        dpi = imgT.info['dpi'][0].numerator
        zoom = 200.0 / dpi
        imgT.close()
        img = cv2.imread(file)
        img = cv2.resize(img, None, fx=zoom, fy=zoom)
        _, img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 140, 255, cv2.THRESH_BINARY)
    else:
        print('File could not be loaded')
        return [], 0

    if img is not []:
        if img.shape[0] > img.shape[1]:
            img = np.rot90(img)
            rotated = 1

    return img, rotated, dpi


def countID(ID):
    """
    Increases given ID by 1
    :param ID: Unique identifier for plans (P), results (R) and symbols (S) as string: 'X000'
    :return: new ID
    """
    num = [int(ID[1]), int(ID[2]), int(ID[3])]
    count = num[2] + 1
    if count < 10:
        num[2] = count
    else:
        count2 = num[1] + 1
        if count2 < 10:
            num[1] = count2
            num[2] = 0
        else:
            count3 = num[0] + 1
            if count3 < 10:
                num[0] = count3
                num[1] = 0
                num[2] = 0
            else:
                print('Can\'t upload more than 999 plans')

    return ID[0] + str(num[0]) + str(num[1]) + str(num[2])
