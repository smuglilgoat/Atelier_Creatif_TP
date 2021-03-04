import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from rich.console import Console
console = Console()

def getPixel(img, center, x, y):
    value = 0
    try:
        if img[x][y] >= center:
            value = 1
    except:
        pass
    return value

def calculLBP(img, x, y):
    center = img[x][y]
    valueArray = []
    valueArray.append(getPixel(img, center, x-1, y+1))     # top_right
    valueArray.append(getPixel(img, center, x, y+1))       # right
    valueArray.append(getPixel(img, center, x+1, y+1))     # bottom_right
    valueArray.append(getPixel(img, center, x+1, y))       # bottom
    valueArray.append(getPixel(img, center, x+1, y-1))     # bottom_left
    valueArray.append(getPixel(img, center, x, y-1))       # left
    valueArray.append(getPixel(img, center, x-1, y-1))     # top_left
    valueArray.append(getPixel(img, center, x-1, y))       # top

    factorArray = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for i in range(len(valueArray)):
        val += valueArray[i] * factorArray[i]
    return val

def main():
    img = cv.imread('car.jpg')
    img = cv.resize(img, (128, 128))
    height, width, channel = img.shape
    console.print('Image=', img)
    console.print('Height=', height, ', Width=', width, ', Channel=', channel)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    result_lbp = []

    for i in range(16):
        for y in range(16):
            img_cropped = gray[8 * i: 8 * i + 8, 8 * y: 8 * y + 8]
            for i in range(8):
                for j in range(8):
                    result_lbp.append(calculLBP(img_cropped, i, j))
            
    result_lbp = np.array(result_lbp)
    console.print('LBP=', result_lbp)

    fig = plt.figure("Image LBP")
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(gray, cmap='gray')
    ax.set_title('Image')
    ax = fig.add_subplot(1, 2, 2)
    plt.hist(result_lbp, bins=256)
    ax.set_title('LBP Histogramme')
    plt.show()

if __name__ == "__main__":
    main()