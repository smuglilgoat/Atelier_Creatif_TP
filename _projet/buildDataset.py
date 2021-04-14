import cv2 as cv
import numpy as np
from skimage.feature import hog
import pickle
from rich.console import Console
console = Console()


def getPixel(img, center, x, y):
    value = 0
    if img[x][y] >= center:
        value = 1
    return value


def calculateLBP(img):
    lbp = np.zeros(img.shape)
    for i in range(1, lbp.shape[0]-1):
        for j in range(1,  lbp.shape[1]-1):
            center = img[i][j]
            valueArray = []
            valueArray.append(getPixel(img, center, i-1, j+1))     # top_right
            valueArray.append(getPixel(img, center, i, j+1))       # right
            valueArray.append(getPixel(img, center, i+1, j+1))     # bottom_right
            valueArray.append(getPixel(img, center, i+1, j))       # bottom
            valueArray.append(getPixel(img, center, i+1, j-1))     # bottom_left
            valueArray.append(getPixel(img, center, i, j-1))       # left
            valueArray.append(getPixel(img, center, i-1, j-1))     # top_left
            valueArray.append(getPixel(img, center, i-1, j))       # top
            lbp[i][j] = np.sum(np.multiply(
                valueArray, [1, 2, 4, 8, 16, 32, 64, 128]))
    return lbp[1:lbp.shape[0]-1:1, 1:lbp.shape[1]-1]


def getLBP(img):
    img = np.pad(img, 1, mode="edge")
    lbp = calculateLBP(img)
    hist, bins = np.histogram(lbp.ravel(), 256, [0, 256])
    return np.array(hist)


def getHOG(img):
    hogHist = hog(img, orientations=9, pixels_per_cell=(
        8, 8), cells_per_block=(1, 1))
    return np.array(hogHist)


def main():
    faceClassifier = cv.CascadeClassifier('classifier\\haarcascade_frontalface_default.xml')
    refLBP = []
    refHOG = []

    for i in range(1, 13+1):
        image = cv.imread('dataset\\'+str(i)+'.jpg')
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        faces = faceClassifier.detectMultiScale(image, scaleFactor=1.05, minNeighbors=4)
        for (x, y, w, h) in faces:
            face = image[y:y+h, x:x+w]
            faceResized = cv.resize(face, (64, 128))
            refLBP.append(getLBP(faceResized))
            refHOG.append(getHOG(faceResized))

    with open('refLBP', 'wb') as fp:
        pickle.dump(refLBP, fp)

    with open('refHOG', 'wb') as fp:
        pickle.dump(refHOG, fp)

    console.print('LBP=', refLBP)
    console.print('HOG=', refHOG)
    pass


if __name__ == "__main__":
    main()
    pass
