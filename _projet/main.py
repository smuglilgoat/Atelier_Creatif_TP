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


def calculateMSE(arrA, arrB):
    arrA = np.array(arrA)
    arrB = np.array(arrB)
    differenceArray = np.subtract(arrA, arrB)
    squaredArray = np.square(differenceArray)
    mse = squaredArray.mean()
    return mse


def main():
    faceClassifier = cv.CascadeClassifier('classifier\\haarcascade_frontalface_default.xml')

    with open('refLBP', 'rb') as fp:
        refLBP = pickle.load(fp)

    with open('refHOG', 'rb') as fp:
        refHOG = pickle.load(fp)

    console.print('LBP=', refLBP)
    console.print('HOG=', refHOG)

    webcam = cv.VideoCapture(0)

    while True:
        _, frame = webcam.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = faceClassifier.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=4)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            faceResized = cv.resize(face, (64, 128))
            faceLBP = getLBP(faceResized)
            faceHOG = getHOG(faceResized)

            error = calculateMSE(faceLBP, refLBP) + calculateMSE(faceHOG, refHOG)

            cv.putText(frame, str(int(error)), (x, y), cv.FONT_HERSHEY_SIMPLEX,
                       1, (255, 0, 0) if error < 3000 else (0, 0, 255), 2)
            cv.circle(frame, (x+w//2, y+h//2), 150, (255, 0, 0)
                      if error < 3000 else (0, 0, 255), 2)
        try:
            cv.imshow('Projet Atelier - Detection Faciale', frame)
        except:
            pass

        key = cv.waitKey(1)
        if key == 27:
            break

    webcam.release()
    cv.destroyAllWindows()
    pass


if __name__ == "__main__":
    main()
    pass
