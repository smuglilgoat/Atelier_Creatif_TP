
import numpy as np
import cv2
from rich.console import Console
console = Console()


def calculateHist(image):
    hist = np.zeros(256)
    tmp = np.unique(image, return_counts=True)
    for i in range(0, len(tmp[0])):
        hist[int(tmp[0][i])] = tmp[1][i]
    return hist


def calculateHOG(array):
    hog = np.zeros(9)
    for i in array:
        index = int((i+90)/22.5)
        hog[index] += 1
    return hog

def getPixel(img, center, x, y):
    value = 0
    try:
        if img[x][y] >= center:
            value = 1
    except:
        pass
    return value

def calculateLBP(img):
    img = np.pad(img, 1, mode="edge")
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


def calculateGradiant(img, x, y):
    gx = img[y+1][x]-img[y-1][x]
    gy = img[x+1][y]-img[x-1][y]
    mag = np.sqrt(gx**2+gy**2)
    dir = np.sign(gy) * 90 if gx == 0 else np.degrees(np.arctan(gy/gx))
    return mag, dir

def calculateDir(image):
    dir = []
    image = np.pad(image, 1, mode="edge")
    for j in range(1, image.shape[0]-1):
        for i in range(1,  image.shape[1]-1):
            dir.append(calculateGradiant(image, j, i)[1])
    return dir


def imgHIST(image, size):
    hist = []
    image = np.pad(image, 1, mode="edge")
    for j in range(1, image.shape[0]-1, size):
        for i in range(1, image.shape[1]-1, size):
            region = image[j:j+size, i:i+size]
            lbp = calculateLBP(region)
            hist.append(calculateHist(lbp))
    return np.array(hist)


def imgHOG(image, size):
    hog = []
    image = np.pad(image, 1, mode="edge")
    for j in range(1, image.shape[0]-1, size):
        for i in range(1, image.shape[1]-1, size):
            region = image[j:j+size, i:i+size]
            regionHOG = calculateDir(region)
            hog.append(calculateHOG(regionHOG))
    return np.array(hog)


def calculateMSE(arrA, arrB):
    arrA = arrA.flatten()
    arrB = arrB.flatten()
    return np.sqrt(((arrA - arrB)**2).mean())

def buildAI(ai):
    lbp = []
    hog = []
    for i in range(1, 13+1):
        try:
            data = cv2.imread('dataset\\'+str(i)+'.jpg')
            data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
            faces = ai.detectMultiScale(data, 1.1, 4)
            for (x, y, w, h) in faces:
                data = cv2.resize(data[y:y+h, x:x+w], (64, 128))
                dataLBP = np.copy(data)
                lbp.append(imgHIST(dataLBP, 8))
                hog.append(imgHOG(data, 8))
        except:
            console.print(i)
    return lbp, hog 

def main():
    faceDetecteAI = cv2.CascadeClassifier('classifier\\face_detection_ai.xml')
    refLBP, refHOG = buildAI(faceDetecteAI)

    webcam = cv2.VideoCapture(0)

    while True:
        _, frame = webcam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceDetecteAI.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            face = cv2.resize(gray[y:y+h, x:x+w], (64, 128))
            imgLBP = imgHIST(face, 8)
            imgHog = imgHOG(face, 8)

            error = 0
            for i in range(0, len(refLBP)):
                error += (calculateMSE(imgLBP, refLBP[i]))
                error += (calculateMSE(imgHog, refHOG[i]))
            border = (0, 0, 255)
            if error < 80:
                border = (255, 0, 0)
            console.print("Erreur MSE:", error)
            cv2.circle(frame, (x+w//2, y+h//2), 150, border, 2)
        try:
            cv2.imshow('Projet Atelier - Detection Faciale', frame)
        except:
            pass
        if cv2.waitKey(1) == 13:
            break
    webcam.release()
    cv2.destroyAllWindows()
    pass


if __name__ == "__main__":
    main()
    pass
