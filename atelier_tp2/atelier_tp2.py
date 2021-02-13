import cv2
from matplotlib import pyplot as plt
from skimage.feature import hog
import numpy as np
from rich.console import Console
console = Console()

car = cv2.imread(
    "D:\\Documents\\Code\\atelier_creatif_tp\\atelier_tp2\\car.jpg")
car = cv2.cvtColor(car, cv2.COLOR_BGR2GRAY)
cv2.imshow('Image_LBP', car)

fd, hog_image = hog(car, orientations=8, pixels_per_cell=(12, 12),
                    cells_per_block=(1, 1), visualize=True, multichannel=False)
plt.axis('off')
plt.imshow(hog_image, cmap=plt.cm.gray)


def gradiant(img, x, y):
    gx = img[y, x+1]-img[y, x-1]
    gy = img[y+1, x]-img[y-1, x]
    console.print(np.arctan(gy/gx))
    if gx == 0: return np.pi/2
    return np.arctan(gy/gx)

# def calc_mse(a, b):
#     sum = 0.0
#     for i in range(256):
#         sum = sum + pow((a[i][0] - b[i][0]), 2)
#     return sum / 256


# def compare_hist():
#     histo_ref = cv2.calcHist([select_img], [0], None, [256], [0, 256])
#     compa = cv2.compareHist(histo_img, histo_ref, cv2.HISTCMP_CHISQR)
#     console.print(histo_img.size)
#     console.print("Result Comparaison : ", calc_mse(histo_img, histo_ref))


def mouse_click(event, x, y, flags, param):
    global mouseX, mouseY
    global histo_img
    global crop_img

    if event == cv2.EVENT_LBUTTONDOWN:
        console.print("MouseX: ", x, "\t| MouseY: ", y)
        mouseX, mouseY = x, y
        crop_img = car[y-6:y+6, x-6:x+6]
        if crop_img.shape != (12, 12):
            console.print(
                ":warning: Selected Area isn't 12x12 area, Please select a correct zone :warning:", style="red on white")
        else:
            hog = []
            for x in range(4):
                for y in range(4):
                    hog.append((gradiant(crop_img, (x * 3) + 1, (y * 3) + 1) * (180/np.pi)))
            console.print("Hog angles : ", hog)
            fig = plt.figure("Selected_Output")
            ax = fig.add_subplot(1, 2, 1)
            plt.imshow(crop_img, cmap='gray')
            ax.set_title('Cropped Image')
            ax = fig.add_subplot(1, 2, 2)
            bins = [-22.5, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5]
            plt.hist(hog, bins=bins,rwidth=0.5)
            xticks = [(bins[idx+1] + value)/2 for idx, value in enumerate(bins[:-1])]
            plt.xticks(xticks, labels = [1, 2, 3, 4, 5, 6, 7, 8])
            ax.set_title('Histogramme')
            plt.show()


cv2.setMouseCallback('Image_LBP', mouse_click)

cv2.waitKey(0)
cv2.destroyAllWindows()
