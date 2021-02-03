import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import random
from rich.console import Console
console = Console()

randX = random.randint(32, 268)
randY = random.randint(32, 136)
console.print("Selected zone X : ", randX)
console.print("Selected zone Y : ", randY)

lbp = cv2.imread("D:\\Documents\\Code\\atelier_creatif_tp\\Atelier_TP1\\lbp.png")
lbp = cv2.cvtColor(lbp, cv2.COLOR_BGR2GRAY)
cv2.imshow('Image_LBP',lbp)

plt.figure('Selected Area')
select_img = lbp[randY-32:randY+32, randX-32:randX+32]
plt.imshow(select_img, cmap = 'gray')
plt.show()


def calc_mse(a, b):
    sum = 0.0
    for i in range(256):
        sum = sum + pow((a[i][0] - b[i][0]), 2)
    return sum / 256
    
def compare_hist():
    histo_ref = cv2.calcHist([select_img], [0], None, [256], [0, 256])
    # console.print("histo_ref :", histo_ref)
    # console.print("histo_img :", histo_img)
    compa = cv2.compareHist(histo_img, histo_ref, cv2.HISTCMP_CHISQR)
    console.print("Result Comparaison : ", calc_mse(histo_img, histo_ref))
    
def mouse_click(event,x,y,flags,param):
    global mouseX,mouseY
    global histo_img
    global crop_img

    if event == cv2.EVENT_LBUTTONDOWN:
        console.print("MouseX: ", x, "\t| MouseY: ", y)
        mouseX,mouseY = x,y
        crop_img = lbp[y-32:y+32, x-32:x+32]
        if crop_img.shape != (64, 64):
            console.print(":warning: Selected Area isn't 64x64 image, Please select a correct zone :warning:", style="red on white")
        else:
            histo_img = cv2.calcHist([crop_img], [0], None, [256], [0, 256])
            compare_hist()
            fig = plt.figure("Selected_Output")
            ax = fig.add_subplot(1, 2, 1)
            plt.imshow(crop_img, cmap = 'gray')
            ax.set_title('Cropped Image')
            ax = fig.add_subplot(1, 2, 2)
            plt.plot(histo_img)
            ax.set_title('Image Histogramme')
            plt.show()
        

cv2.setMouseCallback('Image_LBP', mouse_click)

cv2.waitKey(0)
cv2.destroyAllWindows()



# plt.plot(histo_found)
# plt.show()

# float MSE(int[] hist1, int[] hist2) {
#   float somme = 0;
#   for (int i=0; i < 256; i++) {
#     somme += Math.pow(hist1[i]-hist2[i], 2);
#   }
  
#   return somme /= 256.0;
# }

