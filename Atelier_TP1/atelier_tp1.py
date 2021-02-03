import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

lbp = cv2.imread("D:\\Documents\\Code\\atelier_creatif_tp\\Atelier_TP1\\lbp.png")
lbp = cv2.cvtColor(lbp, cv2.COLOR_BGR2GRAY)
cv2.imshow('Image_LBP',lbp)
def find_seg():
    found = 1000
    x = 0
    y = 0
    histo_img = cv2.calcHist([crop_img], [0], None, [256], [0, 256])
    for i in range(104):
        for j in range(236):
            elem = lbp[j:j+64, i:i+64]
            histo_elem = cv2.calcHist([elem], [0], None, [256], [0, 256])
            compa = cv2.compareHist(histo_img, histo_elem, cv2.HISTCMP_CHISQR)
            if(compa < found):
                found = compa
                histo_found = histo_elem
                x = i
                y = j
    print("Result X : ", x)
    print("Result Y : ", y)
    print("Result correlation : ", found)
    found_img = lbp[y:y+64, x:x+64]
    plt.figure(1)
    plt.imshow(found_img, cmap='gray')
    
def mouse_click(event,x,y,flags,param):
    global mouseX,mouseY
    global histo_img
    global crop_img

    if event == cv2.EVENT_LBUTTONDOWN:
        print("MouseX: ", x, "\t| MouseY: ", y)
        mouseX,mouseY = x,y
        crop_img = lbp[y:y+64, x:x+64]
        histo_img = cv2.calcHist([crop_img], [0], None, [256], [0, 256])
        cv2.imshow('Cropped',crop_img)
        plt.figure(0)
        plt.plot(histo_img)
        find_seg()
        plt.show()
        

cv2.setMouseCallback('Image_LBP', mouse_click)

cv2.waitKey(0)
cv2.destroyAllWindows()



# plt.plot(histo_found)
# plt.show()


