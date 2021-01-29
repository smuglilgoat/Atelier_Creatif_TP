import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import random

randX = random.randint(0, 104)
randY = random.randint(0, 236)
print("Selected zone X : ", randX)
print("Selected zone Y : ", randY)

lbp = cv2.imread("D:\\Documents\\Code\\atelier_creatif_tp\\Atelier_TP1\\lbp.png")
lbp = cv2.cvtColor(lbp, cv2.COLOR_BGR2GRAY)
plt.imshow(lbp, cmap='gray')

crop_img = lbp[randY:randY+64, randX:randX+64]
plt.imshow(crop_img, cmap='gray')

histo_img = cv2.calcHist([crop_img], [0], None, [256], [0, 256])
plt.plot(histo_img)
plt.show()

found = 1000
x = 0
y = 0
histo_img = cv2.calcHist([crop_img], [0], None, [256], [0, 256])
#cv2.normalize(histo_img, histo_img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);
for i in range(104):
    for j in range(236):
        elem = lbp[j:j+64, i:i+64]
        histo_elem = cv2.calcHist([elem], [0], None, [256], [0, 256])
        #cv2.normalize(histo_elem, histo_elem, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);
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
plt.imshow(found_img, cmap='gray')

plt.plot(histo_found)
plt.show()

# def gestion_souris(event,x,y,flags,param):
#     global mouseX,mouseY
#     global histo2
#     if event == cv2.EVENT_RBUTTONDBLCLK:
#         print(x,y)
#         fen02 = img_gray[y:y+32, x:x+32]      # ]/ y ensuite x
#         histo2= plt.hist(fen02.ravel(),256,[0,256]);
#         plt.show()
#         print (diff)


#         mouseX,mouseY = x,y

# cv2.setMouseCallback('Fenetre',gestion_souris)
