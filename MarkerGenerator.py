import cv2
import numpy as np
import random as rnd

# hepsi tam sayı olmalı
quantity = 30
markerCodeSizeInBits = 5
borderSizeInBits = 1
pixelSizeEachBit = 50

# 0-1 arasında olmalı
alpha = 0.5


lenght = (markerCodeSizeInBits + 2 * borderSizeInBits) * pixelSizeEachBit
for i in range(quantity):
    artag_img = np.zeros((lenght, lenght, 1),dtype=float)

    # her bit için...
    for i in range(borderSizeInBits, markerCodeSizeInBits+borderSizeInBits):
        for j in range(borderSizeInBits, markerCodeSizeInBits+borderSizeInBits):
            # eğer alphadan büyük çıkar ise (i,j) deki bit 0'dır
            if rnd.random() >= alpha:
                continue

            # biti 1 çıkan kısmı belli boyutlara göre dolduruyoruz
            for x in range(pixelSizeEachBit):
                ix = i * pixelSizeEachBit + x
                for y in range(pixelSizeEachBit):
                    iy = j * pixelSizeEachBit + y
                    artag_img[ix][iy] = 255

    cv2.imshow("xd",artag_img)
    cv2.waitKey(500)
