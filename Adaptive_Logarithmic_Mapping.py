import cv2
import random as rnd
import numpy as np
import math
import numpy.linalg as linalg
import matplotlib.pyplot as plt

def gamma_correction(pixel):
    if pixel <= 0.0031308:
        return pixel * 12.92
    else :
        return 1.055 * (pixel**1/2.4) - 0.055
        #return pixel * 2
    
def Adaptive_Logarithmic_Mapping(HDR_image, b = 0.85, L_dmax = 100):
    width, height, _ = HDR_image.shape
    CIE_image = cv2.cvtColor(HDR_image, cv2.COLOR_BGR2XYZ)

    L_avg = np.sum(CIE_image[:,:,1])/(width*height)
    _, L_wmax, _, _ = cv2.minMaxLoc(CIE_image[:,:,1])

    L_wmax /= L_avg
    multiplier = L_dmax*0.01/math.log(L_wmax+1, 10)
    bias = math.log(b)/math.log(0.5)

    LDR_image = HDR_image.copy()
    for i in range(width):
        for j in range(height):
            Y_w = CIE_image[i][j][1]/L_avg
            L_d = multiplier * math.log(Y_w + 1) * math.log(2 + ((Y_w/L_wmax)**bias) * 8) 
            scale = L_d/CIE_image[i][j][1]
            LDR_image[i][j][0] = gamma_correction(scale*CIE_image[i][j][0])
            LDR_image[i][j][1] = gamma_correction(scale*CIE_image[i][j][1])
            LDR_image[i][j][2] = gamma_correction(scale*CIE_image[i][j][2])

    LDR_image = cv2.cvtColor(LDR_image, cv2.COLOR_XYZ2BGR)
    return LDR_image


