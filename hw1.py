import cv2
import random as rnd
import numpy as np
import math
import numpy.linalg as linalg
import matplotlib.pyplot as plt


def hat_weight(z):
    if z <= Z_mid:
        return z - Z_min
    return Z_max - z

#open file
#Note:It is just a sample, and should be more user-friendly
filenames = input().split()
time_str = input().split()
time = []
for t in time_str:
    time.append(1/float(t))
print(time)
time_log = np.log(time)
print(time_log)
imgs = []
p = 0
for file in filenames:
    imgs.append(cv2.imread(file))
    p += 1
#TODO Image alignment(can pass)

#sampling 50 pixels
w, h, _ = imgs[0].shape
N = 50
sample_point = []
for i in range(50):
    x = rnd.randint(w / 8, w * 7 / 8)
    y = rnd.randint(h / 8, h * 7 / 8)
    sample_point.append((x, y))
print(sample_point)
#parameters
Z_max, Z_min, Z_mid = 255, 0, 127
ld = 20
A = np.zeros((3, N * p + 255, 256 + N), dtype=float)
b = np.zeros((3, N * p + 255), dtype=float)
#fill in metrix A and array b
sol = []
for color in range(3):
    for i in range(N * p + 255):
        if i <= (N * p - 1):
            P = i // N
            n = i % N
            Z_ij = imgs[P][sample_point[n][0]][sample_point[n][1]][color]
            W_zij = hat_weight(Z_ij)
            A[color][i][Z_ij] = W_zij
            A[color][i][n + 256] = -W_zij
            b[color][i] = time_log[P] * W_zij
        elif i != N * p:
            ld_power2 = ld ** 2
            w_z = hat_weight(i - (N * p + 1) + 1)
            A[color][i][i - (N * p + 1)] = ld_power2 * w_z
            A[color][i][i - (N * p + 1) + 1] = -2 * ld_power2 * w_z
            A[color][i][i - (N * p + 1) + 2] = ld_power2 * w_z
        else:
            A[color][i][Z_mid] = 1
    #print(b[color])
    #print(A[color])
    #least square solution
    x, res, rank, s  = linalg.lstsq(A[color], b[color], rcond=None)
    print(x, len(x))
    sol.append(x)
    for n_ in range(256):
        plt.plot(x[n_], n_, 'o', label='solution', markersize=5)
    plt.show()
#generate HDR
hdr_img = np.zeros((w, h, 3), dtype=float)
for color in range(3):
    for i in range(w):
        for j in range(h):
            sum_E, sum_w = 0, 0
            for k in range(p):
                z = imgs[k][i][j][color]
                w_z = hat_weight(Z_ij)
                g_z = sol[color][z]
                sum_E += (w_z * (g_z - time_log[k]))
                sum_w += w_z
            hdr_img[i][j][color] = pow(np.e, (sum_E / sum_w))
        print('complete', i, color)
cv2.imwrite("test.hdr", hdr_img.astype(np.float32))