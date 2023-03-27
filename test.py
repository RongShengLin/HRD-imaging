import numpy as np 
Z_mid = 128
Z_min = 0
Z_max = 255
def hat_weight(z):
    if z <= Z_mid:
        return z - Z_min + 1e-3
    return Z_max + 1e-3 - z

x = np.array([[[1,2], [3, 4]], [[200, 255], [188, 190]], [[127, 122], [120, 253]]])
hat = np.vectorize(hat_weight)
print(hat(x))
time = np.array([1, 2, 3, 4])
time2 = np.zeros((4, 3, 2, 5))   #(w, h, color)
for i in range(4):
    time2[i, :, :, :] = time[i]
print(time2)