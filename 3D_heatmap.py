import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

imgL = cv2.imread('right.png')
imgR = cv2.imread('left.png')

imgL_new=cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
imgR_new=cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

stereo = cv2.StereoBM_create(numDisparities = 16, blockSize = 41)
disparity = stereo.compute(imgL_new, imgR_new)

disparity = (disparity-np.min(disparity))/(np.max(disparity)-np.min(disparity))

x_l = []
y_l = []
z_l = []
for x in range(np.shape(disparity)[0]):
    for y in range(np.shape(disparity)[1]):
        x_l.append(x)
        y_l.append(y)
        z_l.append(disparity[x, y])

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
from mpl_toolkits.mplot3d import axes3d

x = np.arange(np.shape(disparity)[0])
y = np.arange(np.shape(disparity)[0])

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
X, Y = np.meshgrid(x, y)

disparity = np.array(disparity)
Z = disparity[X, Y]

surf = ax.plot_surface(X, Y, Z, cmap = cm.coolwarm, linewidth = 0, antialiased = False)
for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(.001)
    plt.show()