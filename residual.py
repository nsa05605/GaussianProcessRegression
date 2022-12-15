#!/usr/bin/env python
import matplotlib.colors
import numpy as np
import pandas
import matplotlib.pyplot as plt
import GPy
import yaml
import cv2
from sklearn.preprocessing import StandardScaler

directory = "/home/nsa05605/PycharmProjects/GPR/"
RBF_file = "RBF.txt"
MATERN_file = "MATERN.txt"
output_file = "residual.txt"

coord_raw = np.array(pandas.read_csv(directory + RBF_file, delimiter=',', header=0, usecols=['x','y'])) # x, y 좌표
RBF_raw = np.array(pandas.read_csv(directory + RBF_file, delimiter=',', header=0, usecols=['mean']))
MATERN_raw = np.array(pandas.read_csv(directory + MATERN_file, delimiter=',', header=0, usecols=['mean']))

residual = RBF_raw-MATERN_raw
#scalar = StandardScaler()
#df = residual.copy()
#df[:] = scalar.fit_transform(df[:])

print(max(RBF_raw))

#print(residual)
#print(df)
#print(min(residual)) # -4.5
#print(max(residual)) #  7.2

# print(RBF_raw-MATERN_raw)   # 차이

mapFile = "/JSI/JSI_SLAM_map.pgm"
metaFile = "/JSI/JSI_SLAM_map.yaml"

FREESPACE = 254
OCCUPIED = 0
UNKNOWN = 205

try:
    map = cv2.imread(directory + mapFile, -1)
except cv2.error as ecv:
    print(ecv)

with open(directory + metaFile, 'r') as stream:
    try:
        metadata = yaml.safe_load(stream)
    except yaml.YAMLERROR as exc:
        print(exc)

for key in metadata.keys():
    print(key, metadata[key])

freespaceOnly = True; #Generate grid coordinates only for cells marked as free space on the map

if freespaceOnly:
    # X (width) = columns (index 1), Y (height) = rows (index 0)
    cells = np.where(map == FREESPACE)
    grid = np.array(list(zip(cells[1], (map.shape[0] - cells[0])))) # nrows - yIdx gives y axis in the correct direction
    grid = grid*metadata["resolution"]
    grid[:,0] = grid[:,0] + metadata["origin"][0] #X positions
    grid[:,1] = grid[:,1] + metadata["origin"][1] #Y positions
    #Only include values of y < 0.0 m and x < 2.5 m to avoid wasting time on open areas of LU neutron lab not plotted
    grid[np.where(  (grid[:,0]<=2.5) & (grid[:,1]<=0.0)  ),:]
else: #Else generate a grid including all cells
    xnew = np.arange(0, map.shape[1])*metadata["resolution"] + metadata["origin"][0]
    ynew = np.arange(0, map.shape[0])*metadata["resolution"] + metadata["origin"][1]
    xv, yv = np.meshgrid(xnew,ynew)
    grid = np.dstack((xv.ravel(), yv.ravel()))[0]

pred_points = grid #np.linspace(0,24,100)[:, None]

f_mean = residual

x_values = np.arange(0, map.shape[1])*metadata["resolution"] + metadata["origin"][0]
y_values = np.arange(0, map.shape[0])*metadata["resolution"] + metadata["origin"][1]
implot = plt.imshow(map, cmap='gray', extent = [np.min(x_values), np.max(x_values), np.min(y_values), np.max(y_values)], aspect='equal')
plt.scatter(x = pred_points[:,0], y = pred_points[:,1], c = f_mean, marker="s", s = 1.0, vmin=-7.5, vmax=7.5, cmap='seismic')
plt.xlim([-20.0,6.0])
plt.ylim([-12,12])
plt.colorbar()
plt.show()