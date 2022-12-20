import numpy as np
import pandas
import matplotlib.pyplot as plt
import GPy
import yaml
import cv2
from mpl_toolkits.mplot3d import Axes3D
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

directory = "C:/Users/rail/PycharmProjects/GaussianProcessRegression/LUNL/"
input_file = "LUNL_radiation_data.txt"
output_file = "LUNL_2Drad_GPR.txt"

data_raw = np.array(pandas.read_csv(directory + input_file, delimiter=',', header=0, usecols=['x','y','counts']))
X = np.array(data_raw[:, 0:2])
Y = np.array([data_raw[:, 2]]).T
print("Data Ready")

poisson_likelihood = GPy.likelihoods.Poisson()
laplace_inf = GPy.inference.latent_function_inference.Laplace()

kernel = 1 * RBF(length_scale=1.0)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
gp.fit(X, Y)
print(gp.kernel_)

mapFile = "LUNL_SLAM_map.pgm"
metaFile = "LUNL_SLAM_map.yaml"

FREESPACE = 254
OCCUPIED = 0
UNKNOWN = 205

print("Reading ROS SLAM Map")
try:
    map = cv2.imread(directory + mapFile, -1)
except cv2.error as ecv:
    print(ecv)

print("Map Width (x) : ", map.shape[1], " , Map Height (y) : ", map.shape[0])

with open(directory + metaFile, 'r') as stream:
    try:
        metadata = yaml.safe_load(stream)
    except yaml.YAMLERROR as exc:
        print(exc)

print("Map Metadata : " )
for key in metadata.keys():
    print(key ,metadata[key])

freespaecOnly = True;

if freespaecOnly:
    cells = np.where(map == FREESPACE)
    grid = np.array(list(zip(cells[1], (map.shape[0] - cells[0]))))
    grid = grid * metadata["resolution"]
    grid[:,0] = grid[:,0] + metadata["origin"][0]
    grid[:,1] = grid[:,1] + metadata["origin"][1]
    grid[np.where((grid[:,0]<=2.5) & (grid[:,1]<=0.0)), :]
else:
    xnew = np.arange(0, map.shape[1])*metadata["resolution"] + metadata["origin"][0]
    ynew = np.arange(0, map.shape[0])*metadata["resolution"] + metadata["origin"][1]
    xv, yv = np.meshgrid(xnew, ynew)
    grid = np.dstack((xv.ravel(), yv.ravel()))[0]


pred_points = grid
print("Generated Grid of Points : " + str(grid.shape[0]))
print("Sampling At Grid Points")
mean_prediction, std_prediction = gp.predict(pred_points, return_std=True)
mean_prediction = np.exp(mean_prediction)

print(max(mean_prediction))
print(min(mean_prediction))
'''
x_values = np.arange(0, map.shape[1]) * metadata["resolution"] + metadata["origin"][0]
y_values = np.arange(0, map.shape[0]) * metadata["resolution"] + metadata["origin"][1]
implot = plt.imshow(map, cmap='gray', extent=[np.min(x_values), np.max(x_values), np.min(y_values), np.max(y_values)], aspect='equal')
plt.scatter(x = pred_points[:, 0], y = pred_points[:, 1], c = mean_prediction, marker='s', s = 1.0, vmin=0.0, vmax=1.2*max(mean_prediction), cmap='jet')
plt.xlim([-4, 4])
plt.ylim([-8, 0])
plt.colorbar()
plt.show()
'''