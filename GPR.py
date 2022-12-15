#!/usr/bin/env python
import numpy as np
# Data import
import pandas
# Plotting
import matplotlib.pyplot as plt
# Gaussian Process Regression library
import GPy
# required for ros map import
import yaml
import cv2


"""
Gaussian Process Regression to interpolate point gamma radiation count data collected by a mobile UGV into a 2D representation.
This uses the openly available GPy library to perform the regression.
"""

# Directory for file storage
directory = "/home/rail/PycharmProjects/GPR/JSI/"
input_file = "JSI_radiation_data.txt"
output_file = "JSI_2Drad_GPR.txt"

# Import raw data
# File format = timestamp, x, y, z, counts (per second)
# Only extract the necessary columns of x, y, and counts
data_raw = np.array(pandas.read_csv(directory + input_file, delimiter=',', header=0, usecols=['x','y','counts']))

# Extract the x and y position values as "X" inputs for regression, along with count values as "Y" in the regression
X = np.array(data_raw[:,0:2])   # x, y
Y = np.array([data_raw[:,2]]).T # counts
print("Data Ready")


"""
Gaussian Process Regression
Use a matern kernel (radiation fluctuations due to sources) with a bias term (due to homogenous background radiation)
Define individual kernel primitives with:
input_dim=2 - i.e. x and y coordinates
variance - starting guess at the magnitude of the contribution (will be altered by the regression)
lengthscale = 0.5 - how closely correlated data is in space, start at 0.5 m (will be altered by the regression)
ARD=False - the kernel is NOT independant in all dimensions (because we expect the function to work radially)
"""
# As the data is counts, the error (variance) associated is equal to the mean - this is not the case for gaussians at small values
# A poisson approximation is used to tackle low count rate issues
poisson_likelihood = GPy.likelihoods.Poisson()
#poisson_likelihood = GPy.likelihoods.Gaussian()
laplace_inf = GPy.inference.latent_function_inference.Laplace()

#k1 = GPy.kern.Matern32(input_dim=2, variance=0.54, lengthscale=0.95, ARD=False)
#k1 = GPy.kern.Matern52(input_dim=2, variance=0.8, lengthscale=0.5, ARD=False)
k1 = GPy.kern.RBF(input_dim=2, variance=0.54, lengthscale=0.95, ARD=False)
#k1 = GPy.kern.Exponential(input_dim=2, variance=0.8, lengthscale=0.5, ARD=False)
k2 = GPy.kern.Bias(input_dim = 2, variance = 0.6)
kernel = k1 + k2 # Combine the two kernel primitives
print("Kernel Initialised")


# Build the regression model
# Resources: https://nbviewer.jupyter.org/github/SheffieldML/notebook/blob/master/GPy/Poisson%20regression%20tutorial.ipynb and older https://notebook.community/SheffieldML/notebook/GPy/Poisson%20regression%20tutorial
m = GPy.core.GP(X=X, Y=Y, likelihood=poisson_likelihood, inference_method=laplace_inf, kernel=kernel)
print("Pre-optimisation:")
print(m)
# Optimize model and automatically plot coarse representation
m.optimize(messages=True,max_f_eval = 1000)

print("Optimised:")
print(m)
m.plot_f()
plt.show()

# Predict values based on map locations generated by robot SLAM
# Define map files
mapFile = "JSI_SLAM_map.pgm"
metaFile = "JSI_SLAM_map.yaml"

FREESPACE = 254
OCCUPIED = 0
UNKNOWN = 205

# Import Trinary (three values only) map file - 3 values are permitted for free space (254), occupied space (0), unknown space (205) - I think!
print("Reading ROS SLAM Map")
try:
    map = cv2.imread(directory + mapFile, -1)
except cv2.error as ecv:
    print(ecv)

print("Map Width (x): ", map.shape[1], " , Map Height (y): ", map.shape[0])

# Import metadata file
"""
METADATA INFO
origin = bottom lefthand corner of the image in x, y, z metric coordinates in metres
resolution = metric size per cell in metres
"""
with open(directory + metaFile, 'r') as stream:
    try:
        metadata = yaml.safe_load(stream)
    except yaml.YAMLERROR as exc:
        print(exc)

print("Map Metadata:")
for key in metadata.keys():
    print(key, metadata[key])

#Generate grid based on map
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



# Predict the values of mean and variance at different times
pred_points = grid #np.linspace(0,24,100)[:, None]
print("Generated Grid Of Points: " + str(grid.shape[0]))
# Predictive GP for log intensity mean and variance
print("Sampling At Grid Points")
f_mean, f_var = m._raw_predict(pred_points)
f_mean = np.exp(f_mean)

# Plot GPR output on map
# Adjust plot coordinates based on size and resolution of the SLAM map
x_values = np.arange(0, map.shape[1])*metadata["resolution"] + metadata["origin"][0]
y_values = np.arange(0, map.shape[0])*metadata["resolution"] + metadata["origin"][1]
implot = plt.imshow(map, cmap='gray', extent = [np.min(x_values), np.max(x_values), np.min(y_values), np.max(y_values)], aspect='equal')
plt.scatter(x = pred_points[:,0], y = pred_points[:,1], c = f_mean, marker="s", s = 1.0, vmin=0.0, vmax=1.2*max(f_mean), cmap='jet')
plt.xlim([-20.0,6.0])
plt.ylim([-12,12])
plt.colorbar()
plt.show()


# 지도에 방사선 측정 데이터 출력하는 plot 생성
xvs = X[:,0]
yvs = X[:,1]
value = np.array([data_raw[:,2]])
implot = plt.imshow(map, cmap='gray', extent = [np.min(x_values), np.max(x_values), np.min(y_values), np.max(y_values)], aspect='equal')
plt.scatter(x=xvs, y=yvs, cmap='jet', s=5, c=value)
plt.xlim([-20.0,6.0])
plt.ylim([-12,12])
plt.colorbar()
plt.show()


f_upper, f_lower = f_mean + 2*np.sqrt(f_var), f_mean - 2.*np.sqrt(f_var)

gpy_output = np.c_[grid, np.exp(f_mean), np.exp(f_var), np.exp(f_upper), np.exp(f_lower)]

# Locally save file of data for testing
f = open(directory + output_file, "wt") # Should clear existing file of same name
f.close()
# Could be replaced with Pandas approach
np.savetxt(directory + output_file, gpy_output, delimiter=",", header='"x","y","mean","variance","upper_var","lower_var"', comments='')