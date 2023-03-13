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

import numpy.random


"""
Gaussian Process Regression to interpolate point gamma radiation count data collected by a mobile UGV into a 2D representation.
This uses the openly available GPy library to perform the regression.
"""

# Directory for file storage
directory = "C:/Users/jihun/PycharmProjects/GaussianProcessRegression/LUNL/"
input_file = "LUNL_radiation_data.txt"
# output_file = "LUNL_2Drad_GPR.txt"

# Import raw data
# File format = timestamp, x, y, z, counts (per second)
# Only extract the necessary columns of x, y, and counts
data_raw = np.array(pandas.read_csv(directory + input_file, delimiter=',', header=0, usecols=['x','y','z','counts']))

# Extract the x and y position values as "X" inputs for regression, along with count values as "Y" in the regression
X = np.array(data_raw[:,0:2])   # x, y
Y = np.array([data_raw[:,3]]).T # counts

### 여기를 이제 바꿀 예정
X_new = np.zeros((201, 2))
Y_new = np.zeros((201, 1))

'''
## 방법 1
## 2개 중 1개씩 샘플링
for i in range(201):
    X_new[i] = X[i*2+1]
    Y_new[i] = Y[i*2+1]

output_file = "compare_Method1_Mat32_LUNL_2Drad_GPR.txt"
'''

'''
## 방법 2
## 높은 부분에 집중해서 샘플링
## 10개씩 담는 슬라이딩 윈도우 형식으로 데이터를 나누고, 해당 윈도우의 값들의 합에 따라 샘플링의 수를 정하는 방법


np.random.seed(0)
rand_arr = np.array([0,1,2,3,4,5,6,7,8,9])
result_arr = np.zeros(10)

X_idx = 0   # X_new, Y_new에 담긴 idx
n_samples = 0   # 각 슬라이딩 윈도우에서 뽑을 샘플의 수
for sidx in range(40):
    sum_window = 0  # 각 슬라이딩 윈도우의 측정값의 합
    for midx in range(10):
        sum_window += Y[sidx*10 + midx]
    if sum_window < 20:
        n_samples = 2
    elif sum_window < 40:
        n_samples = 4
    elif sum_window < 60:
        n_samples = 6
    else:
        n_samples = 7
    result_arr = np.random.choice(rand_arr, n_samples, replace=False)

    for i in range(n_samples):
        X_new[X_idx] = X[sidx*10 + result_arr[i]]
        Y_new[X_idx] = Y[sidx*10 + result_arr[i]]
        X_idx += 1

print(X_idx)    # 110

X_new = X_new[:X_idx, ] # 추출한 샘플링 수만큼 크기 맞춰주기
Y_new = Y_new[:X_idx, ]

output_file = "compare_Method2_Mat32_LUNL_2Drad_GPR.txt"
'''


## 방법 3
## 그냥 랜덤으로 샘플링
np.random.seed(0)
rand_arr = np.arange(0, 403, 1)
result_arr = np.random.choice(rand_arr, 201, replace=False)

for i in range(201):
    X_new[i] = X[result_arr[i]]
    Y_new[i] = Y[result_arr[i]]

output_file = "compare_Method3_Invsq_LUNL_2Drad_GPR.txt"



print("Data Ready")

print(X.shape)  # (403, 2)
print(Y.shape)  # (403, 1)

print(X_new.shape)
print(Y_new.shape)



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

linear      = GPy.kern.Linear(input_dim=2, variances=1.0)
matern32    = GPy.kern.Matern32(input_dim=2, variance=1.0, lengthscale=1.0)
matern52    = GPy.kern.Matern52(input_dim=2, variance=1.0, lengthscale=1.0)
rbf         = GPy.kern.RBF(input_dim=2, variance=1.0, lengthscale=1.0)
exponential = GPy.kern.Exponential(input_dim=2, variance=1.0, lengthscale=1.0)
ratquad     = GPy.kern.RatQuad(input_dim=2, variance=1.0, lengthscale=1.0)
expquad     = GPy.kern.ExpQuad(input_dim=2, variance=1.0, lengthscale=1.0)

myInvSquare = GPy.kern.MyInvSquare(input_dim=2, variance=1.0, lengthscale=1.0)
myAddMatExp = GPy.kern.MyAddMat32Exp(input_dim=2, variance=1.0, lengthscale=1.0)
myMulMatExp = GPy.kern.MyMulMat32Exp(input_dim=2, variance=1.0, lengthscale=1.0)

bias = GPy.kern.Bias(input_dim=2, variance=0.3)
# kernel = myAddMatExp + bias
# kernel = myMulMatExp + bias
kernel = myInvSquare + bias
# kernel = matern32 + bias

print("Kernel Initialised")

# Build the regression model
# Resources: https://nbviewer.jupyter.org/github/SheffieldML/notebook/blob/master/GPy/Poisson%20regression%20tutorial.ipynb and older https://notebook.community/SheffieldML/notebook/GPy/Poisson%20regression%20tutorial
# m = GPy.core.GP(X=X, Y=Y, likelihood=poisson_likelihood, inference_method=laplace_inf, kernel=kernel)

m = GPy.models.GPRegression(X=X_new, Y=Y_new, kernel=kernel)
m.likelihood = poisson_likelihood
m.inference_method = laplace_inf

#print("Pre-optimisation:")
#print(m)
# Optimize model and automatically plot coarse representation
m.optimize(messages=True,max_f_eval = 3000)

# print("Optimised:")
# print(m)
#m.plot_f()
#plt.show()

# Predict values based on map locations generated by robot SLAM
# Define map files
mapFile = "LUNL_SLAM_map.pgm"
metaFile = "LUNL_SLAM_map.yaml"

FREESPACE = 254
OCCUPIED = 0
UNKNOWN = 205

# Import Trinary (three values only) map file - 3 values are permitted for free space (254), occupied space (0), unknown space (205) - I think!
# print("Reading ROS SLAM Map")
try:
    map = cv2.imread(directory + mapFile, -1)
except cv2.error as ecv:
    print(ecv)

# print("Map Width (x): ", map.shape[1], " , Map Height (y): ", map.shape[0])

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

# print("Map Metadata:")
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
# pred_points = grid #np.linspace(0,24,100)[:, None]
pred_points = X


# print("Generated Grid Of Points: " + str(grid.shape[0]))
# Predictive GP for log intensity mean and variance
# print("Sampling At Grid Points")
f_mean, f_var = m._raw_predict(pred_points)
# f_mean, f_var = m.predict(pred_points)
f_mean = np.exp(f_mean)
print(max(f_mean))
print(min(f_mean))


# Plot GPR output on map
# Adjust plot coordinates based on size and resolution of the SLAM map
x_values = np.arange(0, map.shape[1])*metadata["resolution"] + metadata["origin"][0]
y_values = np.arange(0, map.shape[0])*metadata["resolution"] + metadata["origin"][1]
implot = plt.imshow(map, cmap='gray', extent = [np.min(x_values), np.max(x_values), np.min(y_values), np.max(y_values)], aspect='equal')
plt.scatter(x = pred_points[:,0], y = pred_points[:,1], c = f_mean, marker="s", s = 1.0, vmin=0.0, vmax=1.2*max(f_mean), cmap='jet')
plt.xlim([-4,4])
plt.ylim([-8,0])
plt.colorbar()
plt.show()

# 지도에 방사선 측정 데이터 출력하는 plot 생성
xvs = X_new[:,0]
yvs = X_new[:,1]
value = np.array([Y_new[:,0]])
implot = plt.imshow(map, cmap='gray', extent = [np.min(x_values), np.max(x_values), np.min(y_values), np.max(y_values)], aspect='equal')
plt.scatter(x=xvs, y=yvs, cmap='jet', s=5, c=value)
plt.xlim([-4,4])
plt.ylim([-8,0])
plt.colorbar()
plt.show()

f_upper, f_lower = f_mean + 2*np.sqrt(f_var), f_mean - 2.*np.sqrt(f_var)

#gpy_output = np.c_[grid, np.exp(f_mean), np.exp(f_var), np.exp(f_upper), np.exp(f_lower)]
gpy_output = np.c_[X, f_mean]


# Locally save file of data for testing
f = open(directory + output_file, "wt") # Should clear existing file of same name
f.close()
# Could be replaced with Pandas approach
np.savetxt(directory + output_file, gpy_output, delimiter=",", header='"x","y","mean"', comments='')
