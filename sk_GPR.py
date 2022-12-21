import GPy.likelihoods
import numpy as np
import pandas
import matplotlib.pyplot as plt
import yaml
import cv2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

directory = "C:/Users/jihun/PycharmProjects/GaussianProcessRegression/LUNL/"
input_file = "LUNL_radiation_data.txt"
output_file = "LUNL_2Drad_GPR.txt"

# import raw data
# fine format = timestame, x, y, z, counts (per second)
# only extract the necessary columns of x, y, z, and counts
data_raw = np.array(pandas.read_csv(directory + input_file, delimiter=',', header=0, usecols=['x','y','z','counts']))

X = np.array(data_raw[:,0:3])   # x, y, z
Y = np.array([data_raw[:,3]]).T # counts

print("Data Ready")

"""
Gaussian Process Regression
Use a RBF kernel and matern kernel(radiation fluctutation due to sources) with a bias term (due to homogenous background radiation)
input_dim=3 - i.e. x, y, z coordinates
variance - starting guess at the magnitude of the contribution (will be altered by the regression)
lengthscale = 0.5 - how closely correlated data is in space, start at 0.5m (will be altered by the regression)
ARD=False - the kernel is NOT independant in all dimensions (because we expect the function to work radially)
"""

# As the data is counts, the error (variance) associated is equal to the mean - this is not the case for gaussians at small values
# A poisson approximation is used to tackle low count rate issues
poisson_likelihood = GPy.likelihoods.Poisson()
laplace_inf = GPy.inference.latent_function_inference.Laplace()

k1 = 1 * RBF(length_scale=0.5)

kernel = k1

GP = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
GP.fit(X, Y)
GP.kernel_

mean_prediction, std_prediction = GP.predict(X, return_std=True)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.array(data_raw[:, 0])
y = np.array(data_raw[:, 1])
z = np.array(data_raw[:, 2])

ax.scatter(x,y,z, cmap='jet', alpha=0.5)
plt.xlim(-5, 2)
plt.ylim(-7, 0)

#plt.plot(X, Y)
#plt.scatter(X, Y, label="Observations")
#plt.plot(X, mean_prediction, label="Mean Prediction")
plt.show()

print(mean_prediction)

'''
### 3차원 경로 plot으로 나타내기 ###
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.array(data_raw[:, 0])
y = np.array(data_raw[:, 1])
z = np.array(data_raw[:, 2])

#ax.plot(x,y,z)
ax.scatter(x, y, z, color='r', alpha=0.5)
plt.xlim(-5, 2)
plt.ylim(-7, 0)
plt.show()
'''