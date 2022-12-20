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

data_raw = np.array(pandas.read_csv(directory + input_file, delimiter=',', header=0, usecols=['x','y','z','counts']))

X = np.array([[0, 0, 0], [1, 0, 1], [2, 1, 2], [3, 2, 1]])
Y = np.array([[0], [1], [4], [2]])
print(X)
print(Y)

# Gaussian Process Regression
# A poisson distribution is used
poisson_likelihood = GPy.likelihoods.Poisson()
laplace_inf = GPy.inference.latent_function_inference.Laplace()

k1 = GPy.kern.RBF(input_dim=3, variance=0.5, lengthscale=0.8, ARD=False)
k2 = GPy.kern.Bias(input_dim=3, variance=0.6)   # noise
kernel = k1 + k2
print("Kernel Initialized")

m = GPy.core.GP(X=X, Y=Y, likelihood=poisson_likelihood, inference_method=laplace_inf, kernel=kernel)
print("Pre-optimization : ")
print(m)
m.optimize(messages=True, max_f_eval=1000)
print("Optimized :")
print(m)

pred_point = np.array([[0,0,0], [0,0,1], [0,0,2], [0,0,3],
                      [0,1,0], [0,1,1], [0,1,2], [0,1,3],
                      [0,2,0], [0,2,1], [0,2,2], [0,2,3],
                      [0,3,0], [0,3,1], [0,3,2], [0,3,3],
                      [1,0,0], [1,0,1], [1,0,2], [1,0,3],
                      [1,1,0], [1,1,1], [1,1,2], [1,1,3],
                      [1,2,0], [1,2,1], [1,2,2], [1,2,3],
                      [1,3,0], [1,3,1], [1,3,2], [1,3,3],
                      [2, 0, 0], [2, 0, 1], [2, 0, 2], [2, 0, 3],
                      [2, 1, 0], [2, 1, 1], [2, 1, 2], [2, 1, 3],
                      [2, 2, 0], [2, 2, 1], [2, 2, 2], [2, 2, 3],
                      [2, 3, 0], [2, 3, 1], [2, 3, 2], [2, 3, 3],
                      [3, 0, 0], [3, 0, 1], [3, 0, 2], [3, 0, 3],
                      [3, 1, 0], [3, 1, 1], [3, 1, 2], [3, 1, 3],
                      [3, 2, 0], [3, 2, 1], [3, 2, 2], [3, 2, 3],
                      [3, 3, 0], [3, 3, 1], [3, 3, 2], [3, 3, 3],
                      ])

f_mean, f_var = m._raw_predict(pred_point)
f_mean = np.exp(f_mean)

