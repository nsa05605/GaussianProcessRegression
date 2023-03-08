import math

import numpy as np
import matplotlib.pyplot as plt
import GPy
import GPy.kern.src.MyKernels
import random
import cv2
import pylab as pb

# X = np.array([-2.0, -1.9, -1.6, -1.2, -0.6, -0.2, 0, 0.3, 0.5, 1.6, 1.9]).reshape(-1,1)
# Y = np.array([1, 3, 4, 7, 3, 2, 0, 5, 9, 7, 5]).reshape(-1,1)

sample_size = 10
np.random.seed(0)
X = np.linspace(0.05, 0.95, sample_size).reshape(-1, 1)
Y = (-np.cos(np.pi * X) + np.sin(4 * np.pi * X) + np.random.normal(loc=0.0, scale=0.1, size=(sample_size, 1)))

k1 = GPy.kern.RBF(input_dim=1, lengthscale=0.2)
k2 = GPy.kern.Bias(input_dim=1)
kernel = k1 + k2

gaussian_likelihood = GPy.likelihoods.Gaussian()
poisson_likelihood  = GPy.likelihoods.Poisson()
laplace_inference = GPy.inference.latent_function_inference.Laplace()

# model = GPy.core.GP(X=X, Y=Y, kernel=kernel, likelihood=gaussian_likelihood)
# model = GPy.core.GP(X=X, Y=Y, kernel=kernel, likelihood=poisson_likelihood, inference_method=laplace_inference)


model = GPy.models.GPRegression(X=X, Y=Y, kernel=kernel)
print(model)
model.optimize(messages=True, max_iters=1000)
print(model)

model.plot(figsize=(5,3))
plt.show()