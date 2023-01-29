# https://github.com/LeeDoYup/Gaussian-Process-Gpy/blob/master/1_GP_basics_KOR.ipynb 를 참고해서 작성

from __future__ import print_function

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import GPy

# GP는 함수 f(x) 값들의 분포(distribution)을 정의하는 방법이며, 이 분포가 정규 분포(Gaussian distribution)을 따른다고 가정함
# 즉 관찰한 n개의 (x에 따른) y값들 사이의 관계가 정규 분포를 따른다고 모델링함으로써 함수 f를 직접 추정하지 않고도 데이터를 모델링하는 방법
# mean function m(x)와 covariance function k(x,x')은 x1, ..., xn의 데이터에 대해 (f(x1), ..., f(xn))~N(m,K) 라고 표현 가능하다.(이때 k와 K 구분)

kernel = GPy.kern.Exponential(input_dim=1, lengthscale=0.9)

sample_size = 500
points = np.linspace(0, 5, sample_size).reshape(-1, 1)
# 이때 reshape(-1,1)은 2차원 배열로 변경하는 의미로 1개씩 담긴 것을 의미
# 즉 위의 points.shape은 기존에 (500,)에서 (500, 1)로 바뀜

mean_vector = np.zeros(sample_size)
covariance_matrix = kernel.K(points, points)

realization_number = 3
realization = np.random.multivariate_normal(mean_vector, covariance_matrix, realization_number)

for index in range(realization_number):
    plt.plot(points, realization[index, :])
plt.xlabel('Points')
plt.ylabel('Values')
plt.show()


### Building GPR model
# 모델링하려는 함수 f가 아래와 같은 구조를 따른다고 가정해보자
# f(x) = -cos(pi*x) + sin(4*pi*x), x in [0,1]
# with noise, y(x) = f(x) + ε
# 위의 함수를 따르는 샘플 10개를 만들어 보면,
sample_size = 10
points = np.linspace(0.05, 0.95, sample_size).reshape(-1,1)
values = (-np.cos(np.pi * points) + np.sin(4 * np.pi * points) + np.random.normal(loc=0.0, scale=0.1, size=(sample_size, 1)))

plt.figure(figsize=(5,3))
plt.plot(points, values, '.')
plt.xlabel('Points')
plt.ylabel('Values')
plt.tight_layout()
plt.show()

## Define covariance function
# GP의 핵심은 결국 데이터 사이의 관계를 모델링하는데 사용하는 covariance function(=kernel)을 무엇으로 정의할 것인가이다.
# 대표적으로 사용하는 kernel은 RBF(=squared exponential) kernel이며, 아래와 같은 식으로 표현된다.
# k(x,x') = \sigma^2 * exp(-||x-x'||^2 / (2*(length_scale)^2))
# 아래와 같이 GPy.kern.RBF 함수를 사용하여 kernel을 정의할 수 있다.
# length_scale 변화에 따라 kernel 값이 어떻게 변하는지 확인해보자
input_dim = 1
variance = 1
lengthscale = 0.2
kernel = GPy.kern.RBF(input_dim=input_dim, variance=variance, lengthscale=lengthscale)

## Create GPR model
model = GPy.models.GPRegression(points, values, kernel)
print(model)
model.plot(figsize=(5,3))
plt.show()


### Parameters of the covariance function
# lengthscale이 커질수록 RBF kernel은 평평한 함수가 된다.
# 즉, 데이터 사이의 거리가 충분히 멀더라도 covariance가 높다고 판단한다.

kernel = GPy.kern.RBF(1)
lengthscale_array = np.asarray([0.01, 0.1, 0.2, 0.5, 1.0, 10.0])
figure_handle, axes = plt.subplots(2, 3, figsize=(9.5, 5), sharex=True, sharey=True)
print(figure_handle) # (950x500)
print(axes)
# subplots(rows, cols, figsize)

for lengthscale, selected_axes in zip(lengthscale_array, axes.ravel()): # ravel() : 다차원 배열을 1차원 배열로 펴주는 함수
    kernel.lengthscale = lengthscale
    model = GPy.models.GPRegression(points, values, kernel)
    # print(model)
    model.plot(ax=selected_axes)
    selected_axes.set_xlabel('Points')
    selected_axes.set_ylabel('Values')
plt.show()

### Tuning parameter of covariance function
# 그렇다면 covariance function의 parameter 튜닝은 어떻게 할까? 튜닝은 일반적으로
# 1. likelihood를 최대화 maximizing likelihood
# 2. LOO CV(Leave-one-out Cross-Validation) score
# 두 가지를 확인한다.
# GPy는 하나하나 튜닝이 어려운 사람을 위해 model.optimizer() 함수를 제공한다.

kernel = GPy.kern.RBF(input_dim=1, lengthscale=0.1)
model = GPy.models.GPRegression(X=points, Y=values, kernel=kernel)

model.optimize(messages=True)

print("print model")
print(model)
model.plot(figsize=(5,3))
plt.show()


### Noise variance
# 우리가 GP 모델에 추가하는 Noise variance(sigma_0)는 cost function에서 regularization과 같은 역할을 한다.
# 즉, noise가 커지면 우리가 보유하고 있는 데이터에 대한 불확실도가 올라가므로 좀 더 일반화된 방향으로 모델이 학습된다.
# Noise variance 값이 커지면, 우리의 훈련 데이터에 특화되는 성향이 낮아지므로, 좀 더 부드러운(smooth) 모델이 학습된다.

