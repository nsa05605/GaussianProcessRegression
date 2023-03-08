import numpy as np
import pandas
import matplotlib.pyplot as plt
import GPy
import yaml
import cv2
from mpl_toolkits.mplot3d import Axes3D
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


directory = "C:/Users/jihun/PycharmPRojects/GaussianProcessRegression/"
#directory = "/home/rail/PycharmProjects/GaussianProcessRegression/"
input_file = "docs/radiation_data.txt"
output_file = "Drone_3Drad_GPR_RBF_res40.txt"

data_raw = np.array(pandas.read_csv(directory + input_file, delimiter=',', header=0, usecols=['x','y','z','counts']))

X = np.array(data_raw[:,0:3])       # x, y, z
Y = np.array([data_raw[:, 3]]).T    # counts
print("Data Ready")

# Gaussian Process Regression
# A poisson distribution is used
poisson_likelihood = GPy.likelihoods.Poisson()
laplace_inf = GPy.inference.latent_function_inference.Laplace()

# k1 = GPy.kern.Matern32(input_dim=3, variance=0.8, lengthscale=0.5, ARD=False)
k1 = GPy.kern.MyMatern32(input_dim=3, variance=0.8, lengthscale=0.5, ARD=False)
k2 = GPy.kern.Bias(input_dim=3, variance=0.3)   # noise
kernel = k1 + k2
print("Kernel Initialized")

m = GPy.core.GP(X=X, Y=Y, likelihood=poisson_likelihood, inference_method=laplace_inf, kernel=kernel)
print("Pre-optimization : ")
print(m)
m.optimize(messages=True, max_f_eval=1000)
print("Optimized :")
print(m)

resolution = 0.40

### 지도가 아직 없기 때문에 시작 위치 및 예측 범위를 직접 만들어 줘야 함
origin = [1.0, -2.0]   # 시작 위치

### 161 x 81 x 41 크기로 만들면 될듯?
x = np.arange(0, 161) * resolution + origin[0]
y = np.arange(0, 81) * resolution + origin[1]
xv, yv = np.meshgrid(x, y)
m_grid = np.dstack((xv.ravel(), yv.ravel()))[0]

print("m_grid.shape : ")
print(m_grid.shape)

pred_point = m_grid
print("Sampling at Grid Points")

psize = pred_point.shape[0]

z = np.zeros((psize, 1))
p = np.hstack([pred_point, z])

for i in range(1, 61):
    z = np.ones((psize, 1))
    z *= 0.40*i
    p1 = np.hstack([pred_point, z])
    p = np.vstack((p, p1))
print(p.shape)

### 3차원으로 확장하면 여기에서 차원 관련 오류
### 해결
f_mean, f_var = m._raw_predict(p)
f_mean = np.exp(f_mean)

# print(f_mean)
# print(max(f_mean))
# print(min(f_mean))

print("f_mean's shape : ")
print(f_mean.shape)

# print(max(f_mean))
# print(np.where(f_mean == max(f_mean)))
# print(f_mean[:,0][0])
# print(f_mean[:,0][220948])

### f_mean이 특정 임계값(ex_ 2?)보다 낮으면, 해당 지점을 pred_point에서 제거하는 코드

p = np.delete(p, np.where(f_mean < 15), axis=0)
f_mean = np.delete(f_mean, np.where(f_mean < 15), axis=0)
# print(p.shape)
# print(f_mean.shape)

# print(p)
# print(f_mean)

### (0,0,0) 좌표에 0값을 하나 추가
p = np.append(p, [[0.0, 0.0, 0.0]], axis=0)
f_mean = np.append(f_mean, [[0.0]], axis=0)

print("revised f_mean's shape : ")
print(f_mean.shape)

print("max and min values of f_mean : ")
print(max(f_mean))
print(min(f_mean))

###

xs = p[:, 0]    # x
ys = p[:, 1]    # y
zs = p[:, 2]    # z
color = f_mean

x = np.array(data_raw[:, 0])
y = np.array(data_raw[:, 1])
z = np.array(data_raw[:, 2])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs, ys, zs, c=color, marker='s', s=3, cmap='jet', alpha=0.3)
ax.scatter(x, y, z, c=Y, s=5, cmap='jet')
ax.set_xlim([0,8])
ax.set_ylim([-4,4])
ax.set_zlim([0,8])
plt.show()


### 측정 데이터를 경로에 따라 점으로 나타내기

fig2 = plt.figure(figsize=(6,6))
ax2 = fig2.add_subplot(111, projection='3d')
x = np.array(data_raw[:, 0])
y = np.array(data_raw[:, 1])
z = np.array(data_raw[:, 2])
ax2.set_xlim([0,8])
ax2.set_ylim([-4,4])
ax2.set_zlim([0,8])
ax2.scatter(x,y,z,c=Y, cmap='jet')
plt.show()

gpy_output = np.c_[p, f_mean]
f = open(directory + output_file, "wt")
f.close()

np.savetxt(directory + output_file, gpy_output, delimiter=',', header='"x","y","z","mean"', comments='')