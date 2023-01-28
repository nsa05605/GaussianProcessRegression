import math

import numpy as np
import matplotlib.pyplot as plt
import GPy
import random
import cv2
import sklearn.gaussian_process.kernels

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared

### 1차원 방사선 데이터 만들기
# 기본적인 생각은 하나의 x 지점에 대해 방사선 원점을 설정하고, 그 강도를 지정
# x는 0~10 사이의 좌표(간격을 0.1로 설정?)
# y는 x=5인 지점을 방사선 근원에서 1m 떨어진 점으로 설정하고,
# 거리(x=5인 점과 현재 x' 사이의 거리 + 1m)에 따라 줄어드는 역제곱 법칙을 적용해서 계산


### GPR을 적용하기 위해 측정 데이터를 선정
# x 값을 임의로 10개? 정도 선정
# 갖고 있는 x, y 쌍을 GPR에 적용
# 이때 커널 함수는 Matern32, RBF 사용해서 비교

# 거리는 x[100]-x[i]
X = np.arange(-5, 5.01, 0.01)
Y = np.zeros(1001)
Y[500] = 50

cnt=1
while(True):
    difference = X[500] - X[500 - cnt]
    distance = math.sqrt((difference**2 + 1))
    Y[500 - cnt] = Y[500] / (distance ** 2)
    Y[500 + cnt] = Y[500 - cnt]
    #print("{2:03d}번째 distance : {0:0.6f}, y값 : {1:0.6f}".format(distance, Y[100 - cnt], cnt))
    if (cnt==500):
        break
    cnt += 1

# measure_x, measure_y 를 랜덤으로 뽑기
# 해당 값들은 GPR에 적용할 데이터
measure_x = np.zeros(11).reshape(-1,1)
measure_y = np.zeros(11).reshape(-1,1)
random.seed(0)
for i in range(11):
    # rnd = random.randint(0, 1000)
    # noise = random.randrange(-3,3)
    # print(rnd)
    # measure_x[i] = X[rnd]
    # measure_y[i] = Y[rnd] + noise
    measure_x[i] = X[i*100]
    measure_y[i] = Y[i*100]

print("Data Ready")

poisson_likelihood = GPy.likelihoods.Poisson()
laplace_inf = GPy.inference.latent_function_inference.Laplace()

k1 = GPy.kern.Matern32(input_dim=1, variance=1.0, lengthscale=0.1, ARD=False)
#k1 = GPy.kern.Matern52(input_dim=1, variance=1.0, lengthscale=1.0, ARD=False)
#k1 = GPy.kern.RBF(input_dim=1, variance=1.0, lengthscale=1.0, ARD=False)
#k1 = GPy.kern.Exponential(input_dim=1, variance=1.0, lengthscale=1.0, ARD=False)
#k1 = GPy.kern.PeriodicExponential()
k2 = GPy.kern.Bias(input_dim=1, variance=0.3)
kernel = k1 + k2
print("Kernel Initialized")
# k1.plot()
# plt.xlim([-10,10])
# plt.ylim([0, 1])
# plt.show()

#model = GPy.core.GP(X=measure_x, Y=measure_y, likelihood=poisson_likelihood, inference_method=laplace_inf, kernel=kernel)
model = GPy.models.GPRegression(measure_x, measure_y, kernel)
print("Pre-optimization : ")
print(model)
model.optimize(messages=True)
print("Optimized : ")
print(model)

# 예측할 값들의 범위
pred_x = np.arange(-5, 5.01, 0.01).reshape(-1,1)

f_mean, f_var = model._raw_predict(pred_x)
#f_mean = np.exp(f_mean)

#plt.scatter(x=pred_x, y=f_mean, c=f_mean, marker="s", s=1.0, vmin=0.0, vmax=1.2*max(f_mean), cmap="Reds")
model.plot()
plt.scatter(x=X, y=Y, marker="s", s=1.0, vmin=0.0, vmax=1.2*max(Y), c='g')
plt.scatter(x=measure_x, y=measure_y, s=20.0, c='r')
plt.xlim([-5,5])
plt.ylim([-5,53])
#plt.tight_layout()
plt.show()