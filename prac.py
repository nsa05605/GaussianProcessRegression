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
X = np.arange(-6, 6.01, 0.01)
Y = np.zeros(1201)
Y[600] = 100

cnt=1
while(True):
    difference = X[600] - X[600 - cnt]
    ## 방법 1
    distance = math.sqrt((difference**2 + 1))
    X[600 - cnt] = X[600] - (distance-1)
    X[600 + cnt] = X[600] + (distance-1)

    ## 방법 2
    #distance = math.sqrt((difference + 1) ** 2)

    Y[600 - cnt] = Y[600] / (distance ** 2)
    Y[600 + cnt] = Y[600 - cnt]

    #print("{2:03d}번째 distance : {0:0.6f}, y값 : {1:0.6f}".format(distance, Y[100 - cnt], cnt))
    if (cnt==600):
        break
    cnt += 1

print("Data Ready")

poisson_likelihood = GPy.likelihoods.Poisson()
laplace_inf = GPy.inference.latent_function_inference.Laplace()

numMeasurement = 51
err = []
err_0, err_1, err_2, err_3, err_4, err_5, err_6, err_7, err_8, err_9, err_10 = [], [], [], [], [], [], [], [], [], [], []
err.append(err_0)
err.append(err_1)
err.append(err_2)
err.append(err_3)
err.append(err_4)
err.append(err_5)
err.append(err_6)
err.append(err_7)
err.append(err_8)
err.append(err_9)
err.append(err_10)

for sIdx in range(10):  # seed index
    measure_x = np.zeros(numMeasurement).reshape(-1, 1)
    measure_y = np.zeros(numMeasurement).reshape(-1, 1)
    random.seed(sIdx*30)
    np.random.seed(sIdx*30)
    for i in range(numMeasurement):
        rnd = random.randint(0, 1000)
        noise = Y[rnd] - np.random.poisson(Y[rnd], 1)
        #print("Real Y[rnd] : {}, noise : {}".format(Y[rnd], noise))
        measure_x[i] = X[rnd]
        measure_y[i] = Y[rnd] + noise

    for kIdx in range(7):  # kernel index

        matern32    = GPy.kern.Matern32(input_dim=1, variance=1.0, lengthscale=1.0, ARD=False)
        matern52    = GPy.kern.Matern52(input_dim=1, variance=1.0, lengthscale=1.0, ARD=False)
        rbf         = GPy.kern.RBF(input_dim=1, variance=1.0, lengthscale=1.0, ARD=False)
        exponential = GPy.kern.Exponential(input_dim=1, variance=1.0, lengthscale=1.0, ARD=False)
        ratquad     = GPy.kern.RatQuad(input_dim=1, variance=1.0, lengthscale=1.0, ARD=False)

        if kIdx == 0:
            kernel = matern32
        elif kIdx == 1:
            kernel = matern52
        elif kIdx == 2:
            kernel = rbf
        elif kIdx == 3:
            kernel = exponential
        elif kIdx == 4:
            kernel = matern32 + exponential
        elif kIdx == 5:
            kernel = matern52 + exponential
        elif kIdx == 6:
            kernel = ratquad
        elif kIdx == 7:
            kernel = matern32 + ratquad
        elif kIdx == 8:
            kernel = exponential + ratquad
        elif kIdx == 9:
            kernel = matern32 + ratquad + exponential
        elif kIdx == 10:
            kernel = matern32 * ratquad + exponential


        #model = GPy.core.GP(X=measure_x, Y=measure_y, likelihood=poisson_likelihood, inference_method=laplace_inf, kernel=kernel)
        model = GPy.models.GPRegression(X=measure_x, Y=measure_y, kernel=kernel)
        model.inference_method = laplace_inf
        model.likelihood = poisson_likelihood

        # print("model.likelihood : ")
        # print(model.likelihood)
        #
        # print("model.inference_method : ")
        # print(model.inference_method)

        # print("Pre-optimization : ")
        # print(model)
        model.optimize(messages=False)
        #print("Optimized : ")
        #print(model)


        # 예측할 값들의 범위
        pred_x = X.reshape(-1, 1)

        f_mean, f_var = model._raw_predict(pred_x)
        f_mean = np.exp(f_mean)

        #print(max(f_mean))
        #print(min(f_mean))

        # #plt.scatter(x=pred_x, y=f_mean, c=f_mean, marker="s", s=1.0, vmin=0.0, vmax=1.2*max(f_mean), cmap="Reds")
        # model.plot()
        # #plt.scatter(x=X, y=Y, marker="s", s=1.0, vmin=0.0, vmax=1.2*max(Y), c='g')
        # plt.plot(X, Y, c='g')
        # plt.scatter(x=measure_x, y=measure_y, s=20.0, c='r')
        # plt.xlim([-7,7])
        # plt.ylim([-5,60])
        # #plt.tight_layout()
        # plt.show()

        ### Least Squares 계산
        ### 실제 값(Y[i])과 예측한 값(f_mean[i]) 차이의 제곱을 모두 더하고 나눠줌

        error = 0
        for i in range(len(pred_x)):
            error += (Y[i]-f_mean[i])**2
        error /= len(pred_x)
        #print(error)

        err[kIdx].append(error)

    print("seed {} finished".format(sIdx))

kernels = ['matern32', 'matern52', 'rbf', 'exponential', 'matern32 + exponential', 'matern52 + exponential',
           'ratquad', 'matern32 + ratquad', 'exponential + ratquad', 'matern32 + ratquad + exponential', 'matern32 * ratquad + exponential']
for i in range(len(err)):
    sum = 0
    for j in range(len(err[i])):
        #print("{}번째 커널의 {}번째 error : {}".format(i, j, err[i][j]))
        sum += err[i][j]
    avg = sum / 100
    print("{} 커널의 평균 error : {}".format(kernels[i], avg))