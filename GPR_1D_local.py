import math

import numpy as np
import matplotlib.pyplot as plt
import GPy
import GPy.kern.src.MyKernels
import random
import cv2


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
    #print(distance)

    ## 방법 2
    #distance = math.sqrt((difference + 1) ** 2)

    Y[600 - cnt] = Y[600] / (distance ** 2)
    Y[600 + cnt] = Y[600 - cnt]

    #print("{2:03d}번째 distance : {0:0.6f}, y값 : {1:0.6f}".format(distance, Y[100 - cnt], cnt))
    if (cnt==600):
        break
    cnt += 1


a = 11
b = 60

# measure_x, measure_y 를 랜덤으로 뽑기
# 해당 값들은 GPR에 적용할 데이터
# 포아송 분포를 갖는 노이즈를 만들기
measure_x = np.zeros(22).reshape(-1,1)
measure_y = np.zeros(22).reshape(-1,1)
seed = 15
random.seed(seed)
np.random.seed(seed)
for i in range(a):
    rnd = random.randint(0, 1200)

    # pNoise = np.random.poisson(Y[rnd], 10)
    # num = 0
    # for cnt in range(10):
    #     num += pNoise[cnt]
    # avg = num / 10
    # noise = Y[rnd] - avg

    # noise = Y[rnd] - np.random.poisson(Y[rnd], 1)
    #
    # print("Real Y[rnd] : {}, noise : {}".format(Y[rnd], noise))
    # measure_x[i] = X[rnd]
    # measure_y[i] = Y[rnd] + noise

    cnt = i
    measure_x[cnt] = X[600-cnt*b]
    measure_x[10+cnt] = X[600+cnt*b]
    measure_y[cnt] = Y[600-cnt*b]
    measure_y[10+cnt] = Y[600+cnt*b]

measure_x[21] = X[600]
measure_y[21] = Y[600]

print("Data Ready")

poisson_likelihood = GPy.likelihoods.Poisson()
laplace_inf = GPy.inference.latent_function_inference.Laplace()

linear      = GPy.kern.Linear(input_dim=1, variances=1.0)
matern32    = GPy.kern.Matern32(input_dim=1, variance=1.0, lengthscale=1.0)
matern52    = GPy.kern.Matern52(input_dim=1, variance=1.0, lengthscale=1.0)
rbf         = GPy.kern.RBF(input_dim=1, variance=1.0, lengthscale=1.0)
exponential = GPy.kern.Exponential(input_dim=1, variance=1.0, lengthscale=1.0)
ratquad     = GPy.kern.RatQuad(input_dim=1, variance=1.0, lengthscale=1.0)
expquad     = GPy.kern.ExpQuad(input_dim=1, variance=1.0, lengthscale=1.0)

myInvSquare = GPy.kern.MyInvSquare(input_dim=1, variance=1.0, lengthscale=1.0)
myAddMatExp = GPy.kern.MyAddMat32Exp(input_dim=1, variance=1.0, lengthscale=1.0)
myMulMaxExp = GPy.kern.MyMulMat32Exp(input_dim=1, variance=1.0, lengthscale=1.0)

bias = GPy.kern.Bias(input_dim=1, variance=0.3)
kernel = matern32 + bias
# kernel = myInvSquare + bias
# kernel = myAddMatExp + bias
# kernel = myMulMaxExp + bias


print("Kernel Initialized")
# kernel.plot()
# plt.xlim([-5,5])
# plt.ylim([0, 1.2])
# plt.show()


### global prediction
model_g = GPy.core.GP(X=measure_x, Y=measure_y, likelihood=poisson_likelihood, inference_method=laplace_inf, kernel=kernel)

# model_g = GPy.models.GPRegression(X=measure_x, Y=measure_y, kernel=kernel)
# model_g.inference_method = laplace_inf
# model_g.likelihood = poisson_likelihood

print("model.likelihood : ")
print(model_g.likelihood)

print("model.inference_method : ")
print(model_g.inference_method)

print("Pre-optimization : ")
print(model_g)
model_g.optimize(messages=True, max_iters=1000)
print("Optimized : ")
print(model_g)


# 예측할 값들의 범위
pred_x = X.reshape(-1, 1)
print(pred_x.shape)

f_mean_g, f_var_g = model_g._raw_predict(pred_x)
f_mean_g = np.exp(f_mean_g)
### likelihood를 Poisson으로 설정하면 log(gp)로 계산되기 때문에 exp()을 해주어야 함
# link_dunctions.py에서 class Log()의 transf 참고

print(max(f_mean_g))
print(min(f_mean_g))

#f_upper, f_lower = f_mean + 2*np.sqrt(f_var), f_mean - 2.*np.sqrt(f_var)



### local prediction
kernel_l = GPy.kern.Matern32(input_dim=1, variance=1.0, lengthscale=1.0, ARD=False)
model_l = GPy.core.GP(X=measure_x, Y=measure_y, likelihood=poisson_likelihood, inference_method=laplace_inf, kernel=kernel_l)

#print(model_l)

f_mean_l, f_var_l = model_l._raw_predict(pred_x)
f_mean_l = np.exp(f_mean_l)

print(max(f_mean_l))
print(min(f_mean_l))

f_mean = np.zeros(1201).reshape(-1, 1)
f_var = np.zeros(1201).reshape(-1, 1)
for i in range(1201):
    #f_mean[i] = (f_mean_l[i] + f_mean_g[i])/2
    f_mean[i] = np.sqrt(f_mean_l[i] * f_mean_g[i])
    f_var[i] = np.sqrt(f_var_l[i] * f_var_g[i])

print(f_mean.shape)
print(max(f_mean))
print(min(f_mean))


model_g.plot_f()
#plt.show()

#model_l.plot_f()
#plt.show()

#plt.scatter(x=pred_x, y=f_mean, c=f_mean, marker="s", s=1.0, vmin=0.0, vmax=1.2*max(f_mean), cmap="Reds")
plt.plot(pred_x, f_mean, c='r')
plt.plot(X, Y, c='g')


model_g.plot()
# #model.plot_confidence()
# #plt.scatter(x=X, y=Y, marker="s", s=1.0, vmin=0.0, vmax=1.2*max(Y), c='g')
# plt.plot(X, Y, c='g')
plt.scatter(x=measure_x, y=measure_y, s=20.0, c='r')
plt.xlim([-7,7])
plt.ylim([-5,110])
#plt.tight_layout()
plt.show()

### Confidence

### Least Squares 계산
### 실제 값(Y[i])과 예측한 값(f_mean[i]) 차이의 제곱을 모두 더하고 나눠줌

error = 0
for i in range(len(pred_x)):
    error += (Y[i]-f_mean[i])**2
error /= len(pred_x)
print(error)