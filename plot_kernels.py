import math
import matplotlib.pyplot as plt
import numpy as np


### Ground Truth data
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

kern_InvSquare   = np.zeros(1201)
kern_RBF         = np.zeros(1201)
kern_Matern32    = np.zeros(1201)
kern_Exponential = np.zeros(1201)
kern_AddMat32Exp = np.zeros(1201)
kern_MulMat32Exp = np.zeros(1201)
kern_MulRBFExp   = np.zeros(1201)

for i in range(1201):
    kern_InvSquare[i]   = 100/((abs(X[i])+1)**2)
    kern_RBF[i]         = 100*np.exp(-abs(X[i])**2 / 2)
    kern_Matern32[i]    = 100*(1+np.sqrt(3)*abs(X[i])) * np.exp(-np.sqrt(3)*abs(X[i]))
    kern_Exponential[i] = 100*np.exp(-abs(X[i]))
    kern_AddMat32Exp[i] = 100*((1+np.sqrt(3)*abs(X[i])) * np.exp(-np.sqrt(3)*abs(X[i])) + np.exp(-abs(X[i]))) / 2
    kern_MulMat32Exp[i] = 100*(1+np.sqrt(3)*abs(X[i])) * np.exp(-np.sqrt(3)*abs(X[i])) * np.exp(-abs(X[i]))
    kern_MulRBFExp[i]   = 100*(np.exp(-abs(X[i])**2 / 2) * np.exp(-abs(X[i])))

plt.plot(X, Y, c='g')
#plt.plot(X, kern_InvSquare, c='c')
plt.plot(X, kern_RBF, c='y')
plt.plot(X, kern_Matern32, c='r')
plt.plot(X, kern_Exponential, c='b')
plt.plot(X, kern_AddMat32Exp, c='m')
plt.plot(X, kern_MulMat32Exp, c='k')
plt.plot(X, kern_MulRBFExp, c='c')

plt.show()

err_InvSq, err_RBF, err_Mat32,err_Exp,err_AddMat32Exp,err_MulMat32Exp,err_MulRBFExp = 0,0,0,0,0,0,0

for i in range(len(Y)):
    err_InvSq += (Y[i] - kern_InvSquare[i]) ** 2
    err_RBF += (Y[i]-kern_RBF[i])**2
    err_Mat32 += (Y[i]-kern_Matern32[i])**2
    err_Exp += (Y[i]-kern_Exponential[i])**2
    err_AddMat32Exp += (Y[i]-kern_AddMat32Exp[i])**2
    err_MulMat32Exp += (Y[i]-kern_MulMat32Exp[i])**2
    err_MulRBFExp += (Y[i]-kern_MulRBFExp[i])**2
err_InvSq /= len(Y)
err_RBF /= len(Y)
err_Mat32 /= len(Y)
err_Exp /= len(Y)
err_AddMat32Exp /= len(Y)
err_MulMat32Exp /= len(Y)
err_MulRBFExp /= len(Y)

#print(err_InvSq)
print(err_RBF)
print(err_Mat32)
print(err_Exp)
print(err_AddMat32Exp)
print(err_MulMat32Exp)
print(err_MulRBFExp)