### Regression and Least Squares

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

plt.style.use('seaborn-poster')

# Let's generate x and y, and add some noise into y
x = np.linspace(0, 10, 101)
y = 0.1*np.exp(0.3*x) + 0.1*np.random.random(len(x))

# Let's have a look of the data
plt.figure(figsize=(10,8))
plt.plot(x,y,'b.')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Let's fit the data after we applied the log trick.
A = np.vstack([x, np.ones(len(x))]).T
beta, log_alpha = np.linalg.lstsq(A, np.log(y), rcond=None)[0]
alpha = np.exp(log_alpha)
print(f'alpha={alpha}, beta={beta}')

# Let's have a look of the data
plt.figure(figsize=(10,8))
plt.plot(x, y, 'b.')
plt.plot(x, alpha*np.exp(beta*x), 'r')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# # Log trick for power functions
# x_d = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
# y_d = np.array([0, 0.8, 0.9, 0.1, -0.6, -0.8, -1, -0.9, -0.4])
#
# plt.figure(figsize=(12,8))
# for i in range(1, 7):
#
#     # get the polynomial coefficients
#     y_est = np.polyfit(x_d, y_d, i)
#     plt.subplot(2, 3, i)
#     plt.plot(x_d, y_d, 'o')
#     plt.plot(x_d, np.polyval(y_est, x_d))
#     plt.title(f'Polynomial order {i}')
#
# plt.tight_layout()
# plt.show()

# Let's define the function form
def func(x, a, b):
    y = a*np.exp(b*x)
    return y

alpha, beta = optimize.curve_fit(func, xdata=x, ydata=y)[0]
print(f'alpha={alpha}, beta={beta}')

