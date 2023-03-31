import numpy as py


def euler_method(function, x_0, xfinal, y_0, iterations):
    #determine step size
    h = (xfinal - x_0)/iterations
    #initialize x and y
    y = y_0
    x = x_0

    for n in range(iterations):
        slope = eval(function)
        y_next = y + h*slope
        x += h
        y = y_next

    return y

#Euler's Method
    #function: t - y^2, range: 0<t<2, iterations: 10, point: f(0)=1
function = "x - y**2"
x_i = 0
x_f = 2
y_i = 1
n = 10

answer = euler_method(function, x_i, x_f, y_i, n)
print(answer)
