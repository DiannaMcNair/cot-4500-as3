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

def runge_kutta(function, xinitial, xfinal, yinitial, iterations):
    h = (xfinal - xinitial)/iterations

    y_i = yinitial
    x_i = xinitial

    for i in range(iterations):
        y = y_i
        x = x_i
        k1 = eval(function)

        x = x_i +(h/2)
        y = y_i +((h/2)*k1)
        k2 = eval(function)

        y = y_i +((h/2)*k2)
        k3 = eval(function)

        y = y_i + (h*k3)
        x = x_i + h
        k4 = eval(function)

        y_next = y_i + ((h/6)*(k1 + (2*k2) + (2*k3) + k4))
        y_i = y_next
        x_i += h

    return y_i


#Euler's Method
    #function: t - y^2, range: 0<t<2, iterations: 10, point: f(0)=1
function = "x - y**2"
x_i = 0
x_f = 2
y_i = 1
n = 10

answer = euler_method(function, x_i, x_f, y_i, n)
print(answer)


#Runge-Kutta
    #function: t - y^2, range: 0<t<2, iterations: 10, point: f(0)=1

answer = runge_kutta(function, x_i, x_f, y_i, n)
print(answer)