import numpy as np
import matplotlib.pyplot as plt

def fun1(x):
    return x**3 + 5 * x**2 + 10

def fun2(x):
    return x**2 + np.cos(x)

def fun3(x):
    return 3*x**4

def partial_fun1(x):
    return 3*x**2 + 10*x

def partial_fun2(x):
    return 2*x - np.sin(x)

def partial_fun3(x):
    return 12*x**3

def Fun(x):
    return fun3(fun2(fun1(x)))

def partial_Fun(x):
    return partial_fun3(fun2(fun1(x))) * partial_fun2(fun1(x)) * partial_fun1(x)

def finite_difference(x):
    ε = 1e-9
    return (Fun(x + ε) - Fun(x)) / ε

def partial_finite_difference(x):
    ε1 = 1e-9
    # ε2 = 10**(-np.floor(np.log10(fun1(x)))) * 1e-9
    # ε3 = 10**(-np.floor(np.log10(fun2(fun1(x))))) * 1e-9
    ε2 = 1e-9
    ε3 = 1e-9
    return ((fun3(fun2(fun1(x)) + ε3) - fun3(fun2(fun1(x)))) / ε3) * ((fun2(fun1(x) + ε2) - fun2(fun1(x))) / ε2) * ((fun1(x + ε1) - fun1(x)) / ε1)

x = np.arange(0.0, 1.0, 0.001)
partial_Fun(1.0)
finite_difference(1.0)
partial_finite_difference(1.0)

error_partial = abs(partial_finite_difference(x) - partial_Fun(x))
error_finite = abs(finite_difference(x) - partial_Fun(x))
print(np.mean(error_partial))
print(np.mean(error_finite))


