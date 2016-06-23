import math
import random
import matplotlib.pyplot as plt

def unit(x, a, b, c, d):
    return max(a * x + b, \
               c * x + d)

def fun(x, Bs):
    out = 0.0
    for i in range(len(Bs)/4):
        a = Bs[i*4+0]
        b = Bs[i*4+1]
        c = Bs[i*4+2]
        d = Bs[i*4+3]

        out += unit(x, a, b, c, d)
    return out

def obj(Xs, Ts, Bs):
    out = 0.0
    for i, _ in enumerate(Xs):
        x = Xs[i]; x = fun(x, Bs)
        t = Ts[i]
        out += (x - t)**2
    return out / len(Xs)

def clip(x, min=0.0, max=1.0):
    if   x < min: return min
    elif x > max: return max
    else:         return x

#          f(a+h) - f(a)
# lim h->0 -------------
#                h
def gradientDescentStep(Xs, Ts, Bs, rate):
    Gs = [0.0 for i in range(len(Bs))]

    # Compute gradient
    h = 0.000001
    for i, _ in enumerate(Bs):
        prev  = Bs[i]

        Bs[i] = prev - h; subObj = obj(Xs, Ts, Bs)
        Bs[i] = prev + h; addObj = obj(Xs, Ts, Bs)

        Bs[i] = prev

        Gs[i] = (addObj - subObj) / (2.0 * h)

    # Take step along the gradient
    for i, _ in enumerate(Bs):
        Bs[i] -= rate * Gs[i]

    # Return L_1 norm of gradient
    l1 = 0.0
    for i, _ in enumerate(Gs):
        l1 += Gs[i]
    return l1

def gradientDescent(Xs, Ts, Bs=None, rate=1.1):
    numUnits = 10
    if not Bs:
        Bs = [random.uniform(-1.0, 1.0) for i in range(numUnits * 4)]

    for i in range(200):
        print rate, gradientDescentStep(Xs, Ts, Bs, rate)
        print rate, obj(Xs, Ts, Bs)
        print

    return Bs

def makePoints(fun=math.sin, numPoints=100):
    Xs = [i / (numPoints-1.0) - 0.5 for i in range(numPoints)]
    Ts = mapPoints(fun, Xs)

    return Xs, Ts

def mapPoints(fun, Xs):
    return [fun(x) for x in Xs]

Xs, Ts = makePoints(lambda x: math.sin(x * 3.14 * 3), 50)

Bs = None
Bs = gradientDescent(Xs, Ts, Bs,    0.01000)

Ls = mapPoints((lambda x: fun(x, Bs)), Xs)

plt.plot(Xs, Ts, 'ro')
plt.plot(Xs, Ls)
plt.show()
