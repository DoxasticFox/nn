import math
import numpy as np
import itertools
import scipy.misc
import scipy.ndimage.filters
import scipy.special
import random
import copy
import pickle

saveDir = '/tmp/tmp-400/'
fileNum = 0

def checkpoint(net):
    plot(net)
    save(net)

def plot(net):
    global saveDir
    global fileNum

    nG = len(net[ 0][0]) # Plot width
    nG = len(net[-1][0]) # Plot width
    #nG = 14             # Plot width
    nI = len(net[0])    # Input size / 2
    nL = len(net)       # num layers

    pI = nI * nG + nI
    pJ = nL * nG + nL
    im = [[0.0 for i in range(pI)] for j in range(pJ)]
    for i, layer in enumerate(net):
        for j, unit in enumerate(layer):
            for k in range(nG):
                for l in range(nG):
                    K = k / float(nG - 1)
                    L = l / float(nG - 1)
                    im[k + i*nG + i][l + j*nG + j] = fun(K, L, unit)

    # Save to files
    im = scipy.misc.toimage(im, cmin=0.0, cmax=1.0)
    im.save('%simg-%06d.png' % (saveDir, fileNum))

def save(net):
    global saveDir
    global fileNum

    with open('%snet-%06d.net' % (saveDir, fileNum), 'wb') as netFile:
        netString = pickle.dumps(net)
        netFile.write(netString)

def load(index):
    global saveDir
    global fileNum

    global g
    global sequenceLength

    fileNum = index
    with open('%snet-%06d.net' % (saveDir, fileNum), 'rb') as netFile:
        netString = netFile.read()
        net       = pickle.loads(netString)

        g              = len(net[0][0])
        sequenceLength = len(net[0]   ) * 2
        fileNum       += 1

        return net

def bin(g, x):
    bx = g * x
    bx = max(0,   bx)
    bx = min(g-1, bx)
    bx = int(bx)
    return bx

def bins(g, x, y):
    return bin(g, x), bin(g, y)

def fun(x, y, Bs):
    g = len(Bs)
    bx, by = bins(g, x, y)
    return Bs[bx][by]

def multiFun(Xs, net):
    activated     = []
    addActivation = activated.append

    tmp       = Xs[:]
    for     i in range(len(net   )):
        for j in range(len(net[i])):
            g = len(net[i][j])
            a = (i, j) + bins(g, tmp[j*2+0], tmp[j*2+1])
            addActivation(a)

            tmp[j] = fun(tmp   [j*2+0],
                         tmp   [j*2+1],
                         net[i][j    ])

    return tmp[0], activated

def makeFun(g, c=None):
    lo = +0.0
    hi = +1.0
    Bs = [[None for i in range(g)] for j in range(g)]
    for i in range(len(Bs)):
        for j in range(len(Bs)):
            if   c == None:
                Bs[i][j] = (i+j)/(2.0*(g-1))
            elif c == 'n':
                Bs[i][j] = np.random.uniform()
            else:
                Bs[i][j] = c
    return Bs

def makeNet(dim, g):
    assert np.log2(dim) % 1 - 0.001 < 0.0

    net = []
    while dim >= 2:
        dim /= 2
        if dim >= 2:
            layer = [makeFun(g     ) for i in range(dim)]
        else:
            layer = [makeFun(g, 0.5) for i in range(dim)]
        net.append(layer)
        print g

    return net

def blurHorz(net, i, j, k, l):
    unit  = net[i][j]
    #kMax = len(unit   ) - 1
    lMax  = len(unit[0]) - 1
    if 0 <  l  < lMax:
        return (unit[k  ][l-1] + unit[k  ][l  ] + unit[k  ][l+1])/3.0
    if 0 == l:
        return (unit[k  ][l  ] + unit[k  ][l+1]                 )/2.0
    if      l == lMax:
        return (unit[k  ][l  ] + unit[k  ][l-1]                 )/2.0

def blurVert(net, i, j, k, l):
    unit  = net[i][j]
    kMax  = len(unit   ) - 1
    #lMax = len(unit[0]) - 1
    if 0 <  k  < kMax:
        return (unit[k-1][l  ] + unit[k  ][l  ] + unit[k+1][l  ])/3.0
    if 0 == k:
        return (unit[k  ][l  ] + unit[k+1][l  ]                 )/2.0
    if      k == kMax:
        return (unit[k  ][l  ] + unit[k-1][l  ]                 )/2.0

def applyBlur(net):
    horz = [[[[blurHorz(net,  i, j, k, l) for l in range(len(net [i][j][k]))]
                                          for k in range(len(net [i][j]   ))]
                                          for j in range(len(net [i]      ))]
                                          for i in range(len(net          ))]

    vert = [[[[blurVert(horz, i, j, k, l) for l in range(len(horz[i][j][k]))]
                                          for k in range(len(horz[i][j]   ))]
                                          for j in range(len(horz[i]      ))]
                                          for i in range(len(horz         ))]
    return vert

def selectiveBlur(net, exclude, repeat=3):
    if repeat == 0:
        return net

    blurred = applyBlur(net)
    for i, j, k, l in exclude:
        blurred[i][j][k][l] = \
            net[i][j][k][l]

    return selectiveBlur(blurred, exclude, repeat-1)

def normaliseNet(net):
    for     i in range(len(net   ) - 1):
        for j in range(len(net[i])):
            normaliseUnit(net[i][j])

def normaliseUnit(unit):
    rowMins = [min(row) for row in unit]; unitMin = min(rowMins)
    rowMaxs = [max(row) for row in unit]; unitMax = max(rowMaxs)
    div     = unitMax - unitMin

    if div == 0.0: return

    for     i in range(len(unit   )):
        for j in range(len(unit[i])):
            unit[i][j] = (unit[i][j] - unitMin) / div

def obj(net, Xs, Ts):
    assert len(Xs) == len(Ts)

    out = 0.0
    for i, x in enumerate(Xs):
        x = Xs[i]
        ybar = multiFun(x, net)[0]
        out += (ybar - Ts[i])**2
    return out

def localObj(net, Xs, T, i, j, pi, pj):
    return (multiFun(Xs, net)[0] - T)**2

def multiGrad(Xs, net, T):
    _, act = multiFun(Xs, net)
    #act = act[-1:]

    grads   = []
    addGrad = grads.append

    for a in act:
        i, j, pi, pj = a

        layerIdx     = min(i+1, len(net)-1)
        g            = len(net[layerIdx][0])

        prev = net[i][j][pi][pj]
        if   prev % (1.0/g) <= 1.0/(2 * g) or bin(g, prev) == g - 1:
            # Left  grad
            net[i][j][pi][pj] = prev - 1.0/g; aObj = localObj(net, Xs, T, i, j, pi, pj)
            net[i][j][pi][pj] = prev        ; bObj = localObj(net, Xs, T, i, j, pi, pj)
        else:
            # Right grad
            net[i][j][pi][pj] = prev        ; aObj = localObj(net, Xs, T, i, j, pi, pj)
            net[i][j][pi][pj] = prev + 1.0/g; bObj = localObj(net, Xs, T, i, j, pi, pj)
        net[i][j][pi][pj] = prev

        grad = (bObj - aObj) * g
        addGrad(grad)

    #grads = gradSgn(grads)
    grads = normaliseGradient(grads)
    return zip(grads, act)

def normaliseGradient(gradientVector):
    maxVal = max(gradientVector)
    minVal = min(gradientVector)
    if maxVal > -minVal:
        absMax = maxVal
    else:
        absMax = -minVal
    if absMax == 0.0: return gradientVector
    else:             return [g/absMax for g in gradientVector]

def gradSgn(gradientVector):
    return [sgn(x) for x in gradientVector]

def numOnes(bits):
    n = 0
    for b in bits:
        if b > 0.5:
            n += 1
    return n

def evenParity(bits):
    #return numOnes(bits)/float(len(bits))
    if numOnes(bits) % 2:
        return 1.0
    return 0.0

# Search #######################################################################

# for each example:
#     for each unit:
#         for each pixel triggered by example:
#             try a different intensity
#             check objective with {example}
#             keep change on improvement of objective

def clip(x):
    min = 0.0
    max = 1.0
    if x < min: return min
    if x > max: return max
    return x

def sgn(x):
    #if math.fabs(x) < 0.01: return 0.0
    if x > 0.0: return + 1.0
    if x < 0.0: return - 1.0
    else:       return   0.0

def search(net, Xs, Ts):
    g            = len(net[0][0])
    rateInv      = float(g)

    zipped       = zip(Xs, Ts)
    randomChoice = random.choice
    sampleSize   = g * g

    global fileNum
    print sampleSize

    while True:
        # Make a batch
        if fileNum <= g * 10:
            batch = [] # Warm it up!
        else:
            batch = [randomChoice(zipped) for i in range(sampleSize)]
        #batch = [randomChoice(zipped) for i in range(sampleSize)]


        # Take a step for each sample in batch
        for X, T in batch:
            GAs = multiGrad(X, net, T)
            for ga in GAs:
                grad, act    = ga
                i, j, pi, pj = act

                net[i][j][pi][pj] -= grad / rateInv
                net[i][j][pi][pj]  = clip(net[i][j][pi][pj])

        # Selectively blur based on activations after batch update
        activations = []
        for X, T in batch:
            _, acts = multiFun(X, net)
            activations += acts
        net = selectiveBlur(net, exclude=activations)
        normaliseNet(net)

        if fileNum % 2 == 0:
            checkpoint(net)
            print 'plot', fileNum
        if fileNum % 20 == 0:
            sample = random.sample(zipped, 500)
            Xs_, Ts_ = zip(*sample)
            print obj(net, Xs_, Ts_)
        fileNum += 1

################################################################################

# Make some data
trainingSize   = 100000
sequenceLength = 8
g              = 50

#m  = loadMnist()
#Xs = m[0]
#Ys = [float(m[1][i] == 0) for i in range(len(m[1]))]

#Xs = [[int(np.random.uniform() + 0.5) for i in range(sequenceLength)] for j in range(trainingSize)]
Xs = [[np.random.uniform() for i in range(sequenceLength)] for j in range(trainingSize)]
Ys = [evenParity(x) for x in Xs]

# Fit
#bl = makeNet(sequenceLength, g)
bl = load(1000)
bl[0][0] = makeFun(g, 0.5)
normaliseNet(bl)
search(bl, Xs, Ys)

# Benchmark
total = len(Xs)
right = 0
for j in range(total):
    x = [np.random.uniform() for i in range(sequenceLength)]
    d = np.abs(multiFun(x, bl)[0] - evenParity(x))
    if d < 0.01:
        right += 1
print 'Right (%): ', (right/float(total)) * 100
