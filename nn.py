import math
import numpy as np
import itertools
import scipy.misc
import scipy.ndimage.filters
import scipy.special
import random
import copy
import pickle

saveDir = '/tmp/tmp-100/'
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
                    im[k + i*nG + i][l + j*nG + j] = evalUnit(K, L, unit)

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

    global res
    global sequenceLength

    fileNum = index
    with open('%snet-%06d.net' % (saveDir, fileNum), 'rb') as netFile:
        netString = netFile.read()
        net       = pickle.loads(netString)

        res            = len(net[0][0])
        sequenceLength = len(net[0]   ) * 2
        fileNum       += 1

        return net

def bin(res, x):
    bx = res * x
    bx = max(0,   bx)
    bx = min(res-1, bx)
    bx = int(bx)
    return bx

def bins(x, y, res):
    return bin(res, x), bin(res, y)

def evalUnit(x, y, unit):
    res    = len(unit)
    bx, by = bins(x, y, res)
    return unit[bx][by]

def multiFun(Xs, net):
    activated     = []
    addActivation = activated.append

    tmp       = Xs[:]
    for     i in range(len(net   )):
        for j in range(len(net[i])):
            res = len(net[i][j])
            a = (i, j) + bins(tmp[j*2+0], tmp[j*2+1], res)
            addActivation(a)

            tmp[j] = evalUnit(tmp   [j*2+0],
                              tmp   [j*2+1],
                              net[i][j    ])

    return tmp[0], activated

def dxUnit(x, y, unit):
    res    = len(unit)
    bx, by = bins(x, y, res)

    if   bin(res, x) == 0:               return res * (unit[bx+1][by] - unit[bx  ][by])
    elif bin(res, x) == res - 1:         return res * (unit[bx  ][by] - unit[bx-1][by])
    elif x % (1.0/res) <= 1.0/(2 * res): return res * (unit[bx  ][by] - unit[bx-1][by])
    else:                                return res * (unit[bx+1][by] - unit[bx  ][by])

def dyUnit(x, y, unit):
    res    = len(unit)
    bx, by = bins(x, y, res)

    if   bin(res, y) == 0:               return res * (unit[bx][by+1] - unit[bx][by  ])
    elif bin(res, y) == res - 1:         return res * (unit[bx][by  ] - unit[bx][by-1])
    elif y % (1.0/res) <= 1.0/(2 * res): return res * (unit[bx][by  ] - unit[bx][by-1])
    else:                                return res * (unit[bx][by+1] - unit[bx][by  ])

def dwUnit(x, y, unit):
    return 1.0

def dxErr(x, t):
    return x - t

def backprop(net, Xs, t):
    Zs  = forward (net, Xs);              # print 'Zs ', Zs
    DXs = dxUnits (net, Zs);              # print 'DXs', DXs
    DYs = dyUnits (net, Zs);              # print 'DYs', DYs
    DWs = dwUnits (net, Zs);              # print 'DWs', DWs
    Ds  = backward(Zs, DXs, DYs, DWs, t); # print 'Ds ', Ds
    As  = activationLocations(net, Zs)
    return zipDs(Ds, As)

def forward(net, Xs):
    width = len(net[0]) * 2

    depth = math.log(width, 2) + 1
    depth = int(depth)

    Zs  = [[0.0 for j in range(width / 2**i)] for i in range(depth)]

    for i, x in enumerate(Xs):
        Zs[0][i] = x

    for     i in range(1, len(Zs     )   ):
        for j in range(0, len(Zs[i-1]), 2):
            x    = Zs [i-1][j]
            y    = Zs [i-1][j+1]
            unit = net[i-1][j/2]

            Zs[i][j/2] = evalUnit(x, y, unit)

    return Zs

def dxUnits(net, Zs):
    return dUnits(net, Zs, dxUnit)

def dyUnits(net, Zs):
    return dUnits(net, Zs, dyUnit)

def dwUnits(net, Zs):
    width = len(Zs[0]) / 2
    depth = len(Zs   ) - 1

    Ds  = [[0.0 for j in range(width / 2**i)] for i in range(depth)]

    for     i in range(1, len(Zs     )   ):
        for j in range(0, len(Zs[i-1]), 2):
            x    = Zs [i-1][j]
            y    = Zs [i-1][j+1]
            unit = net[i-1][j/2]

            Ds[i-1][j/2] = dwUnit(x, y, unit)

    return Ds

def dUnits(net, Zs, dUnit):
    width = len(Zs[0]) / 2 / 2
    depth = len(Zs   ) - 1 - 1

    Ds  = [[0.0 for j in range(width / 2**i)] for i in range(depth)]

    for     i in range(2, len(Zs     )   ):
        for j in range(0, len(Zs[i-1]), 2):
            x    = Zs [i-1][j]
            y    = Zs [i-1][j+1]
            unit = net[i-1][j/2]

            Ds[i-2][j/2] = dUnit(x, y, unit)

    return Ds

def backward(Zs, DXs, DYs, DWs, t):
    width = len(Zs[0]) / 2
    depth = len(Zs   ) - 1

    Ds  = [[1.0 for j in range(width / 2**i)] for i in range(depth)]

    # Backward pass: Multiply the DXs and DYs appropriately
    for     i in range(len(Ds) - 2, -1,         -1):
        for j in range(0,           len(Ds[i]),  2):
            Ds[i][j  ] = Ds[i+1][j/2] * DXs[i][j/2]
            Ds[i][j+1] = Ds[i+1][j/2] * DYs[i][j/2]

    # Multiply by DWs element-wise
    for     i in range(len(Ds   )):
        for j in range(len(Ds[i])):
            Ds[i][j] *= DWs[i][j]

    # Multiply by dxErr element-wise
    output = Zs[-1][-1]
    dErr = dxErr(output, t)
    for     i in range(len(Ds   )):
        for j in range(len(Ds[i])):
            Ds[i][j] *= dErr

    return Ds

def activationLocations(net, Zs):
    res = len(net[0][0])
    As  = [[(i, j/2) + bins(Zs[i][j], Zs[i][j+1], res) for j in range(0, len(Zs[i]),  2)] \
                                                       for i in range(0, len(Zs   )-1  )]
    return As

def zipDs(Ds, As):
    # Assume Ds and As have the same dimensions
    return [(Ds[i][j], As[i][j]) for i in range(len(Ds   )) \
                                 for j in range(len(Ds[i]))]

def makeUnit(res, c=None):
    lo = +0.0
    hi = +1.0
    unit = [[None for i in range(res)] for j in range(res)]
    for i in range(len(unit)):
        for j in range(len(unit)):
            if   c == None:
                unit[i][j] = (i+j)/(2.0*(res-1))
            elif c == 'n':
                unit[i][j] = np.random.uniform()
            else:
                unit[i][j] = c
    return unit

def makeNet(dim, res):
    assert np.log2(dim) % 1 - 0.001 < 0.0

    net = []
    while dim >= 2:
        dim /= 2
        if dim >= 2:
            layer = [makeUnit(res     ) for i in range(dim)]
        else:
            layer = [makeUnit(res, 0.5) for i in range(dim)]
        net.append(layer)
        print res

    return net

def blurHorz(net, i, j, k, l):
    unit  = net[i][j]
    #kMax = len(unit   ) - 1
    lMax  = len(unit[0]) - 1
    if 0 <  l  < lMax:
        return (unit[k  ][l-1] + unit[k  ][l  ] + unit[k  ][l+1])/3.0
    if 0 == l:
        return (                 unit[k  ][l  ] + unit[k  ][l+1])/2.0
    if      l == lMax:
        return (unit[k  ][l-1] + unit[k  ][l  ]                 )/2.0

def blurVert(net, i, j, k, l):
    unit  = net[i][j]
    kMax  = len(unit   ) - 1
    #lMax = len(unit[0]) - 1
    if 0 <  k  < kMax:
        return (unit[k-1][l  ] + unit[k  ][l  ] + unit[k+1][l  ])/3.0
    if 0 == k:
        return (                 unit[k  ][l  ] + unit[k+1][l  ])/2.0
    if      k == kMax:
        return (unit[k-1][l  ] + unit[k  ][l  ]                 )/2.0

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

def multiGrad(net, Xs, T):
    _, act = multiFun(Xs, net)
    #act = act[-1:]

    grads   = []
    addGrad = grads.append

    for a in act:
        i, j, pi, pj = a

        layerIdx     = min(i+1, len(net)-1)
        res            = len(net[layerIdx][0])

        prev = net[i][j][pi][pj]
        if   prev % (1.0/res) <= 1.0/(2 * res) or bin(res, prev) == res - 1:
            # Left  grad
            net[i][j][pi][pj] = prev - 1.0/res; aObj = localObj(net, Xs, T, i, j, pi, pj)
            net[i][j][pi][pj] = prev        ; bObj = localObj(net, Xs, T, i, j, pi, pj)
        else:
            # Right grad
            net[i][j][pi][pj] = prev        ; aObj = localObj(net, Xs, T, i, j, pi, pj)
            net[i][j][pi][pj] = prev + 1.0/res; bObj = localObj(net, Xs, T, i, j, pi, pj)
        net[i][j][pi][pj] = prev

        grad = (bObj - aObj) * res
        addGrad(grad)

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
    else:             return [res/absMax for res in gradientVector]

def numOnes(bits):
    n = 0
    for b in bits:
        if b > 0.5:
            n += 1
    return n

def evenParity(bits):
    #rounded = [int(bit + 0.5) for bit in bits]
    #strs    = [str(r)         for r   in rounded]
    #binStr  = ''.join(strs)
    #num     = int(binStr, 2)
    #return int(num % 5 == 0)
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

def search(net, Xs, Ts):
    res            = len(net[0][0])
    rateInv      = float(res)

    zipped       = zip(Xs, Ts)
    randomChoice = random.choice
    sampleSize   = res * res

    global fileNum
    print sampleSize

    while True:
        # Make a batch
        if fileNum <= res * 10:
            batch = [] # Warm it up!
        else:
            batch = [randomChoice(zipped) for i in range(sampleSize)]
        #batch = [randomChoice(zipped) for i in range(sampleSize)]

        # Take a step for each sample in batch
        for X, T in batch:
            GAs = backprop(net, X, T)

            grads, acts = zip(*GAs)
            grads = normaliseGradient(grads)
            GAs   = zip(grads, acts)

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
sequenceLength = 16
res            = 100

#m  = loadMnist()
#Xs = m[0]
#Ys = [float(m[1][i] == 0) for i in range(len(m[1]))]

#Xs = [[int(np.random.uniform() + 0.5) for i in range(sequenceLength)] for j in range(trainingSize)]
Xs = [[np.random.uniform() for i in range(sequenceLength)] for j in range(trainingSize)]
Ys = [evenParity(x) for x in Xs]

# Fit
bl = makeNet(sequenceLength, res)
#bl = load(1200)
#bl[0][0] = makeUnit(res, 0.5)
#normaliseNet(bl)
search(bl, Xs, Ys)

#backprop(bl, [0.1, 0.2, 0.3, 0.4]*2, 0.5)
#for i in range(5):
    #print 'go!'
    #for i in range(res*res):
        #backprop(bl, [0.1, 0.2, 0.3, 0.4]*2, 0.5)
    #print 'done'
