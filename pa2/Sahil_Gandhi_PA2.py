
import math
import pdb
import traceback

import cvxopt
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from scipy.optimize import fmin_bfgs

''' Util Code '''
sigmoid = lambda x: 1 / (1 + np.exp(np.negative(x)))

def plotDecisionBoundary(X, Y, scoreFn, values, title = ""):
    # Plot the decision boundary. For that, we will asign a score to
    # each point in the mesh [x_min, m_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = max((x_max-x_min)/200., (y_max-y_min)/200.)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    zz = np.array([scoreFn(x) for x in np.c_[xx.ravel(), yy.ravel()]])
    zz = zz.reshape(xx.shape)
    print(zz)
    print(zz.shape)
    pl.figure()
    CS = pl.contour(xx, yy, zz, values, colors = 'green', linestyles = 'solid', linewidths = 2)
    pl.clabel(CS, fontsize=9, inline=1)
    # Plot the training points
    pl.scatter(X[:, 0], X[:, 1], c=(1.-Y), s=50, cmap = pl.cm.cool)
    # pl.scatter(X[:, 0], X[:, 1])
    pl.title(title)
    pl.axis('tight')
    pl.show()

''' Logistic Regression '''
# LR training code - l2 regularization 
def costLR(w, X, y, penalty=0.000001): # NLL for y in {-1,1}
    cost = np.sum(
        [np.log(1 + np.exp(np.negative(np.dot(np.dot(w.T, X[i]), y[i])))) 
            for i in range(len(X))]
    )
    return cost + penalty*(np.dot(w.T, w))

def runLR(X, Y, penalty):
    ithetaLR = np.random.rand(X.shape[1], 1)
    optThetaLR = fmin_bfgs(f=costLR, x0=ithetaLR, args=(X, Y, penalty))
    return optThetaLR

# Define the predictLR(x) function, which uses trained parameters
def predictLR(x, w, poly):
    if poly:
        x = np.hstack((1,x))
        for i in range(x.shape[0]):
            for j in range(i,x.shape[0]):
                x_ij = np.multiply(x[i],x[j])
            x = np.hstack((x,x_ij))
        x = x.reshape(-1,1)
    return sigmoid(w.T.dot(x))

def misclassificationLR(X, y, w, poly):
    ''' calculates the misclassifications in the given data '''
    yp = np.matrix([[1 if predictLR(X[i], w, poly) > 0.5 else -1] for i in range(X.shape[0])])
    return y.shape[0] - np.sum(y==yp)

def generatePhi(X):
    phi = np.append(np.ones((len(X),1)), X, axis=1)
    for i in range(X.shape[1]):
        for j in range(i, X.shape[1]):
            xij = np.multiply(X[:,i],X[:,j])     
            phi=np.append(phi, np.reshape(xij,(phi.shape[0],1)), axis = 1)

    return phi

def mainLR(name, penalty, poly=False):
    # parameters
    print('======Training======')
    # load data from csv files
    train = np.loadtxt('data/data_'+name+'_train.csv')
    X = train[:,0:2]
    Y = train[:,2:3]

    if poly:
        phi = generatePhi(X)
    else:
        phi = X

    optThetaLR = runLR(phi, Y, penalty)
    missclfn = misclassificationLR(X, Y, optThetaLR, poly)

    # plot training results
    plotDecisionBoundary(X, Y.ravel(), lambda x: predictLR(x, optThetaLR, poly=poly), [0.5], title = 'LR Train on {} data, lambda: {}; misclassification: {}'.format(name, penalty, missclfn))

    print('======Validation======')
    # load data from csv files
    validate = np.loadtxt('data/data_'+name+'_validate.csv')
    X = validate[:,0:2]
    Y = validate[:,2:3]
    if poly:
        phi = generatePhi(X)
    else:
        phi = X

    missclfn = misclassificationLR(X, Y, optThetaLR, poly)
    print('misclassifications: ', missclfn)
    # plot validation results
    plotDecisionBoundary(X, Y.ravel(), lambda x: predictLR(x, optThetaLR, poly=poly), [0.5], title = 'LR Validate on {} data, lambda: {}; misclassification: {}'.format(name, penalty, missclfn))

# mainLR('ls', 0)
# mainLR('ls', 0.8)
# mainLR('ls', 7)
# mainLR('nls', 0)
# mainLR('nls', 0.8)
# mainLR('nls', 7)
# mainLR('nonlin', 0)
# mainLR('nonlin', 0.8)
# mainLR('nonlin', 7)

# mainLR('ls', 0, poly=True)
# mainLR('ls', 0.8, poly=True)
# mainLR('ls', 7, poly=True)
# mainLR('nls', 0, poly=True)
# mainLR('nls', 0.8, poly=True)
# mainLR('nls', 7, poly=True)
# mainLR('nonlin', 0, poly=True)
# mainLR('nonlin', 0.8, poly=True)
# mainLR('nonlin', 7, poly=True)


''' SVM '''
def primal(X, y, C):
    n, m = X.shape
    
    P = np.identity(n+m+1)
    q = np.vstack((np.zeros((m+1,1)), C*np.ones((n,1))))
    G = np.zeros((2*n, m+1+n))
    G[:n,0:m] = y*X
    G[:n,m] = y.T
    G[:n,m+1:]  = np.identity(n)
    G[n:,m+1:] = np.identity(n)
    G = -G
    h = np.zeros((2*n,1))
    h[:n] = -1

    P = cvxopt.matrix(P,P.shape, tc='d')
    q = cvxopt.matrix(q,q.shape, tc='d')
    G = cvxopt.matrix(G,G.shape, tc='d')
    h = cvxopt.matrix(h,h.shape, tc='d')
    
    sol = cvxopt.solvers.qp(P, q, G, h)
    xs = np.ravel(sol['x'])
    # alpha = xs
    bias = xs[m]
    wt = xs[:m]

    return bias, wt

def dual(X, y, C):
    n, m = X.shape
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i,j] = np.dot(X[i], X[j])

    P = cvxopt.matrix(np.outer(y,y) * K)
    q = cvxopt.matrix(np.ones(n) * -1)
    A = cvxopt.matrix(y, (1,n))
    b = cvxopt.matrix(0.0)

    if C is None:
        G = cvxopt.matrix(np.diag(np.ones(n) * -1))
        h = cvxopt.matrix(np.zeros(n))
    else:
        tmp1 = np.diag(np.ones(n) * -1)
        tmp2 = np.identity(n)
        G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
        tmp1 = np.zeros(n)
        tmp2 = np.ones(n) * C
        h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

    sol = cvxopt.solvers.qp(P, q, G, h, A, b)

    xs = np.ravel(sol['x'])

    sv = xs>1e-5
    support_vectors = np.where(sv)
    alphas = np.array(xs[support_vectors])
    X_sv = X[support_vectors]
    y_sv = y[support_vectors]
    
    # weight = sum(ai.yi.xi)
    wt = np.zeros(m).T
    for i in range(len(alphas)):
        wt += alphas[i] * y_sv[i] * X_sv[i]

    # bias = yk - w.xk (for any k such that C > alpha_k > 0)
    try:
        _indices = np.array(np.where(C > alphas))
        _index = _indices[0][0]
        bias = (y_sv[_index] - np.dot(X_sv[_index], wt))[0]
        print('bias', bias)
    except IndexError:
        raise IndexError('No multiplier found suitable for bias term') 

    return bias, wt

def predictSVM(X, wt, bias):
    return np.dot(X, wt) + bias

def misclassificationSVM(X, y, pfun):
    yp = np.sign(pfun(X)).reshape(y.shape)
    return y.shape[0] - np.sum(y==yp)

def mainSVM(name, C, form='dual'):
    assert form in ['primal', 'dual']

    # parameters
    print('======Training======')
    # load data from csv files
    train = np.loadtxt('data/data_'+name+'_train.csv')
    # use deep copy here to make cvxopt happy
    X = train[:, 0:2].copy()
    Y = train[:, 2:3].copy()
    # Carry out training, primal and/or dual
    if form == 'primal':
        bias, wt = primal(X, Y, C)
    else:  # dual
        bias, wt = dual(X, Y, C)
    pfun = lambda x: predictSVM(x, wt, bias)
    missclfn = misclassificationSVM(X, Y, pfun)

    # plot training results
    plotDecisionBoundary(X, Y.ravel(), pfun, [-1, 0, 1], title = 'SVM {} Train on {} data, C={}; misclassification = {}'.format(form, name, C, missclfn))


    print('======Validation======')
    # load data from csv files
    validate = np.loadtxt('data/data_'+name+'_validate.csv')
    X = validate[:, 0:2]
    Y = validate[:, 2:3]
    missclfn = misclassificationSVM(X, Y, pfun)

    # plot validation results
    plotDecisionBoundary(X, Y.ravel(), pfun, [-1, 0, 1], title = 'SVM {} Validate on {} data, C={}; misclassification = {}'.format(form, name, C, missclfn))

# mainSVM('ls', 0.5, 'primal')
# mainSVM('ls', 10, 'primal')
# mainSVM('ls', 0.5, 'dual')
# mainSVM('ls', 10, 'dual')
# mainSVM('nls', 0.5, 'primal')
# mainSVM('nls', 10, 'primal')
# mainSVM('nls', 0.5, 'dual')
# mainSVM('nls', 10, 'dual')
# mainSVM('nonlin', 0.5, 'primal')
# mainSVM('nonlin', 10, 'primal')
# mainSVM('nonlin', 0.5, 'dual')
# mainSVM('nonlin', 10, 'dual')


''' Kernels and SVM '''

def gaussian(x, y, sigma=0.5):
    return np.exp(-np.linalg.norm(x-y)**2 / (2*(sigma**2)))

def polynomial(x, y, d=3, c=1):
    return (np.dot(x, y) + c)**d

def linear(x, y):
    return np.dot(x, y)

def dual_kernel(X, y, C, kernel):
    n, m = X.shape
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i,j] = kernel(X[i], X[j])

    P = cvxopt.matrix(np.outer(y,y) * K)
    q = cvxopt.matrix(np.ones(n) * -1)
    A = cvxopt.matrix(y, (1,n))
    b = cvxopt.matrix(0.0)

    if C is None:
        G = cvxopt.matrix(np.diag(np.ones(n) * -1))
        h = cvxopt.matrix(np.zeros(n))
    else:
        tmp1 = np.diag(np.ones(n) * -1)
        tmp2 = np.identity(n)
        G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
        tmp1 = np.zeros(n)
        tmp2 = np.ones(n) * C
        h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

    sol = cvxopt.solvers.qp(P, q, G, h, A, b)

    a = np.ravel(sol['x'])
    xs = a>1e-5
    ind = np.arange(len(a))[xs]
    a = a[xs]
    X_sv = X[xs]
    y_sv = y[xs]
    b = 0
    for n in range(len(a)):
        b += (y_sv[n] - np.sum(a * y_sv * K[ind[n],xs]))
    b /= len(a)

    return b, a, y_sv, X_sv

def predictSVMKernel(X, bias, a, sv_y, sv, kernel):
    y_predict = np.zeros(len(X))
    s=0
    for a, sv_y, sv in zip(a, sv_y, sv):
        s += a * sv_y * kernel(X, sv)
    y_predict = s
    # print(bias, y_predict)
    return y_predict

def misclassificationSVMKernel(X, y, pfun):
    yp = np.sign([pfun(x) for x in X]).reshape(y.shape)
    return y.shape[0] - np.sum(y==yp)

def mainSVMKernel(name, C, kernel):

    # parameters
    print('======Training======')
    # load data from csv files
    train = np.loadtxt('data/data_'+name+'_train.csv')
    # use deep copy here to make cvxopt happy
    X = train[:, 0:2].copy()
    Y = train[:, 2:3].copy()
    
    # Carry out training, primal and/or dual
    b, a, sv_y, sv = dual_kernel(X, Y, C, kernel)
    pfun = lambda x: predictSVMKernel(x, b, a, sv_y, sv, kernel)
    missclfn = misclassificationSVMKernel(X, Y, pfun)
    print('missclfn', missclfn)
    
    print('plotting..')
    # plot training results
    plotDecisionBoundary(X, Y.ravel(), pfun, [-1, 0, 1], title = 'SVM dual-{} Train on {} data, C={}; misclassification = {}'.format(kernel.__name__, name, C, missclfn))

    print('======Validation======')
    # load data from csv files
    validate = np.loadtxt('data/data_'+name+'_validate.csv')
    X = validate[:, 0:2]
    Y = validate[:, 2:3]
    missclfn = misclassificationSVMKernel(X, Y, pfun)
    print('missclfn', missclfn)
    
    print('Plotting..')
    # plot validation results
    plotDecisionBoundary(X, Y.ravel(), pfun, [-1, 0, 1], title = 'SVM dual-{} Validate on {} data, C={}; misclassification = {}'.format(kernel.__name__, name, C, missclfn))

# gaus005 = lambda x, y: gaussian(x, y, 0.05)
# gaus05 = lambda x, y: gaussian(x, y, 0.5)
# gaus5 = lambda x, y: gaussian(x, y, 5)
# mainSVMKernel('ls', 10, gaus005)
# mainSVMKernel('ls', 10, gaus05)
# mainSVMKernel('ls', 10, gaus5)

# mainSVMKernel('nls', 10, gaus005)
# mainSVMKernel('nls', 10, gaus05)
# mainSVMKernel('nls', 10, gaus5)

# mainSVMKernel('nonlin', 10, gaus005)
# mainSVMKernel('nonlin', 10, gaus05)
# mainSVMKernel('nonlin', 10, gaus5)


def cVSMisclassfications():
    data = ['ls', 'nls', 'nonlin']
    # data = ['nonlin']
    cs = np.arange(1, 12, 1)
    kernels = [polynomial]
    for datum in data:
        train = np.loadtxt('data/data_'+datum+'_train.csv')
        validate = np.loadtxt('data/data_'+datum+'_validate.csv')
        X_train = train[:, 0:2].copy()
        Y_train = train[:, 2:3].copy()        
        X_val = validate[:, 0:2]
        Y_val = validate[:, 2:3]
        missclfn_train = list()
        missclfn_val = list()
        for c in cs:
            for kernel in kernels:
                b, a, sv_y, sv = dual_kernel(X_train, Y_train, c, kernel)
                pfun = lambda x: predictSVMKernel(x, b, a, sv_y, sv, kernel)
                missclfn_train.append(misclassificationSVMKernel(X_train, Y_train, pfun))
                missclfn_val.append(misclassificationSVMKernel(X_val, Y_val, pfun))
        plt.plot(cs, missclfn_train, 'r')
        plt.plot(cs, missclfn_val, 'g')
        plt.xlabel('C values')
        plt.ylabel('Mis-classifications')
        plt.title('{} data with {} kernel'.format(datum, kernel.__name__))
        plt.show()

# cVSMisclassfications()