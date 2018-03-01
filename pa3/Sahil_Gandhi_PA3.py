import math
import pdb
import random

import numpy as np
import pylab as pl
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture as GMM

''' util code '''
def plotGauss2D(pos, P, r = 2., color = 'black'):
    U, s, Vh = pl.linalg.svd(P)
    orient = math.atan2(U[1,0],U[0,0])*180/math.pi
    ellipsePlot = Ellipse(xy=pos, width=2*r*math.sqrt(s[0]),
              height=2*r*math.sqrt(s[1]), angle=orient,
              edgecolor=color, fill = False, lw = 3, zorder = 10)
    ax = pl.gca()
    ax.add_patch(ellipsePlot)
    return ellipsePlot
 
#############################
# Mixture Of Gaussians
#############################
 
# A simple class for a Mixture of Gaussians
class MOG:
    def __init__(self, pi = 0, mu = 0, var = 0):
        self.pi = pi
        self.mu = mu
        self.var = var
    def plot(self, color = 'black'):
        return plotGauss2D(self.mu, self.var, color=color)
    def __str__(self):
        return "[pi=%.2f,mu=%s, var=%s]"%(self.pi, self.mu.tolist(), self.var.tolist())
    __repr__ = __str__
 
def plotMOG(X, param, colors = ('blue', 'yellow', 'black', 'red', 'cyan'), title=''):
    fig = pl.figure()                   # make a new figure/window
    ax = fig.add_subplot(111, aspect='equal')
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    ax.set_xlim(min(x_min, y_min), max(x_max, y_max))
    ax.set_ylim(min(x_min, y_min), max(x_max, y_max))
    for (g, c) in zip(param, colors[:len(param)]):
        e = g.plot(color=c)
        ax.add_artist(e)
    plotData(X)
    pl.title(title)
    pl.show()    
 
def plotData(X):
    pl.plot(X[:,0:1].T[0],X[:,1:2].T[0], 'gs')
 
def varMat(s1, s2, s12 = 0):
    return pl.array([[s1, s12], [s12, s2]])
 
def randomParams(X, m=2):
    # m is the number of mixtures
    # this function is used to generate random mixture, in your homework you should use EM algorithm to get real mixtures.
    (n, d) = X.shape
    # A random mixture...
    return [MOG(pi=1./m, 
                mu=X[random.randint(0,n-1),:], 
                var=varMat(3*random.random(), 3*random.random(), 3*random.random()-1.5)) \
            for i in range(m)]

''' EM algorithm '''

def e(x, params, k):
    ''' computes the responsibilities (memberships) '''
    # responsibility = [[] for _ in range(x.shape[0])]
    responsibility = np.zeros(x.shape[0], k)
    for i in range(x.shape[0]):
        for j in range(k):
            responsibility[i].append(
                params['w'][j] * multivariate_normal.pdf(
                    x[i], params['mu'][j], params['sigma'][j], allow_singular=True
                )
            )
        responsibility[i] = responsibility[i]/sum(responsibility[i])
    return np.array(responsibility)

def m(x, params, responsibilities, k, covariance_type):
    ''' computes the distribution parameters; mu, sigma, w (mixture coefficients)  '''
    Nk = np.sum(responsibilities, axis=0)
    percent_assigned_to_cluster = np.divide(Nk, len(x))

    params['w'] = percent_assigned_to_cluster

    params['mu'] = [
        np.divide(
            np.sum(
                np.multiply(responsibilities.T[i].reshape(-1, 1), x),
                axis=0), Nk[i]) for i in range(k)
    ]

    print('x-mu : ', (x - params['mu'][0]).shape)
    print('rik: ', responsibilities.T[0].reshape(-1, 1).shape)
    print('rik: ', responsibilities[0].shape)
    print('nk:', Nk.shape)
    if covariance_type == 'full':
        params['sigma'] = [
            np.divide(
                np.dot((x - params['mu'][i]).T,
                    np.multiply(responsibilities.T[i].reshape(-1, 1),
                                (x - params['mu'][i]))), Nk[i])
            for i in range(k)
        ]
    elif covariance_type == 'diag':
        params['sigma'] = [
            np.divide(
                np.multiply(
                    responsibilities.T[i].reshape(-1, 1),
                    np.power(x - params['mu'][i], 2)
                ), Nk[i]
            )
            for i in range(k)
        ]
    print(params)
    exit(1)
    return params

def em(x, params, k, covariance_type='full', threshold=1e-8):
    iter = 0
    ll_lst = list()
    while True:
        iter += 1
        responsibilities = e(x, params, k)
        params = m(x, params, responsibilities, k, covariance_type)

        ll = 0
        print('..., ', params)
        for i in range(x.shape[0]):
            for j in range(k):
                # print('x', x[i])
                # print('m', params['mu'][j])
                # print('s', params['sigma'][j])
                a = responsibilities[i][j] * multivariate_normal.pdf(x[i], params['mu'][j], params['sigma'][j], allow_singular=True)
                b = responsibilities[i][j] * np.log(params['w'][j])
                ll+= a+b
        
        # print(responsibilities)
        # print(params)
        # print(ll)

        if iter > 2 and np.isclose(ll_lst[-1], ll, atol=threshold):
            break
        ll_lst.append(ll)
    print('iters : {}'.format(iter))
    print('params: {}'.format(params))
    print('loglik: {}'.format(ll))
    return iter, params, ll

def run(data, k, params=None, covariance_type='full'):
    x = np.loadtxt('data/'+data+'.txt')
    if params is None:
        params = {
            'mu': [],
            'sigma': [],
            'w': [],
        }
        for i in range(k):
            params["mu"].append(x[np.random.randint(0, x.shape[0]-1), :])
            params['sigma'].append(np.eye(x.shape[1]))
            params['w'].append(1./k)
    
    iters, params, obj = em(x, params, k, covariance_type)
    plotMOG(x, [MOG(pi=params["w"][i], mu=params["mu"][i], var=params["sigma"][i]) for i in range(k)], title=data)

# gm = GMM(2)
run('data_1_small', 2)#, covariance_type='diag')
# run('data_1_large', 2)
# gm.fit(np.loadtxt('data/data_1_large.txt'))
# run('data_2_small', 2)
# run('data_2_large', 2)
# gm.fit(np.loadtxt('data/data_2_large.txt'))
# run('data_3_small', 2)
# run('data_3_large', 2)
# gm.fit(np.loadtxt('data/data_3_large.txt'))
# run('mystery_1', 2)
# run('mystery_2', 2)
