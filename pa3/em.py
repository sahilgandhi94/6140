import itertools
import math
import pdb
import random
import traceback
from copy import deepcopy

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
import scipy
from matplotlib.patches import Ellipse
from scipy.misc import logsumexp
from scipy.stats import multivariate_normal
from sklearn.metrics import pairwise_distances_argmin
from sklearn.mixture import GaussianMixture

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
 
def plotMOG(X, param, colors = ('blue', 'yellow', 'black', 'red', 'cyan'), title='', save=False):
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
    if save:
        pl.savefig(title+'.png', bbox_inches='tight')
    else:
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

class KMeans:
    def __init__(self, datasrc, n_clusters):
        self.x = np.loadtxt('data/'+datasrc+'.txt')
        self.n_elements = self.x.shape[0]
        self.n_clusters = n_clusters
        self._plot_title = datasrc+' KMeans; K='+str(n_clusters)

    def fit(self, plot=False, rseed=2, **kwargs):
        # 1. Randomly choose clusters
        rng = np.random.RandomState(rseed)
        i = rng.permutation(self.x.shape[0])[:self.n_clusters]
        centers = self.x[i]
        metric = []
        while True:
            labels = pairwise_distances_argmin(X=self.x, Y=centers)
            new_centers = np.array([self.x[labels == i].mean(0) for i in range(self.n_clusters)]).reshape(self.n_clusters, -1)
            metric.append(np.linalg.norm([self.x[i] - new_centers[j] for i, j in enumerate(labels)]))
            if np.allclose(centers, new_centers):
                break
            centers = new_centers
        self.centroids = centers
        self.labels = labels
        self.objective_list = metric

        if plot:
            self.plot(kwargs)

    def plot(self, save=False, **kwargs):
        for i in range(self.n_clusters):
            _data = self.x[np.where(self.labels == i)]
            pl.scatter(_data.T[0], _data.T[1])
        pl.title(self._plot_title)
        if save:
            pl.savefig(self._plot_title+'.png', bbox_inches='tight')
            pl.clf()
        else:
            pl.show()
    
    @staticmethod
    def plot_k_vs_objective(datasrc, ks=list(range(1,6)), title='', save=False):
        title = title if title else datasrc+' #Components vs KMeans objective'
        iters = []
        lls = []
        for k in ks:
            g = KMeans(datasrc, k)
            g.fit()
            lls.append(g.objective_list[-1])
            iters.append(len(g.objective_list))

        pl.plot(ks, lls, label='Objective', color='red')
        pl.plot(ks, iters, label='Iteration', color='blue')
        pl.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
        pl.xlabel('Number of components (K)')
        pl.title(title)
        if save:
            pl.savefig(title+'.png', bbox_inches='tight')
            pl.clf()
        else:
            pl.show()

class GMM:
    def __init__(self, datasrc, n_clusters, x=None, mu=None, cov=None, w=None, covariance_type='full', convergence_threshold=1e-8):
        self.x = np.loadtxt('data/'+datasrc+'.txt') if x is None else x
        self.n_clusters = n_clusters
        self.n_elements = self.x.shape[0]
        self.n_dim = self.x.shape[1]
        self.covariance_type = covariance_type
        self.convergence_threshold = convergence_threshold
        self.log_likelihoods = list()
        self._plot_title = datasrc+' GMM; Cov Type='+covariance_type

        pmu = []
        pcov = []
        pw = []
        if mu is None or cov is None or w is None:
            for i in range(n_clusters):
                pmu.append(self.x[i,:])
                pcov.append(np.eye(self.x.shape[1])) # identity
                # pcov.append(np.ones((self.x.shape[1], self.x.shape[1]))) # all 1's
                pw.append(1./n_clusters)

        self.mu = pmu if mu is None else mu
        self.cov = pcov if cov is None else cov
        self.w = pw if w is None else w

    def e(self):
        ''' computes the responsibilities (memberships) '''
        rs = []
        cov = deepcopy(self.cov)
        for i in range(self.n_elements):
            xi = self.x[i, :]
            nums = []
            for j in range(self.n_clusters):
                cov_k = self.cov[j]
                cov_k += 1e-9*np.eye(self.n_dim)  # shift the covariance matrix by an epsilon
                diff = xi - self.mu[j]
                L = np.linalg.cholesky(cov_k)
                log_det = 2*np.sum(np.log(np.diag(L)))
                num = np.log(self.w[j]) - np.dot(diff.T, np.dot(np.linalg.pinv(cov_k), diff))*0.5 - 0.5*self.n_dim*np.log(2*np.pi) * 0.5*log_det
                nums.append(num.item())
            nums = np.exp(np.array(nums) - logsumexp(nums))
            rs.append(nums)
        self.R = np.array(rs)        

    def m(self):
        ''' computes the parameters of the gaussian '''
        wts = []
        covs = []
        mus = []
        for k in range(self.n_clusters):            
            r_k = self.R[:, k].reshape(-1, 1)
            wts.append(np.sum(r_k) / self.n_elements)
            mu_k = np.sum(r_k * self.x, axis=0) / np.sum(r_k)
            mus.append(mu_k)
            diff = self.x - mu_k
            if(self.covariance_type=='full'):
                _cov = np.dot((r_k * diff).T, diff) / np.sum(r_k)
            elif(self.covariance_type=='diag'):
                _cov = np.diag(np.sum(r_k * np.power(diff, 2), axis=0)/np.sum(r_k))
            else:
                raise ValueError()
            covs.append(_cov)
        self.cov = np.array(covs)
        self.wt = np.array(wts)
        self.mu = np.array(mus)

    def likelihood(self, predicted=False):
        mem = self.R if predicted is False else self.PP
        cov = deepcopy(self.cov)
        ll = 0
        for i in range(self.n_elements):
            xi = self.x[i, :]
            nums = []
            for j in range(self.n_clusters):
                cov_k = self.cov[j]
                cov_k += 1e-9*np.eye(self.n_dim)  # shift the covariance matrix by an epsilon
                diff = self.x[i] - self.mu[j]
                L = np.linalg.cholesky(cov_k)
                log_det = 2*np.sum(np.log(np.diag(L)))
                val =  - np.dot(diff.T, np.dot(np.linalg.pinv(cov_k), diff))*0.5 - 0.5*self.n_dim*np.log(2*np.pi) * 0.5*log_det
                ll += (mem[i, j]*val + mem[i, j]*np.log(self.w[j]))
        return ll

    def predict_probabilities(self, y=None):
        ''' computes the predicted memberships '''
        rs = []
        _x = self.x if y is None else y
        cov = deepcopy(self.cov)
        for i in range(_x.shape[0]):
            xi = _x[i, :]
            nums = []
            for j in range(self.n_clusters):
                cov_k = self.cov[j]
                cov_k += 1e-9*np.eye(self.n_dim)  # shift the covariance matrix by an epsilon
                diff = xi - self.mu[j]
                L = np.linalg.cholesky(cov_k)
                log_det = 2*np.sum(np.log(np.diag(L)))
                num = np.log(self.w[j]) - np.dot(diff.T, np.dot(np.linalg.pinv(cov_k), diff))*0.5 - 0.5*self.n_dim*np.log(2*np.pi) * 0.5*log_det
                nums.append(num.item())
            nums = np.exp(np.array(nums) - logsumexp(nums))
            rs.append(nums)
        self.PP = np.array(rs)
            
    def fit(self, plot=False, **kwargs):
        ''' run the em loop till convergence '''
        while True:
            self.e()
            self.m()
            ll = self.likelihood()                        
            if len(self.log_likelihoods) > 2 and np.isclose(self.log_likelihoods[-1], ll, atol=self.convergence_threshold):
                break
            self.log_likelihoods.append(ll)
        # print('inter: ', len(self.log_likelihoods), '; ll: ', self.log_likelihoods[-1])
        if plot:
            self._plot_title += '\n#iterations: '+str(len(self.log_likelihoods))+' LL: '+str(self.log_likelihoods[-1])
            self.plot(kwargs)

    def plot(self, save=False, **kwargs):
        plotMOG(
            self.x, [MOG(pi=self.w[i], mu=self.mu[i], var=self.cov[i]) for i in range(self.n_clusters)], 
            title=self._plot_title, save=save
        )
    
    # def plot_vs_avg_ll()

    @staticmethod
    def plot_k_vs_ll_iter(datasrc, ks=list(range(1,6)), covariance_type='full', convergence_threshold=1e-8, title='', save=False):
        title = title if title else datasrc+' #Components vs LL & iterations; Cov Type='+covariance_type
        iters = []
        lls = []
        for k in ks:
            g = GMM(datasrc, k, covariance_type=covariance_type, convergence_threshold=convergence_threshold)
            g.fit()
            lls.append(g.log_likelihoods[-1])
            iters.append(len(g.log_likelihoods))

        pl.plot(ks, lls, label='Log Likelihoods', color='red')
        pl.plot(ks, iters, label='Iterations', color='blue')
        pl.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
        pl.xlabel('Number of components (k)')
        pl.title(title)
        if save:
            pl.savefig(title+'.png', bbox_inches='tight')
            pl.clf()
        else:
            pl.show()

    @staticmethod
    def q3_1(datasrc_small, datasrc_large, save=False):
        i=0
        # iters = []
        for covariance_type in ['full', 'diag']:
            lls = []
            lls_large = []
            for n_clus in range(1, 6):            
                i+=1
                print('Model {}; #Clusters: {}; Covariance Type: {}'.format(i, n_clus, covariance_type))
                g = GMM(datasrc_small, n_clus, covariance_type=covariance_type)
                g.fit()
                lls.append(g.log_likelihoods[-1]/g.n_elements)
                # iters.append(len(g.log_likelihoods))
                gl = GMM(datasrc_large, n_clus, mu=g.mu, cov=g.cov, w=g.w, covariance_type=covariance_type)
                # gl.fit()
                # lls_large.append(gl.log_likelihoods[-1]/gl.n_elements)
                gl.predict_probabilities()
                lls_large.append(gl.likelihood(predicted=True)/gl.n_elements)

            title = datasrc_small.replace('_small', '')+' Model vs Avg-LL; Cov Type='+covariance_type
            pl.plot(list(range(1, 6)), lls, label='Train Data')
            pl.plot(list(range(1, 6)), lls_large, label='Test Data')
            # pl.plot(list(range(1, 11)), iters, label='Iterations', color='blue')
            pl.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
            pl.xlabel('Models')
            pl.ylabel('Average Log-Likelihood')
            # pl.yscale('log')
            pl.title(title)
            if save:
                pl.savefig(title+'.png', bbox_inches='tight')
                pl.clf()
            else:
                pl.show()

    @staticmethod
    def q3_3(datasrc_small, datasrc_large, cross_validation_val, save=False):
        i=0
        for covariance_type in ['full', 'diag']:
            lls = []
            lls_large = []
            for n_clus in range(1, 6):            
                i+=1
                print('model', i)
                scores, wts, mus, covs = cross_validation_score(datasrc_small, cross_validation_val, n_clus, covariance_type)
                lls.append(np.average(scores))

                _i = np.argmax(np.array(scores)) # choose best model

                g = GMM(datasrc_large, n_clus, cov=covs[_i], w=wts[_i], mu=mus[_i], covariance_type=covariance_type)
                g.predict_probabilities() # test on large dataset
                lls_large.append(g.likelihood(predicted=True)/g.n_elements)
                

            title = datasrc_small.replace('_small', '')+' Model vs Cross Validation Avg-LL; '+str(cross_validation_val)+'-fold; Cov Type='+covariance_type
            pl.plot(list(range(1, 6)), lls, label='Small Data')
            pl.plot(list(range(1, 6)), lls_large, label='Large Data')
            pl.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
            pl.xlabel('Models')
            pl.ylabel('Cross Validation Average Log-Likelihood')
            pl.title(title)
            if save:
                pl.savefig(title+'.png', bbox_inches='tight')
                pl.clf()
            else:
                pl.show()

def bic_plots(name):
    x = np.loadtxt('data/{}.txt'.format(name))
    n_clusters = [1, 2, 3, 4, 5]
    bic = []
    lowest_bic = np.infty
    cov_typs = ['full', 'diag']
    best_cov = None
    best_wt = None
    best_mu = None
    for cv in cov_typs:
        for k in n_clusters:
            g = GaussianMixture(k, covariance_type=cv)
            g.fit(x)
            bic.append(g.bic(x))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_cov = g.covariances_
                best_wt = g.weights_
                best_mu = g.means_

    bic = np.array(bic)

    bars = []
    spl = plt.subplot(1, 1, 1)
    color_iter = itertools.cycle(['cornflowerblue', 'darkorange'])
    for i, (cv_type, color) in enumerate(zip(cov_typs, color_iter)):
        xpos = np.array(n_clusters) + .2 * (i - 2)
        bars.append(pl.bar(xpos, bic[i * len(n_clusters):
                                    (i + 1) * len(n_clusters)],
                            width=.2, color=color))

    plt.xticks(n_clusters)
    plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
    plt.title('BIC score per model')
    xpos = np.mod(bic.argmin(), len(n_clusters)) + .65 +\
        .2 * np.floor(bic.argmin() / len(n_clusters))
    plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
    spl.set_xlabel('Number of components')
    spl.legend([b[0] for b in bars], cov_typs)
    plt.savefig('{} BIC score per model'.format(name))
    plt.clf()

def cross_validation_score(name, cross_validation_val, k, cov_typ):
    data = np.loadtxt('data/{}.txt'.format(name))
    s=0
    step = int(data.shape[0]/cross_validation_val)
    e = step
    scores = []
    wts = []
    mus = []
    covs = []
    for i in range(cross_validation_val):
        test = data[s:e,:]
        train = np.concatenate((data[0:s,:], data[e:,:]))
        s = e
        e += step

        g = GMM(name, k, x=train, covariance_type=cov_typ)
        g.fit()
        g_test = GMM(name, k, w=g.w, mu=g.mu, cov=g.cov, x=test, covariance_type=cov_typ)
        g_test.predict_probabilities()
        covs.append(g.cov)
        mus.append(g.mu)
        wts.append(g.w)
        scores.append(g_test.likelihood(predicted=True)/g_test.n_elements)

    return scores, wts, mus, covs
        
sm_datasrcs = ['data_1_small', 'data_1_large', 'data_2_small', 'data_2_large', 'data_3_small', 'data_3_large']
mystery_datasrcs = ['mystery_1', 'mystery_2']

if __name__ == '__main__':
    # GMM.plot_k_vs_ll_iter('data_1_small', save=True)
    # GMM.plot_k_vs_ll_iter('data_1_small', covariance_type='diag', save=True)
    # GMM.plot_k_vs_ll_iter('data_1_large', save=True)
    # GMM.plot_k_vs_ll_iter('data_1_large', covariance_type='diag', save=True)
    # GMM.plot_k_vs_ll_iter('data_2_small', save=True)
    # GMM.plot_k_vs_ll_iter('data_2_small', covariance_type='diag', save=True)
    # GMM.plot_k_vs_ll_iter('data_2_large', save=True)
    # GMM.plot_k_vs_ll_iter('data_2_large', covariance_type='diag', save=True)
    # GMM.plot_k_vs_ll_iter('data_3_small', save=True)
    # GMM.plot_k_vs_ll_iter('data_3_small', covariance_type='diag', save=True)
    # GMM.plot_k_vs_ll_iter('data_3_large', save=True)
    # GMM.plot_k_vs_ll_iter('data_3_large', covariance_type='diag', save=True)

    # k=3
    # d1s_fgmm = GMM('data_1_small', k).fit(plot=True, save=True)
    # d1s_dgmm = GMM('data_1_small', k, covariance_type='diag').fit(plot=True, save=True)
    # d1l_fgmm = GMM('data_1_large', k).fit(plot=True, save=True)
    # d1l_dgmm = GMM('data_1_large', k, covariance_type='diag').fit(plot=True, save=True)

    # d2s_fgmm = GMM('data_2_small', k).fit(plot=True, save=True)
    # d2s_dgmm = GMM('data_2_small', k, covariance_type='diag').fit(plot=True, save=True)
    # d2l_fgmm = GMM('data_2_large', k).fit(plot=True, save=True)
    # d2l_dgmm = GMM('data_2_large', k, covariance_type='diag').fit(plot=True, save=True)

    # d3s_fgmm = GMM('data_3_small', k).fit(plot=True, save=True)
    # d3s_dgmm = GMM('data_3_small', k, covariance_type='diag').fit(plot=True, save=True)
    # d3l_fgmm = GMM('data_3_large', k).fit(plot=True, save=True)
    # d3l_dgmm = GMM('data_3_large', k, covariance_type='diag').fit(plot=True, save=True)

    # d1m_fgmm = GMM('mystery_1', 2).fit(plot=True)
    # d1m_dgmm = GMM('mystery_1', 2, covariance_type='diag').fit(plot=True)

    # d2m_fgmm = GMM('mystery_2', 2).fit(plot=True)
    # d2m_dgmm = GMM('mystery_2', 2, covariance_type='diag').fit(plot=True)


    # KMeans.plot_k_vs_objective('data_1_small', save=True)
    # KMeans.plot_k_vs_objective('data_1_large', save=True)
    # KMeans.plot_k_vs_objective('data_2_small', save=True)
    # KMeans.plot_k_vs_objective('data_2_large', save=True)
    # KMeans.plot_k_vs_objective('data_3_small', save=True)
    # KMeans.plot_k_vs_objective('data_3_large', save=True)


    # km1s = KMeans('data_1_small', 3)
    # km1s.fit()#plot=True, save=True)
    # km1l = KMeans('data_1_large', 3).fit()#plot=True, save=True)
    # km2s = KMeans('data_2_small', 3).fit()#plot=True, save=True)
    # km2l = KMeans('data_2_large', 3).fit()#plot=True, save=True)
    # km3s = KMeans('data_3_small', 3).fit()#plot=True, save=True)
    # km3l = KMeans('data_3_large', 2).fit()#plot=True, save=True)

    # ks = [3,3,3,3,2]
    # for i in range(len(sm_datasrcs)):
    #     ds = sm_datasrcs[i]
    #     k = ks[i]
    #     km1s = KMeans(ds, k)
    #     km1s.fit()
    #     mu = km1s.centroids
    #     w = np.divide(np.unique(km1s.labels, return_counts=True)[1], km1s.n_elements)
    #     GMM(ds, k, mu=mu, w=w).fit(plot=True, save=True)
    #     GMM(ds, k, mu=mu, w=w, covariance_type='diag').fit(plot=True, save=True)

    # GMM.q3_1('data_1_small', 'data_1_large', save=True)
    # GMM.q3_1('data_2_small', 'data_2_large', save=True)
    # GMM.q3_1('data_3_small', 'data_3_large', save=True)

    # GMM('data_1_small', 4, covariance_type='diag').fit(plot=True, save=True)
    # GMM('data_1_large', 4, covariance_type='diag').fit(plot=True, save=True)

    # GMM('data_2_small', 4, covariance_type='diag').fit(plot=True, save=True)
    # GMM('data_2_large', 4, covariance_type='diag').fit(plot=True, save=True)

    # GMM('data_3_small', 5, covariance_type='full').fit(plot=True, save=True)
    # GMM('data_3_large', 5, covariance_type='full').fit(plot=True, save=True)

    print('data 1')
    GMM.q3_3('data_1_small', 'data_1_large', 39, save=True)
    print('data 2')
    GMM.q3_3('data_2_small', 'data_2_large', 39, save=True)
    print('data 3')
    GMM.q3_3('data_3_small', 'data_3_large', 39, save=True)

    # GMM('data_1_small', 3, covariance_type='diag').fit(plot=True, save=True)
    # GMM('data_1_large', 3, covariance_type='diag').fit(plot=True, save=True)

    # GMM('data_2_small', 2, covariance_type='full').fit(plot=True, save=True)
    # GMM('data_2_large', 2, covariance_type='full').fit(plot=True, save=True)

    # GMM('data_3_small', 2, covariance_type='full').fit(plot=True, save=True)
    # GMM('data_3_large', 2, covariance_type='full').fit(plot=True, save=True)

    # bic_plots('mystery_1')
    # bic_plots('mystery_2')

    # GMM('mystery_1', 1).fit(plot=True, save=True)
    # GMM('mystery_2', 1).fit(plot=True, save=True)
