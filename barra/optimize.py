# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import cvxpy as cp
import numpy as np
import pandas as pd

from data import *

def calc_covariance(X, lamb, tau, lag=0, diag=False):
    N, T = X.shape
    weights = (lamb**(1/tau))**np.arange(1, T-lag+1)
    cov = np.zeros(shape=[N, N])
    for k in range(N):
        for l in range(N):
            if diag and k != l: continue
            cov[k][l] = np.nansum((X[k][:T-lag] - np.nanmean(X[k][:T-lag]))\
                                  *(X[l][lag:] - np.nanmean(X[l][lag:])) \
                                  *weights) / np.sum(weights)
    return cov

    
def optimize_portfolio(date):
    print('optimize %s'%date)
    print('calculate covariance...')
    X = load_factor_section(date).drop(['CAP', 'RET'], axis=1)
    F = load_coef_series(date, 252)
    U = load_resi_series(date, 252)
    # F
    F_raw = calc_covariance(F.T.values, 0.5, 90) #numpy
    C = 0; D = 2
    for i in range(1, D+1):
        Ci = calc_covariance(F.T.values, 0.5, 90, i)
        C += (1 - i/(D+1))*(Ci + Ci.T)
    F_nw = 21*(F_raw + C)
    # U
    U_raw = calc_covariance(U.T.values, 0.5, 90, diag=True)
    C = 0; D = 5
    for i in range(1, D+1):
        Ci = calc_covariance(U.T.values, 0.5, 90, i, diag=True)
        C += (1 - i/(D+1))*(Ci + Ci.T)
    U_nw = 21*(U_raw + C)
    # adjustment
    S = X.values.dot(F_nw).dot(X.values.T) + U_nw
    print('optimize portfolio...')
    P = cp.Variable(len(X))
    objective = cp.Minimize(cp.quad_form(P, S))
    constraints = [
        0 <= P, P <= 0.1,
        cp.sum(P) == 1
    ]
    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    P = pd.Series(P.value, index=X.index)
    save_port_section(P, date)
    
    
if __name__ == '__main__':
    
    with open('../data/calendar.txt') as f:
        for date in f:
            date = date.strip()
            if date < '20150210': continue
            if date > '20180427': break
            optimize_portfolio(date)
