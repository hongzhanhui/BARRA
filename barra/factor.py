# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import statsmodels.api as sm

from data import *

def calc_factor_return(date):
    print('calculate factor return %s'%date)
    df = load_factor_section(date).dropna()
    Y = df['RET']
    W = 1/(df['CAP']**0.5)
    W /= W.sum()
    X = df.drop(['CAP', 'RET'], axis=1)
    res = sm.WLS(Y, X, weight=W).fit()
    save_coef_section(res.params, date)
    save_resi_section(res.resid, date)

if __name__ == '__main__':

    with open('../data/calendar.txt') as f:
        for date in f:
            date = date.strip()
            if date < '20140101': continue
            if date > '20180427': break
            calc_factor_return(date)
