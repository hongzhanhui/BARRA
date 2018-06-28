# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd

__all__ = (
    'load_factor_section', 'load_factor_series', 'load_coef_section',
    'load_coef_series', 'load_resi_section', 'load_resi_series',
    'save_factor_section', 'save_coef_section', 'save_resi_section',
    'save_port_section',
)

calendar_path = '../data/calendar.txt'
factor_path = '../data/factors.h5'
coef_path = '../data/coefs.h5'
resi_path = '../data/resis.h5'
port_path = '../data/portfolios.h5'

with open(calendar_path, 'r') as f:
    calendar = [x.strip() for x in f]
    calendar_index = {x:i for i,x in enumerate(calendar)}

def format_hdf_path(date):
    return '/d%s/data'%date
    
def load_data(fname, date, N=1):
    if date not in calendar_index:
        raise ValueError('not valid trade date: %s'%date)
    index = calendar_index[date]
    if index < N - 1:
        raise ValueError('not enough history data')
    data = dict()
    for date in calendar[index-N+1:index+1]:
        data[date] = pd.read_hdf(fname, format_hdf_path(date))
    if N == 1: return data[date]
    data = pd.DataFrame(data).loc[data[date].index].T.sort_index()
    return data
    
def load_factor_section(date):
    return load_data(factor_path, date)

def load_factor_series(date, N):
    return load_data(factor_path, date, N)

def save_factor_section(data, date):
    data.to_hdf(factor_path, format_hdf_path(date))
    
def load_coef_section(date):
    return load_data(coef_path, date)

def load_coef_series(date, N):
    return load_data(coef_path, date, N)

def save_coef_section(data, date):
    data.to_hdf(coef_path, format_hdf_path(date))
    
def load_resi_section(date):
    return load_data(resi_path, date)

def load_resi_series(date, N):
    return load_data(resi_path, date, N)

def save_resi_section(data, date):
    data.to_hdf(resi_path, format_hdf_path(date))

def save_port_section(data, date):
    data.to_hdf(port_path, format_hdf_path(date))
