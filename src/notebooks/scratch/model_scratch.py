#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  4 22:09:20 2024

@author: rowanlavelle
"""

#%%
import pandas as pd
import json
from sqlalchemy import create_engine
import os
import numpy as np
from datetime import datetime
import sys
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

sys.path.append('/Users/rowanlavelle/Documents/Projects/')
from nba.src.db_manager import DBManager
from sklearn.linear_model import LinearRegression

from scipy.ndimage import gaussian_filter1d
import scipy

#%%
def slope(x, y, i):    
    # Calculate slope using finite differences (central difference)
    dx = x[i] - x[i-1]
    dy = y[i] - y[i-1]
    slope = dy / dx
    
    return slope

#%%
def calc_post(obs, hist, signal):
    mu = np.mean(hist)
    tau = np.std(hist)
    
    pbar = obs.mean() + signal
    pstd = obs.std()
    
    n = len(obs)
    
    itau = 1/tau
    post_mu = (itau**2/(itau**2 + n/pstd**2))*mu + ((n/pstd**2)/(itau**2 + n/pstd**2))*pbar
    post_var = (((pstd**2) / n)*tau**2)/(tau**2 + (pstd**2)/n)
    return post_mu, post_var


#%%
db_path = '/Users/rowanlavelle/nba/nba.db'
engine = create_engine(f'sqlite:///{db_path}')

#%%
dbm = DBManager(engine)

#%%
df = dbm.get_player_stats_season('kevin-durant', '00')

#%%
ngames = 30
def objf(theta):
    y = []
    yhat = []
    ps = []
    for i in np.arange(200,len(df)):
        X = df.points.values[:i]
        pnew = X[-1]
        hist = X[:i-ngames]
        obs = X[i-ngames:-1]
        
        signal = (obs - obs.mean()).cumsum()
        smooth_sig = gaussian_filter1d(signal, theta[1])
        s = slope(np.arange(len(smooth_sig)), smooth_sig, len(obs)-1)*theta[0]
        
        post_mu, post_var = calc_post(obs, hist, s)
        y.append(pnew)
        yhat.append(post_mu)
        
        p = scipy.stats.norm.pdf(pnew, loc=post_mu, scale=np.sqrt(post_var))
        ps.append(p)
        
    y = np.array(y)
    yhat = np.array(yhat)
    
    mse = ((y-yhat)**2).mean()
    
    
    return mse

#%%
res = scipy.optimize.minimize(objf, x0=(1,1))

#%%
res

#%%
theta = res.x

#%%
y = []
yhat = []
ps = []
for i in np.arange(200,len(df)):
    X = df.points.values[:i]
    pnew = X[-1]
    hist = X[:i-ngames]
    obs = X[i-ngames:-1]
    
    signal = (obs - obs.mean()).cumsum()
    smooth_sig = gaussian_filter1d(signal, theta[1])
    s = slope(np.arange(len(smooth_sig)), smooth_sig, len(obs)-1)*theta[0]
    
    post_mu, post_var = calc_post(obs, hist, s)
    y.append(pnew)
    yhat.append(post_mu)
    
    p = scipy.stats.norm.pdf(pnew, loc=post_mu, scale=np.sqrt(post_var))
    ps.append(p)
    
y = np.array(y)
yhat = np.array(yhat)
ps = np.array(ps)

#%%
plt.clf()
plt.hist(ps,bins=100)
plt.show()

#%%
yy = y[np.argsort(ps)]
yh = yhat[np.argsort(ps)]

#%%
y0 = yy - yy.mean()
yh0 = yh - yh.mean()

#%%
uv, idx = np.unique(np.sort(ps),return_index=True)

#%%
plt.clf()
plt.plot(y0.cumsum())
plt.plot(yh0.cumsum())
plt.xticks(ticks=idx[::100], labels=np.round(uv[::100], 3))
plt.show()

#%%
y.mean(), yhat.mean()

#%%
a = y[ps > 0.05]
b = yhat[ps > 0.05]

#%%
np.sum(ps > 0.05) / len(ps)


#%%
((a-b)**2).mean()

#%%


