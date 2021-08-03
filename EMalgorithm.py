# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 14:46:35 2021

@author: MSBak
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import random
#%% 데이터 만들기

msmu = [0, 15]
msvars = [12, 3]
n_data = 1000
# phi = [0.7, 0.3]

X = []
for i in range(n_data):
    if random.random() < 0.7:
        tmp = np.random.normal() + msmu[0] + np.sqrt(msvars[0])
    else:
        tmp = np.random.normal() + msmu[1] + np.sqrt(msvars[1])
    X.append(tmp)
X = np.array(X)
plt.hist(X, bins=100)


#%%
# init

sample0 = random.sample(list(X), 20)
sample1 = random.sample(list(X), 20)

mu = np.mean(sample0)
sigma = np.std(sample0)
pi = 0.5

mu2 = np.mean(sample1)
sigma2 = np.std(sample1)

plt.figure()
plt.hist(X, bins=100, density=True)
x = np.linspace(-5, 25, 1000)
p = stats.norm.pdf(x, mu, sigma); plt.plot(x, p)
p = stats.norm.pdf(x, mu2, sigma2); plt.plot(x, p)

#%%
label = np.zeros(n_data) * np.nan
for i in range(n_data):
    p0 = stats.norm.pdf(X[i], mu, sigma)
    p1 = stats.norm.pdf(X[i], mu2, sigma2)
    
    if p0 > p1: label[i] = 0
    else: label[i] = 1

p0_ix = np.where(label==0)[0]
p1_ix = np.where(label==1)[0]

plt.figure()
plt.hist(X, bins=100, density=True)
p = stats.norm.pdf(x, np.mean(X[p0_ix]), np.std(X[p0_ix])); plt.plot(x, p)
p = stats.norm.pdf(x, np.mean(X[p1_ix]), np.std(X[p1_ix])); plt.plot(x, p)

def loss_function():
    likelihood = stats.norm.pdf(rlist, mu, sigma)
    return np.sum(likelihood)


#%%
sample0 = random.sample(list(X), 20)
sample1 = random.sample(list(X), 20)
mu = np.mean(sample0)
sigma = np.std(sample0)
pi = 0.5
mu2 = np.mean(sample1)
sigma2 = np.std(sample1)
pi2 = 0.5

plt.figure()
plt.hist(X, bins=100, density=True)
x = np.linspace(-5, 25, 1000)
p = stats.norm.pdf(x, mu, sigma); plt.plot(x, p)
p = stats.norm.pdf(x, mu2, sigma2); plt.plot(x, p)

l0 = stats.norm.pdf(X, mu, sigma)
l1 = stats.norm.pdf(X, mu2, sigma2)

pi = np.sum(l0)/n_data
l0 = l0 * pi / (l0 * pi + l1 * pi2)
l1 = l1 * pi2 / (l0 * pi + l1 * pi2)
mu0 = np.sum((X * l0)) / np.sum(l0)
std0 = np.sqrt(np.sum((X - mu0)**2 * l0)  / np.sum(l0))

pi2 = np.sum(l1)/n_data

mu1 = np.sum((X * l1)) / np.sum(l1)
std1 = np.sqrt(np.sum((X - mu1)**2 * l1)  / np.sum(l1))

plt.figure()
plt.hist(X, bins=100, density=True)
p = stats.norm.pdf(x, mu0, std0); plt.plot(x, p)
p = stats.norm.pdf(x, mu1, std1); plt.plot(x, p)

mu = mu0; sigma = std0
mu2 = mu1; sigma2 = std1

label = np.zeros(n_data) * np.nan
for i in range(n_data):
    p0 = stats.norm.pdf(X[i], mu, sigma)
    p1 = stats.norm.pdf(X[i], mu2, sigma2)
    
    if p0 > p1: label[i] = 0
    else: label[i] = 1

p0_ix = np.where(label==0)[0]
p1_ix = np.where(label==1)[0]

plt.figure()
plt.hist(X, bins=100, density=True)
p = stats.norm.pdf(x, np.mean(X[p0_ix]), np.std(X[p0_ix])); plt.plot(x, p)
p = stats.norm.pdf(x, np.mean(X[p1_ix]), np.std(X[p1_ix])); plt.plot(x, p)
















