#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 12:22:05 2023
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import time

import oracle_seed as oracle_class
import SSgMavg_class as SSgM

dim = 1
R = 1
I = (-1,-1)
a = 0
L = 1
T = 3000

rep = 10000

# Fix the random generator
seed = 1
G = np.random.RandomState(1)

x_init = G.uniform(-R,R,size=rep)

# Gaussian noise
std = 1
oracle_gauss = oracle_class.oracle_gauss(a,L,std,G)    
        
# Weibull distribution parameters
shape = [1, 0.5, 0.3]
scale = [np.sqrt(1/(math.gamma(1+2/s)-math.gamma(1+1/s)**2)) for s in shape]
oracle_weibull1 = oracle_class.oracle_symm_weibull(a,L,shape[0],scale[0],G)
oracle_weibull2 = oracle_class.oracle_symm_weibull(a,L,shape[1],scale[1],G)
oracle_weibull3 = oracle_class.oracle_symm_weibull(a,L,shape[2],scale[2],G)
 
error_gauss_avg = np.zeros((rep,T+1))
error_weibull1_avg = np.zeros((rep,T+1))
error_weibull2_avg = np.zeros((rep,T+1))
error_weibull3_avg = np.zeros((rep,T+1))

error_gauss_li = np.zeros((rep,T+1))
error_weibull1_li = np.zeros((rep,T+1))
error_weibull2_li = np.zeros((rep,T+1))
error_weibull3_li = np.zeros((rep,T+1))

gamma = np.zeros(T)
w = np.zeros(T)
for i in range(T):
    gamma[i] = 1/np.sqrt(i+1)
    w[i] = 1
ssgm = SSgM.SSgMwavg()

st = time.perf_counter()

for r in range(rep):
    if(r%10==0):
        print(r)
    results = ssgm.optimize(oracle_gauss,w,gamma,I,dim,T,x_init[r])
    error_gauss_avg[r,:] = results[0] 
    error_gauss_li[r,:] = results[1]
    
    results = ssgm.optimize(oracle_weibull1,w,gamma,I,dim,T,x_init[r])
    error_weibull1_avg[r,:] = results[0] 
    error_weibull1_li[r,:] = results[1]
    
    results = ssgm.optimize(oracle_weibull2,w,gamma,I,dim,T,x_init[r])
    error_weibull2_avg[r,:] = results[0]
    error_weibull2_li[r,:] = results[1]
    
    results = ssgm.optimize(oracle_weibull3,w,gamma,I,dim,T,x_init[r])
    error_weibull3_avg[r,:] = results[0]
    error_weibull3_li[r,:] = results[1]
    
end = time.perf_counter()
print('Exectution time %.3f [seconds].'%(end-st))

alpha = 1

avg_gauss_avg = np.mean(error_gauss_avg,axis=0)
ub_gauss_avg = np.percentile(error_gauss_avg,100-alpha,axis=0)

avg_weibull1_avg = np.mean(error_weibull1_avg,axis=0)
ub_weibull1_avg = np.percentile(error_weibull1_avg,100-alpha,axis=0)

avg_weibull2_avg = np.mean(error_weibull2_avg,axis=0)
ub_weibull2_avg = np.percentile(error_weibull2_avg,100-alpha,axis=0)

avg_weibull3_avg = np.mean(error_weibull3_avg,axis=0)
ub_weibull3_avg = np.percentile(error_weibull3_avg,100-alpha,axis=0)

avg_gauss_li = np.mean(error_gauss_li,axis=0)
ub_gauss_li = np.percentile(error_gauss_li,100-alpha,axis=0)

avg_weibull1_li = np.mean(error_weibull1_li,axis=0)
ub_weibull1_li = np.percentile(error_weibull1_li,100-alpha,axis=0)

avg_weibull2_li = np.mean(error_weibull2_li,axis=0)
ub_weibull2_li = np.percentile(error_weibull2_li,100-alpha,axis=0)

avg_weibull3_li = np.mean(error_weibull3_li,axis=0)
ub_weibull3_li = np.percentile(error_weibull3_li,100-alpha,axis=0)

# Save results
np.savez('at_experiment_ss_'+str(seed)+'.npz',
          error_gauss_avg,error_gauss_li,
          error_weibull1_avg,error_weibull1_li,
          error_weibull2_avg,error_weibull2_li,
          error_weibull3_avg,error_weibull3_li)

# Plot results
# Average
plt.figure()
plt.plot(range(T+1),ub_gauss_avg,linestyle=':',color='blue',
         label='Gaussian')
plt.fill_between(range(T+1),avg_gauss_avg,ub_gauss_avg,alpha=0.1,
                 color='blue')
plt.plot(range(T+1),ub_weibull1_avg,linestyle=':',color='green',
         label='Exponential')
plt.fill_between(range(T+1),avg_weibull1_avg,ub_weibull1_avg,alpha=0.1,
                 color='green')
plt.plot(range(T+1),ub_weibull2_avg,linestyle=':',color='orange',
          label='Weibull ('+str(shape[1])+')')
plt.fill_between(range(T+1),avg_weibull2_avg,ub_weibull2_avg,alpha=0.1,
                 color='orange')
plt.plot(range(T+1),ub_weibull3_avg,linestyle=':',color='red',
         label='Weibull ('+str(shape[2])+').')
plt.fill_between(range(T+1),avg_weibull3_avg,ub_weibull3_avg,alpha=0.1,
                 color='red')
plt.xlabel('iterations')
plt.ylabel('error')
plt.grid()
plt.legend()

# Zoom
a = plt.axes([0.2, 0.6, .2, .2])
plt.plot(range(2000,T+1),ub_gauss_avg[2000:],linestyle=':',color='blue')
plt.fill_between(range(2000,T+1),avg_gauss_avg[2000:],ub_gauss_avg[2000:],alpha=0.1,
                  color='blue')
plt.plot(range(2000,T+1),ub_weibull1_avg[2000:],linestyle=':',color='green')
plt.fill_between(range(2000,T+1),avg_weibull1_avg[2000:],ub_weibull1_avg[2000:],alpha=0.1,
                  color='green')
plt.plot(range(2000,T+1),ub_weibull2_avg[2000:],linestyle=':',color='orange')
plt.fill_between(range(2000,T+1),avg_weibull2_avg[2000:],ub_weibull2_avg[2000:],alpha=0.1,
                  color='orange')
plt.plot(range(2000,T+1),ub_weibull3_avg[2000:],linestyle=':',color='red')
plt.fill_between(range(2000,T+1),avg_weibull3_avg[2000:],ub_weibull3_avg[2000:],alpha=0.1,
                  color='red')
plt.title('Last 1000 iterates.')
plt.xticks([])

plt.show()
# plt.savefig('avg_weibull_at.pdf')

# Last iterate
plt.figure()
plt.plot(range(T+1),ub_gauss_li,linestyle=':',color='blue',
         label='Gaussian')
plt.fill_between(range(T+1),avg_gauss_li,ub_gauss_li,alpha=0.1,
                 color='blue')
plt.plot(range(T+1),ub_weibull1_li,linestyle=':',color='green',
         label='Exponential')
plt.fill_between(range(T+1),avg_weibull1_li,ub_weibull1_li,
                 alpha=0.1,color='green')
plt.plot(range(T+1),ub_weibull2_li,linestyle=':',color='orange',
          label='Weibull ('+str(shape[1])+')')
plt.fill_between(range(T+1),avg_weibull2_li,ub_weibull2_li,
                 alpha=0.1,color='orange')
plt.plot(range(T+1),ub_weibull3_li,linestyle=':',color='red',
         label='Weibull ('+str(shape[2])+')')
plt.fill_between(range(T+1),avg_weibull3_li,ub_weibull3_li,
                 alpha=0.1,color='red')
plt.xlabel('iterations')
plt.ylabel('error')
plt.grid()
plt.legend()

# Zoom
a = plt.axes([0.2, 0.6, .2, .2])
plt.plot(range(2000,T+1),ub_gauss_li[2000:],linestyle=':',color='blue')
plt.fill_between(range(2000,T+1),avg_gauss_li[2000:],ub_gauss_li[2000:],alpha=0.1,
                  color='blue')
plt.plot(range(2000,T+1),ub_weibull1_li[2000:],linestyle=':',color='green')
plt.fill_between(range(2000,T+1),avg_weibull1_li[2000:],ub_weibull1_li[2000:],alpha=0.1,
                  color='green')
plt.plot(range(2000,T+1),ub_weibull2_li[2000:],linestyle=':',color='orange')
plt.fill_between(range(2000,T+1),avg_weibull2_li[2000:],ub_weibull2_li[2000:],alpha=0.1,
                  color='orange')
plt.plot(range(2000,T+1),ub_weibull3_li[2000:],linestyle=':',color='red')
plt.fill_between(range(2000,T+1),avg_weibull3_li[2000:],ub_weibull3_li[2000:],alpha=0.1,
                  color='red')
plt.title('Last 1000 iterates.')
plt.xticks([])
plt.show()

plt.show()
# plt.savefig('li_weibull_at.pdf')