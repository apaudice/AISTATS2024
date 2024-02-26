#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 15:56:27 2023
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
T_list = [100,500,1000,1500,2000,2500,3000]
steps = len(T_list)

rep = 10000

# Fix the random generator
seed = 2
G = np.random.RandomState(seed)

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
 
error_gauss_avg = np.zeros((rep,steps))
error_weibull1_avg = np.zeros((rep,steps))
error_weibull2_avg = np.zeros((rep,steps))
error_weibull3_avg = np.zeros((rep,steps))

error_gauss_li = np.zeros((rep,steps))
error_weibull1_li = np.zeros((rep,steps))
error_weibull2_li = np.zeros((rep,steps))
error_weibull3_li = np.zeros((rep,steps))
for s in range(steps):
    print('#Iterations: %d.'%(T_list[s]))
    T = T_list[s]
    gamma = np.zeros(T)
    w = np.zeros(T)

    for i in range(T):
        gamma[i] = 1/np.sqrt(T)
        w[i] = 1
    ssgm = SSgM.SSgMwavg()
    
    st = time.perf_counter()
    for r in range(rep):
        # Gaussian case
        results = ssgm.optimize(oracle_gauss,w,gamma,I,dim,T,x_init[r],True)
        error_gauss_avg[r,s] = results[0]
        error_gauss_li[r,s] = results[1]
        
        # Weibull1 case
        results = ssgm.optimize(oracle_weibull1,w,gamma,I,dim,T,x_init[r],True)
        error_weibull1_avg[r,s] = results[0]
        error_weibull1_li[r,s] = results[1]

        # Weibull2 case
        results = ssgm.optimize(oracle_weibull2,w,gamma,I,dim,T,x_init[r],True)
        error_weibull2_avg[r,s] = results[0]
        error_weibull2_li[r,s] = results[1]
        
        # Weibull3 case
        results = ssgm.optimize(oracle_weibull3,w,gamma,I,dim,T,x_init[r],True)
        error_weibull3_avg[r,s] = results[0]
        error_weibull3_li[r,s] = results[1]
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

# Save the data
# np.savez('ft_experiment_ss_'+str(seed)+'.npz',
#           error_gauss_avg,error_gauss_li,
#           error_weibull1_avg,error_weibull1_li,
#           error_weibull2_avg,error_weibull2_li,
#           error_weibull3_avg,error_weibull3_li)
    
# Plot results
# Average
plt.figure()
plt.plot(T_list,ub_gauss_avg,linestyle=':',marker='x',color='blue',
         label='Gaussian')
plt.plot(T_list,avg_gauss_avg,linestyle=':',marker='o',color='blue')
plt.plot(T_list,ub_weibull1_avg,linestyle=':',marker='x',color='green',
         label='Exponential')
plt.plot(T_list,avg_weibull1_avg,linestyle=':',marker='o',color='green')
plt.plot(T_list,ub_weibull2_avg,linestyle=':',marker='x',color='orange',
          label='Weibull (0.5)')
plt.plot(T_list,avg_weibull2_avg,linestyle=':',marker='o',color='orange')
plt.plot(T_list,ub_weibull3_avg,linestyle=':',marker='x',color='red',
         label='Weibul (0.3)')
plt.plot(T_list,avg_weibull3_avg,linestyle=':',marker='o',color='red')
plt.xlabel('iterations')
plt.ylabel('error')
plt.grid()
plt.legend()
plt.show()
# plt.savefig('avg_weibull_ft_ss_'+str(seed)+'.pdf')

# Last iterate
plt.figure()
plt.plot(T_list,ub_gauss_li,linestyle=':',marker='x',color='blue',
         label='Gaussian')
plt.plot(T_list,avg_gauss_li,linestyle=':',marker='o',color='blue')
plt.plot(T_list,ub_weibull1_li,linestyle=':',marker='x',color='green',
         label='Exponential')
plt.plot(T_list,avg_weibull1_li,linestyle=':',marker='o',color='green')
plt.plot(T_list,ub_weibull2_li,linestyle=':',marker='x',color='orange',
          label='Weibull (0.5)')
plt.plot(T_list,avg_weibull2_li,linestyle=':',marker='o',color='orange')
plt.plot(T_list,ub_weibull3_li,linestyle=':',marker='x',color='red',
         label='Weibul (0.3)')
plt.plot(T_list,avg_weibull3_li,linestyle=':',marker='o',color='red')
plt.xlabel('iterations')
plt.ylabel('error')
plt.grid()
plt.legend()
plt.show()
# plt.savefig('li_weibull_ft_ss_'+str(seed)+'.pdf')