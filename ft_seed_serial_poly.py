#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 18:33:19 2023

@author: andreapaudice
"""

import numpy as np
import matplotlib.pyplot as plt
import time

import oracle_seed as oracle_class
import SSgMavg_class as SSgM

dim = 1
R = 1
I = (-1,-1)
a = 0
L = 1
T_list = [10,100,1000,10000] # [100,500,1000,1500,2000,2500,3000]
steps = len(T_list)

rep = 10000

# Fix the random generator
seed = 1
G = np.random.RandomState(seed)

x_init = G.uniform(-R,R,size=rep)
        
# Pareto distribution parameters
shape_p = [100, 10, 5] 
scale_p = [np.sqrt((s-2)/s)*(s-1) for s in shape_p]
# var = [(s*scale_p**2)/((s-1)**2 * (s-2)) for s in shape_p]
oracle_pareto1 = oracle_class.oracle_symm_pareto(a,L,shape_p[0],scale_p[0])
oracle_pareto2 = oracle_class.oracle_symm_pareto(a,L,shape_p[1],scale_p[1])
oracle_pareto3 = oracle_class.oracle_symm_pareto(a,L,shape_p[2],scale_p[2])

# print(var)

# Gaussian noise
std = 1
oracle_gauss = oracle_class.oracle_gauss(a,L,std,G)    
 
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
        results = ssgm.optimize(oracle_pareto1,w,gamma,I,dim,T,x_init[r],True)
        error_weibull1_avg[r,s] = results[0]
        error_weibull1_li[r,s] = results[1]

        # Weibull2 case
        results = ssgm.optimize(oracle_pareto2,w,gamma,I,dim,T,x_init[r],True)
        error_weibull2_avg[r,s] = results[0]
        error_weibull2_li[r,s] = results[1]
        
        # Weibull3 case
        results = ssgm.optimize(oracle_pareto3,w,gamma,I,dim,T,x_init[r],True)
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
# plt.plot(T_list,avg_gauss_avg,linestyle=':',marker='o',color='blue')
plt.plot(T_list,ub_weibull1_avg,linestyle=':',marker='x',color='green',
         label='Pareto ('+str(shape_p[0])+')')
# plt.plot(T_list,avg_weibull1_avg,linestyle=':',marker='o',color='green')
plt.plot(T_list,ub_weibull2_avg,linestyle=':',marker='x',color='orange',
          label='Pareto ('+str(shape_p[1])+')')
# plt.plot(T_list,avg_weibull2_avg,linestyle=':',marker='o',color='orange')
plt.plot(T_list,ub_weibull3_avg,linestyle=':',marker='x',color='red',
          label='Pareto ('+str(shape_p[2])+')')
# plt.plot(T_list,avg_weibull3_avg,linestyle=':',marker='o',color='red')
plt.xlabel('iterations')
plt.ylabel('error')
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.legend()
plt.show()
# plt.savefig('avg_pareto_ft_ss_'+str(seed)+'.pdf')

# Last iterate
plt.figure()
plt.plot(T_list,ub_gauss_li,linestyle=':',marker='x',color='blue',
         label='Gaussian')
# plt.plot(T_list,avg_gauss_li,linestyle=':',marker='o',color='blue')
plt.plot(T_list,ub_weibull1_li,linestyle=':',marker='x',color='green',
         label='Pareto ('+str(shape_p[0])+')')
# plt.plot(T_list,avg_weibull1_li,linestyle=':',marker='o',color='green')
plt.plot(T_list,ub_weibull2_li,linestyle=':',marker='x',color='orange',
         label='Pareto ('+str(shape_p[1])+')')
# plt.plot(T_list,avg_weibull2_li,linestyle=':',marker='o',color='orange')
plt.plot(T_list,ub_weibull3_li,linestyle=':',marker='x',color='red',
         label='Pareto ('+str(shape_p[2])+')')
# plt.plot(T_list,avg_weibull3_li,linestyle=':',marker='o',color='red')
plt.xlabel('iterations')
plt.ylabel('error')
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.legend()
plt.show()
# plt.savefig('li_pareto_ft_ss_'+str(seed)+'.pdf')