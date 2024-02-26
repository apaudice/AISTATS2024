#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 09:32:05 2023
"""

# -*- coding: utf-8 -*-
import numpy as np
    
class SSgMwavg():
    def __init__(self):
        return None

    def optimize(self,oracle,w,gamma,I,dim,T,x_0,ft=False):
        r = max(I)
        self.reset(x_0)
        if (r!=-1):
            self.project(r)
        if ft==False:
            error_avg, error_li = np.zeros(T+1), np.zeros(T+1)
            error_avg[0] = oracle.getError(self.x_avg)
            error_li[0] = oracle.getError(self.x)
        else:
            error_avg, error_li = oracle.getError(self.x_avg), oracle.getError(self.x)
        
        for t in range(T):
            g = oracle.query(self.x)
            self.x = self.x - gamma[t]*g
            if (r!=-1):
                self.project(r)
            self.x_avg += w[t]*self.x
            if ft==False:
                error_avg[t+1] = oracle.getError(self.x_avg/(np.sum(w[:t+1])+1))
                error_li[t+1] = oracle.getError(self.x)
            else:
                error_avg = oracle.getError(self.x_avg/(np.sum(w[:t+1])+1))
                error_li = oracle.getError(self.x)
        oracle.reset()
        return error_avg, error_li
    
    def reset(self, x_0):
        self.x = x_0
        self.x_avg = x_0
        return None
    
    def project(self, r):
        if (np.abs(self.x)>r):
            self.x = self.x/np.abs(self.x) * r
        return None
    
class SSgMsavg():
    def __init__(self):
        return None

    def optimize(self,oracle,gamma,I,dim,T,x_0,ft=False):
        r = max(I)
        self.reset(x_0, T)
        if (r!=-1):
            self.project(r)
        if ft==False:
            error = np.zeros(T+1)
            error[0] = oracle.getError(self.x[0])
        else:
            error = oracle.getError(self.x_avg[0])
        
        for t in range(T):
            g = oracle.query(self.x[t])
            self.x[t+1] = self.x[t] - gamma[t]*g
            if (r!=-1):
                self.project(r)
            if ft==False:
                x_avg = np.mean(self.x[int(np.floor(t/2))+1:t+2])
                error[t+1] = oracle.getError(x_avg)
            else:
                x_avg = np.mean(self.x[int(t/2)+1:])
                error = oracle.getError(x_avg)
        oracle.reset()
        return error
    
    def reset(self, x_0, T):
        self.x = np.zeros(T+1)
        self.x[0] = x_0
        return None
    
    def project(self, r):
        if (np.abs(self.x)>r):
            self.x = self.x/np.abs(self.x) * r
        return None