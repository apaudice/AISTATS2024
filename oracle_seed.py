#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 18:48:57 2023
"""

import numpy as np
import math
from scipy.stats import pareto
from scipy.stats import burr12
from scipy.special import beta
        
class oracle_gauss:
    def __init__(self,a,L,var,generator):
        self.generator = generator
        self.a = a
        self.L = L
        self.std = np.sqrt(var)
        return None
    
    def query(self, x, m=1):
        g = self.L*np.sign(x-self.a)       
        noise = self.generator.normal(loc=0, scale=self.std, size=m)
        return g+np.mean(noise)
        
    def getError(self, x):
        return self.L*np.abs(x-self.a)
    
    def reset(self):
        self.index = 0

# Weibull noise (0-mean)
class oracle_weibull:
    def __init__(self,a,L,shape,scale,generator):
        self.generator = generator
        self.a = a
        self.L = L
        self.mean = scale*math.gamma(1+1/shape)        
        self.scale = scale
        self.shape = shape
        return None
    
    def query(self, x, m=1):
        g = self.L*np.sign(x-self.a)       
        noise = self.scale*self.generator.weibull(a=self.shape,size=m)-self.mean
        return g+np.mean(noise)
        
    def getError(self, x):
        return self.L*np.abs(x-self.a)
    
    def reset(self):
        self.index = 0
        
# Noise with symmetric Weibull distribution
class oracle_symm_weibull:
    def __init__(self,a,L,shape,scale,generator):
        self.generator = generator
        self.a = a
        self.L = L
        self.mean = scale*math.gamma(1+1/shape)        
        self.scale = scale
        self.shape = shape
        return None   
    
    def query(self, x, m=1):
        g = self.L*np.sign(x-self.a)
        s1 = self.scale*self.generator.weibull(a=self.shape,size=m)
        s2 = self.scale*self.generator.weibull(a=self.shape,size=m)
        noise = (s1-s2)/np.sqrt(2)
        return g+np.mean(noise)
    
    def getError(self, x):
        return self.L*np.abs(x-self.a)
    
    def reset(self):
        self.index = 0

# Noise with Pareto distribution
class oracle_pareto:
    def __init__(self,a,L,shape,scale):
        self.a = a
        self.L = L 
        self.shape, self.scale = shape, scale
        self.mu = self.shape*self.scale/(self.shape-1)
        return None
    
    def query(self, x, m=1):
        g = self.L*np.sign(x-self.a)
        s = pareto.rvs(b=self.shape, loc=0, scale=self.scale, size=m)   
        noise = s-self.mu
        return g+np.mean(noise)
    
    def getError(self, x):
        return self.L*np.abs(x-self.a)
    
    def reset(self):
        self.index = 0

# Noise with symmetric Pareto distribution
class oracle_symm_pareto:
    def __init__(self,a,L,shape,scale):
        self.a=a
        self.L=L 
        self.shape, self.scale = shape, scale
        return None
    
    def query(self, x, m=1):
        g = self.L*np.sign(x-self.a)
        s1 = pareto.rvs(b=self.shape, loc=0, scale=self.scale, size=m)
        s2 = pareto.rvs(b=self.shape, loc=0, scale=self.scale, size=m) 
        noise = (s1-s2)/np.sqrt(2)
        return g+np.mean(noise)
    
    def getError(self, x):
        return self.L*np.abs(x-self.a)
    
    def reset(self):
        self.index = 0
        
# Noise with Burr12 distribution
class oracle_burr12:
    def __init__(self,a,L,c,d):
        self.a = a
        self.L = L 
        self.c, self.d = c, d
        self.mu = d*beta(d-1/c,1+1/c)     
        return None
    
    def query(self, x, m=1):
        g = self.L*np.sign(x-self.a)
        s = burr12.rvs(c=self.c, d=self.d, loc=0, scale=1, size=m)   
        noise = s-self.mu
        return g+np.mean(noise)
    
    def getError(self, x):
        return self.L*np.abs(x-self.a)
    
    def reset(self):
        self.index = 0