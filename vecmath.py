# coding: utf-8
import math
import numpy as np

def normalize(v):
    # return v / np.sqrt((v**2).sum())
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v/norm

def angle(v1, v2):
    return math.acos(np.dot(v1,v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def saturate(x):
    if x < 0:
        return 0
    elif x > 1:
        return 1
    return x

def clamp(x,a,b):
    if x < a:
        return a
    elif x > b:
        return b
    return x
    
def recip(x):
    return 1/x

def mix(a, b, t):
    return a*(1-t) + b*t

def step(edge, x):
    if x < edge:
        return 0
    else:
        return 1

def smoothstep(a, b, t):
    if a >= b:
        return 0
    else:
        x = saturate((t-a)/(b-a))
        return x*x*(3-2*t)

def sign(x):
    if x < 0.0:
        return -1
    else:
        return 1

def zerocheck(x):
    return max(x, np.finfo(float).eps)

def safe_pow(x,y):
    return pow(zerocheck(x), y)