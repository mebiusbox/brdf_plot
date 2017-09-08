# coding: utf-8
import math
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
import transforms3d
import multiprocessing as mp
import functools

## private packages
import vecmath as vm
import brdfs

def sample(brdf, roughness, theta_i):
    
    xaxis = [1,0,0]
    yaxis = [0,1,0]
    zaxis = vm.normalize(np.cross(xaxis, yaxis))
    
    q = transforms3d.quaternions.axangle2quat(yaxis, theta_i)
    ivec = transforms3d.quaternions.rotate_vector(zaxis, q)
    nvec = zaxis
    
    altitudeAngles = np.arange(0, 90, 1)
    azimuthAngles = np.arange(0, 360, 1)
    dotNL = vm.saturate(np.dot(ivec, nvec))
    
    l = math.sqrt(2-2*math.cos(np.radians(1)))
    
    # area = 0
    # E = 0

    radiosity = 0
    for altitude in altitudeAngles:
        
        theta = np.radians(altitude)
        
        # solid angle
        dx = abs(math.sin(theta)) * l
        dy = l
        omega = dx*dy
        
        q = transforms3d.quaternions.axangle2quat(yaxis, theta)
        rvec1 = transforms3d.quaternions.rotate_vector(zaxis, q)
        
        for azimuth in azimuthAngles:
            
            # theta = np.radians(altitude)
            phi = np.radians(azimuth)
            
            # q = transforms3d.quaternions.axangle2quat(yaxis, theta)
            # rvec = transforms3d.quaternions.rotate_vector(zaxis, q)
            q = transforms3d.quaternions.axangle2quat(zaxis, phi)
            rvec = transforms3d.quaternions.rotate_vector(rvec1, q)
            
            dotNV = vm.saturate(np.dot(rvec, nvec))
            hvec = vm.normalize(ivec+rvec)
            dotNH = vm.saturate(np.dot(nvec, hvec))
            dotLH = vm.saturate(np.dot(ivec, hvec))
            dotLV = vm.saturate(np.dot(ivec, rvec))
            
            # dx = abs(math.sin(theta)) * math.sqrt(2-2*math.cos(np.radians(1)))
            # dy = math.sqrt(2-2*math.cos(np.radians(1)))
            # omega = dx*dy
            
            value = brdf(dotNL, dotNV, dotNH, dotLH, dotLV, roughness)
            radiance = value * dotNV * omega
            radiosity += radiance
            
            # area += omega
            # E += dotNV * omega
    # E = pi, area = 2pi
    # print E, area
    return radiosity

if __name__ == '__main__':
    
    incidentAngles = np.arange(0, 91, 5)
    roughnessArray = np.arange(0, 1.1, 0.1)
    
    # ret = map(functools.partial(samples, theta_i=np.radians(0)), roughnessArray)
    # print ret
    
    # param = ('Lambert', brdfs.lambert_diffuse)
    # param = ('Disney Diffuse', brdfs.disney_diffuse)
    # param = ('Renormalized Disney Diffuse (by Frostbite)', brdfs.renormalized_disney_diffuse)
    # param = ('Oren Nayar (Full)', brdfs.oren_nayar_diffuse)
    # param = ('Oren Nayar (Qualitative)', brdfs.qualitative_oren_nayar_diffuse)
    # param = ('Oren Nayar (Tiny improved)', brdfs.improved_oren_nayar_diffuse)
    # param = ('Oren Nayar (Tiny fast improved)', brdfs.fast_improved_oren_nayar_diffuse)
    # param = ('Oren Nayar (Gotanda Normalized with Fresnel)', brdfs.gotanda_oren_nayar_with_fresnel_diffuse)
    # param = ('Oren Nayar (Gotanda GGX)', brdfs.gotanda_oren_nayar_diffuse)
    param = ('GGX Approximation Diffuse', brdfs.ggx_approx_diffuse)
    
    # samples(param[1], 0, 0)
    
    Z = []
    pool = mp.Pool(processes=8)
    for r in roughnessArray:
        print 'roughness: ', r
        ret = [pool.apply_async(sample, args=(param[1], r, np.radians(i))) for i in incidentAngles]
        Z.append([p.get() for p in ret])
    pool.close()
    pool.join()
    
    X, Y = np.meshgrid(incidentAngles, roughnessArray)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X,Y,Z)
    ax.set_xlabel(r'incident angle $\theta$')
    ax.set_ylabel('roughness')
    ax.set_zlabel('radiosity')
    ax.set_zlim(zmin=0)
    # ax.set_zlim(0,1.5)
    plt.title(param[0])
    plt.show()
