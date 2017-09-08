# coding: utf-8
import math
import numpy as np

# private packages
import vecmath as vm

# epsilon
# https://stackoverflow.com/questions/19141432/python-numpy-machine-epsilon

def schlick(cosine, f0, f90):
    return f0 + (f90-f0) * ((1-cosine)**5)

# Lambertian
def lambert_diffuse(dotNL, dotNV, dotNH, dotLH, dotLV, roughness):
    return 1.0 / np.pi

# Disney Diffuse
# Brent Burley, "Physically-Based Shading at Disney", 2012
def disney_diffuse(dotNL, dotNV, dotNH, dotLH, dotLV, roughness):
    a = roughness*roughness
    fd90 = 0.5 + 2.0 * dotLH * dotLH * a
    nl = schlick(dotNL, 1, fd90)
    nv = schlick(dotNV, 1, fd90)
    return (nl*nv) / np.pi
    
# Re-Normalized Disney Diffuse (by Frostbite)
# Sebastien Lagarde, Charles de Rousiers, "Moving Frostbite to Physically Based Rendering 3.0", 2014
def renormalized_disney_diffuse(dotNL, dotNV, dotNH, dotLH, dotLV, roughness):
    energyBias = vm.mix(0.0, 0.5, roughness)
    energyFactor = vm.mix(1.0, 1.0/1.51, roughness)
    fd90 = energyBias + 2.0 * dotLH * dotLH * roughness
    nl = schlick(dotNL, 1, fd90)
    nv = schlick(dotNV, 1, fd90)
    return (nl*nv*energyFactor) / np.pi
    
# Oren-Nayar
# Michael Oren, Shree K. Nayar, "Generalization of Lambert's Reflection Model", 1994
def oren_nayar_diffuse(dotNL, dotNV, dotNH, dotLH, dotLV, roughness):
    theta_i = np.arccos(dotNL)
    theta_r = np.arccos(dotNV)
    sigma2 = roughness*roughness
    # cos(phi) = dot( normalize(L-dot(N,L)*N), normalize(V-(N,V)*N )
    #          = (dot(L,V) - dot(N,L)*dot(N,V)) / (sin(theta_i) * sin(theta_r))
    # cos_phi_diff = (dotLV - dotNL*dotNV) / math.sin(theta_i) * math.sin(theta_r)
    cos_phi_diff = (dotLV - dotNL*dotNV) / (math.sin(theta_i) * math.sin(theta_r) + np.finfo(float).eps)
    alpha = max(theta_i, theta_r)
    beta = min(theta_i, theta_r)
    if alpha > np.pi/2:
        return 0
    
    C1 = 1.0 - 0.5*sigma2 / (sigma2 + 0.33)
    C2 = 0.45 * sigma2 / (sigma2 + 0.09)
    if cos_phi_diff >= 0.0:
        C2 = C2*math.sin(alpha)
    else:
        C2 = C2*(math.sin(alpha) - (2*beta/np.pi)**3)
    C3 = 0.125 * sigma2 / (sigma2 + 0.09) * (((4*alpha*beta)/(np.pi*np.pi))**2)
    
    L1 = C1 + cos_phi_diff * C2 * math.tan(beta) + (1.0 - abs(cos_phi_diff)) * C3 * math.tan((alpha + beta) / 2)
    L2 = 0.17 * (sigma2 / (sigma2 + 0.13)) * (1.0 - cos_phi_diff * (4.0 * beta * beta) / (np.pi * np.pi))
    
    return (L1+L2) / np.pi

# Oren-Nayar (Qualitative)
# Michael Oren, Shree K. Nayar, "Generalization of Lambert's Reflection Model", 1994
def qualitative_oren_nayar_diffuse(dotNL, dotNV, dotNH, dotLH, dotLV, roughness):
    theta_i = np.arccos(dotNL)
    theta_r = np.arccos(dotNV)
    sigma2 = roughness*roughness
    
    alpha = max(theta_i, theta_r)
    beta = min(theta_i, theta_r)
    gamma = (dotLV - dotNL*dotNV) / (math.sin(theta_i) * math.sin(theta_r) + np.finfo(float).eps)
    
    A = 1.0 - 0.5 * (sigma2 / (sigma2 + 0.33))
    
    # Discrepancies caused by the lack of the interreflection component in the
    # qualitative model can be partially compensated by replacing the constant
    # 0.33 in coefficient A with 0.57.
    # A = 1.0 - 0.5 * (sigma2 / (sigma2 + 0.57))
    
    B = 0.45 * (sigma2 / (sigma2 + 0.09))
    C = math.sin(alpha) * math.tan(beta)
    
    L1 = (A + B*max(0, gamma)*C)
    return L1 / np.pi

# Tiny Improved Oren-Nayar
# Yasuhiro Fujii, "A tiny improvement of Oren-Nayar reflection model"
# http://mimosa-pudica.net/improved-oren-nayar.html
def improved_oren_nayar_diffuse(dotNL, dotNV, dotNH, dotLH, dotLV, roughness):
    theta_i = np.arccos(dotNL)
    theta_r = np.arccos(dotNV)
    sigma2 = roughness*roughness
    
    A = 1.0 - 0.5 * (sigma2 / (sigma2 + 0.33)) + 0.17 * sigma2 / (sigma2+0.13)
    B = 0.45 * (sigma2 / (sigma2 + 0.09))
    s = (dotLV - dotNL*dotNV)
    t = 1 if s<=0 else max(math.cos(theta_i), math.cos(theta_r))
    L1 = (A + B*s/t)
    return L1 / np.pi

# Tiny Improved Oren-Nayar (0 <= sigma <= 1)
# Yasuhiro Fujii, "A tiny improvement of Oren-Nayar reflection model"
# http://mimosa-pudica.net/improved-oren-nayar.html
def fast_improved_oren_nayar_diffuse(dotNL, dotNV, dotNH, dotLH, dotLV, roughness):
    theta_i = np.arccos(dotNL)
    theta_r = np.arccos(dotNV)
    sigma2 = roughness*roughness
    
    A = 1.0 / (np.pi + (np.pi/2 - 2/3) * sigma2)
    B = A * sigma2
    s = (dotLV - dotNL*dotNV)
    t = vm.mix(1, max(math.cos(theta_i), math.cos(theta_r)), vm.step(0,s))
    L1 = (A + B*s/t)
    return L1

# Oren Nayar (Normalized with Fresnel)
# Yoshiharu Gotanda, "Designing Reflectance Models for New Consoles", 2014
# http://research.tri-ace.com/
def gotanda_oren_nayar_with_fresnel_diffuse(dotNL, dotNV, dotNH, dotLH, dotLV, roughness):
    f0 = 0.04
    theta_i = np.arccos(dotNL)
    theta_r = np.arccos(dotNV)
    a = roughness*roughness
    
    fdiff = 1.05 * (1.0 - f0) * (1.0 - pow(1.0-dotNL, 5.0)) * (1.0 - pow(1.0 - dotNV, 5.0))
    A = 1.0 - 0.5 * (a / (a + 0.65))
    B = 0.45 * (a / (a + 0.09))
    Bp = dotLV - dotNV*dotNL
    Bm = min(1, dotNL / (dotNV + np.finfo(float).eps))
    
    # with Fresnel
    return ((1.0 - f0) * (fdiff*dotNL*A + B*Bp*Bm)) / np.pi
    
    if Bp >= 0.0:
        return (dotNL*A + B*Bp*Bm) / np.pi
    
    return (dotNL*A + B*Bp) / np.pi
    
# Oren-Nayar (GGX)
# Yoshiharu Gotanda, "Designing Reflectance Models for New Consoles", 2014
# http://research.tri-ace.com/
# http://brabl.com/diffuse-approximation-for-consoles/
def gotanda_oren_nayar_diffuse(dotNL, dotNV, dotNH, dotLH, dotLV, roughness):
    f0 = 0.04
    a = roughness * roughness
    a2 = a*a
    
    Fr1 = (1.0 - (0.542026*a2 + 0.303573*a) / (a2 + 1.36053))
    Fr2 = (1.0 - (vm.safe_pow(1.0 - dotNV, 5.0 - 4.0*a2)) / (a2 + 1.36053))
    Fr3 = (-0.733996*a2*a + 1.50912*a2 - 1.16402*a)
    Fr4 = (vm.safe_pow(1.0 - dotNV, 1.0 + (1.0 / (39.0*a2*a2 + 1.0))))
    Fr = Fr1*Fr2*(Fr3*Fr4+1)
    
    Lm1 = (max(1.0 - (2.0*a), 0.0)*(1.0 - pow(1.0 - dotNL, 5.0)) + min(2.0*a, 1.0))
    Lm2 = ((1.0 - 0.5*a)*dotNL + (0.5*a)*(pow(dotNL, 2.0)))
    Lm = Lm1 * Lm2
    
    Vd1 = (a2 / ((a2 + 0.09)*(1.31072 + 0.995584*dotNV)))
    Vd2 = (1.0 - (vm.safe_pow(1.0 - dotNL, (1.0 - 0.3726732*(dotNV*dotNV)) / (0.188566 + 0.38841*dotNV))))
    Vd = Vd1*Vd2
    
    Bp = dotLV - (dotNV*dotNL)
    if Bp < 0.0:
        Bp *= 1.4*dotNV*dotNL
    
    return (1.05/np.pi)*(1.0 - f0)*(Fr*Lm + Vd*Bp)

# GGX Diffuse Approximation
# Earl Hammon, Jr. "PBR Diffuse Lighting for GGX+Smith Microsurfaces", 2017
# http://gdcvault.com/play/1024478/PBR-Diffuse-Lighting-for-GGX
def ggx_approx_diffuse(dotNL, dotNV, dotNH, dotLH, dotLV, roughness):
    a = roughness*roughness
    
    facing = 0.5 + 0.5*dotLV
    rough  = facing * (0.9 - 0.4*facing) * ((0.5 + dotNH)/(dotNH + np.finfo(float).eps))
    smooth = 1.05 * (1 - vm.safe_pow(1-dotNL, 5)) * (1 - vm.safe_pow(1-dotNV, 5))
    single = vm.mix(smooth, rough, a) / np.pi
    multi  = 0.1159 * a
    # return albedo * (single + albedo * multi)
    return single + multi