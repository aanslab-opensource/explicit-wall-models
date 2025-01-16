import numpy as np
from scipy.special import lambertw
from scipy.integrate import quad


def objfun_pwrel(params, rey, up_ref, kappa, B, delta_model):
    up_cs = laws.up_CaiSagaut_m2_var(rey, kappa, B, p=params[-2], s=params[-1])
    pwrelerr = (up_ref - (delta_model(np.log10(rey), *params[:-2]) + up_cs)) / up_cs
    return np.sum(pwrelerr ** 2)

def model1(log10rey, mu, sigma, scale):
    return scale*np.exp(-np.square((log10rey - mu)*sigma))

def model3(log10rey, mu1, sigma1, scale1, mu2, sigma2, scale2, mu3, sigma3, scale3,):
    return (scale1 * np.exp( -np.square( (log10rey - mu1) * sigma1) ) + 
            scale2 * np.exp( -np.square( (log10rey - mu2) * sigma2) ) +
            scale3 * np.exp( -np.square( (log10rey - mu3) * sigma3) ) )

class laws:
    @staticmethod
    def up_laminar(yp):
        return yp

    @staticmethod
    def up_loglaw(yp, kappa, B):
        return (1 / kappa) * np.log(yp) + B

    @staticmethod
    def yp_Spalding(up, kappa, B):
        kup = kappa * up
        return up + np.exp(-kappa*B) * (np.exp(kup) - 1 - kup - np.square(kup)/2 - np.power(kup, 3)/6)

    @staticmethod
    def up_ODE(yp, kappa, Aplus):
        nutp = lambda yp, kappa, Aplus: kappa*yp*(1 - np.exp(-yp/Aplus))**2
        integrand = lambda yp, kappa, Aplus: 1/(1 + nutp(yp, kappa, Aplus))
        return np.array([quad(integrand, 0, val, args=(kappa, Aplus))[0] for val in yp])

    @staticmethod
    def up_Reichardt(yp, kappa, B1, B2, C):
        return (1 / kappa)*np.log(1 + kappa*yp) + C*(1 - np.exp(-yp/B1) - (yp/B1)*np.exp(-yp/B2))

    @staticmethod
    def up_CaiSagaut_m2(Rey, kappa, E, p, s):
        tmp = np.real(lambertw(Rey * kappa * E, k=0, tol=1e-12))
        return np.power(np.exp(-Rey / s), p) * np.sqrt(Rey) + np.power(1 - np.exp(-Rey / s), p) * tmp / kappa

    @staticmethod
    def up_CaiSagaut_m2_var(Rey, kappa, B, p, s):
        return laws.up_CaiSagaut_m2(Rey, kappa, np.exp(kappa*B), p, s)
