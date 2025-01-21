import numpy as np
from scipy.special import lambertw
from scipy.integrate import quad
import scipy.optimize as optimize
from tqdm import tqdm
import ewmlib



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


#######################################################################################################################


def global_extract_opti_coeffs(results):

    fields = {}

    for key in ['mu1','sigma1','scale1','p','s','success']: 
        fields.update({key:np.zeros_like(results)})

    for key_idx, key in enumerate(fields):
        for idx, _ in np.ndenumerate(results):
            if key != 'success':
                fields[key][idx] = results[idx].x[key_idx]
            else:
                fields['success'][idx] = results[idx].success

    return fields


#######################################################################################################################


def opti_global_eqode():
    
    bounds = [
        (2.43,  3.13),  # mu1
        (0.48,  0.66),  # sigma1
        (0.018, 0.064), # scale1
        (1.22,  1.26),  # p
        (76,    114)    # s
        ]

    kappa, Aplus = np.meshgrid(
        np.linspace(0.38, 0.42, 20),
        np.linspace(  15,   19, 20),
        indexing='ij'
        )

    yp_ref = np.logspace(-1, 4, 500)

    results = np.empty_like(kappa, dtype=object)

    B = np.zeros_like(kappa, dtype=np.float64)

    yp_LOG = np.linspace(5000, 35000, 100)
                
    for idx, _ in tqdm(np.ndenumerate(kappa), total=kappa.size):
        
        B[idx] = np.mean(laws.up_ODE(yp_LOG, kappa[idx], Aplus[idx]) - (1 / kappa[idx]) * np.log(yp_LOG))
        
        up_REF = laws.up_ODE(yp_ref, kappa[idx], Aplus[idx])
        
        rey_REF = up_REF * yp_ref

        result = optimize.shgo(
            func=ewmlib.objfun_pwrel,
            bounds=bounds,
            args=(rey_REF, up_REF, kappa[idx], B[idx], ewmlib.model1),
            constraints=None,
            n=200,
            iters=1,
            callback=None, 
            minimizer_kwargs=None, 
            options=None, 
            sampling_method='simplicial', #'halton' 'sobol'
            workers=1
            )
        
        results[idx] = result

    return kappa, B, Aplus, yp_ref, results, bounds


def opti_global_reichardt_fixedB1B2():

    bounds = [
        (2,     2.4),  # mu1
        (0.76,  0.83), # sigma1
        (0.076, 0.13), # scale1
        (1.24,  1.25), # p
        (112,   133)   # s
        ]

    B1 = 11
    B2 = 3
    
    kappa, C = np.meshgrid(
        np.linspace(0.38, 0.40, 20), # reduced range
        np.linspace(6.80, 7.70, 20),
        indexing='ij'
        )
    
    yp_ref = np.logspace(-2, 4, 500)
    
    results = np.empty_like(kappa, dtype=object)
    
    B = np.zeros_like(kappa, dtype=np.float64)
        
    for idx, _ in tqdm(np.ndenumerate(kappa), total=kappa.size):
                    
        B[idx] = C[idx] + (np.log(kappa[idx]) / kappa[idx])
    
        up_REF  = laws.up_Reichardt(yp_ref, kappa[idx], B1, B2, C[idx])

        rey_REF = up_REF * yp_ref

        result = optimize.shgo(
            func=ewmlib.objfun_pwrel,
            bounds=bounds,
            args=(rey_REF, up_REF, kappa[idx], B[idx], ewmlib.model1),
            constraints=None,
            n=200,
            iters=1,
            callback=None, 
            minimizer_kwargs=None, 
            options=None, 
            sampling_method='simplicial', #'halton' 'sobol'
            workers=1
            )
        
        results[idx] = result
    
    return kappa, B, C, B1, B2, yp_ref, results, bounds


def opti_global_spalding():

    bounds = [
        (1.91,  2.08),  # mu1
        (1.2,   1.33),  # sigma1
        (0.195, 0.275), # scale1
        (1.25,  1.29),  # p
        (217,   378)    # s
        ]

    kappa, B = np.meshgrid(
        np.linspace(0.38, 0.42, 20),
        np.linspace(4.20, 5.50, 20), 
        indexing='ij'
        )
    
    up_ref = np.logspace(-1, np.log10(50), 500)
    
    results = np.empty_like(kappa, dtype=object)    
    
    for idx, _ in tqdm(np.ndenumerate(kappa), total=kappa.size):
                
        yp_REF = laws.yp_Spalding(up_ref, kappa[idx], B[idx])
        
        rey_REF = up_ref * yp_REF

        result = optimize.shgo(
            func=ewmlib.objfun_pwrel,
            bounds=bounds,
            args=(rey_REF, up_ref, kappa[idx], B[idx], ewmlib.model1),
            constraints=None,
            n=200,
            iters=1,
            callback=None, 
            minimizer_kwargs=None, 
            options=None, 
            sampling_method='simplicial', #'halton' 'sobol'
            workers=1
            )
        
        results[idx] = result
        
    return kappa, B, up_ref, results, bounds


#######################################################################################################################

shgokwargs = {
    "constraints":None,
    "n":200,
    "iters":1,
    "callback":None, 
    "minimizer_kwargs":None, 
    "options":None, 
    "sampling_method":'simplicial', # 'halton' 'sobol'
    "workers":1
}

def opti_fixedpms_eqode(mode):

    if mode=="classical": # Classical constants    
        kappa = 0.41
        Aplus = 17
        bounds = [
            (2.5, 3.0),     # mu1
            (1.1, 1.2),     # sigma1
            (0.04, 0.06),   # scale1
            (2.5, 3.0),     # mu2
            (3.0, 3.5),     # sigma2
            (0.1, 0.2),     # scale2
            (2.5, 2.6),     # mu3
            (0.5, 0.7),     # sigma3
            (0.01, 0.02),   # scale3
            (1.0, 1.5),     # p
            (240.0, 250.0), # s
        ]
    elif mode=="highre": # High-re constants
        kappa = 0.387
        Aplus = 15.2516
        bounds = [
            (2.5, 3.0),     # mu1
            (2.5, 3.0),     # sigma1
            (0.1, 0.2),     # scale1
            (3.0, 3.5),     # mu2
            (0.5, 0.6),     # sigma2
            (0.05, 0.06),   # scale2
            (4.0, 4.2),     # mu3
            (0.9, 1.0),     # sigma3
            (-0.1, 0.0),    # scale3
            (1.1, 1.2),     # p
            (210.0, 220.0), # s
        ]
    else:
        exit("ERROR: no constant type specified")
  
    yp_ref = np.logspace(-1, 4, 3000)

    yp_log = np.linspace(5000, 35000, 100)
    
    up_log = laws.up_ODE(yp_log, kappa, Aplus)       
    
    B = np.mean(up_log - (1 / kappa) * np.log(yp_log))
        
    up_eqode = laws.up_ODE(yp_ref, kappa, Aplus)
    
    rey_eqode = up_eqode * yp_ref
        
    results = optimize.shgo(
        func=ewmlib.objfun_pwrel,
        bounds=bounds,
        args=(rey_eqode, up_eqode, kappa, B, ewmlib.model3),
        constraints=shgokwargs['constraints'],
        n=shgokwargs['n'],
        iters=shgokwargs['iters'],
        workers=shgokwargs['workers'],
        callback=shgokwargs['callback'], 
        minimizer_kwargs=shgokwargs['minimizer_kwargs'], 
        options=shgokwargs['options'], 
        sampling_method=shgokwargs['sampling_method']
        )
    
    print(results)

    return kappa, B, Aplus, yp_ref, results
    

def opti_fixedpms_reichardt_fixedB1B2(mode):
    
    if mode=="classical": # Classical constants    
        kappa = 0.41
        C = 7.8
        B = C + np.log(kappa) / kappa
        bounds = [
            (2.0, 2.5),  # mu1
            (0.5, 1.0),  # sigma1
            (-0.1, 0.1), # scale1
            (2.0, 2.5),  # mu2
            (1.5, 2.0),  # sigma2
            (-0.1, 0.1), # scale2
            (3.5, 4.0),  # mu3
            (1.5, 2.0),  # sigma3
            (-0.1, 0.1), # scale3
            (1.0, 1.5),  # p
            (100, 150),  # s
            ]
    elif mode=="highre": # High-re constants
        kappa = 0.387
        B = 4.21
        C = B - (np.log(kappa) / kappa)
        bounds = [
            (1.0, 1.5),   # mu1
            (0.0, 1.0),   # sigma1
            (-0.05, 0.0), # scale1
            (2.0, 2.5),   # mu2
            (2.0, 2.5),   # sigma2
            (0.1, 0.2),   # scale2
            (2.0, 2.5),   # mu3
            (0.5, 1.0),   # sigma3
            (0.05, 0.1),  # scale3
            (1.0, 1.5),   # p
            (100, 150),   # s
            ]
    else:
        exit("ERROR: no constant type specified")

    B1 = 11
    B2 = 3

    yp_ref = np.logspace(-1, 4.5, 1000)

    up_ref  = laws.up_Reichardt(yp_ref, kappa, B1, B2, C)

    rey_ref = up_ref * yp_ref
    
    results = optimize.shgo(
        func=ewmlib.objfun_pwrel,
        bounds=bounds,
        args=(rey_ref, up_ref, kappa, B, ewmlib.model3),
        constraints=shgokwargs['constraints'],
        n=shgokwargs['n'],
        iters=shgokwargs['iters'],
        callback=shgokwargs['callback'], 
        minimizer_kwargs=shgokwargs['minimizer_kwargs'], 
        options=shgokwargs['options'], 
        sampling_method=shgokwargs['sampling_method'],
        workers=shgokwargs['workers']
        )
    
    print(results)
    
    return kappa, B, C, B1, B2, yp_ref, results


def opti_fixedpms_spalding(mode):

    if mode=="classical": # Classical constants    
        kappa = 0.4
        B = 5.5
        bounds = [
            (3.0, 3.1),   # mu1
            (2.8, 2.9),   # sigma1
            (0.28, 0.3),  # scale1
            (2.5, 2.6),   # mu2
            (0.9, 1.0),   # sigma2
            (-0.2, -0.1), # scale2
            (2.6, 2.7),   # mu3
            (1.8, 1.9),   # sigma3
            (-1.3, -1.2), # scale3
            (1.14, 1.15), # p
            (350, 400),   # s
            ]
    elif mode=="highre": # High-re constants
        kappa = 0.387
        B = 4.21
        bounds = [
            (3.0, 3.2),   # mu1
            (2.3, 2.4),   # sigma1
            (-0.1, 0.0),  # scale1
            (2.3, 2.4),   # mu2
            (2.2, 2.3),   # sigma2
            (-0.4, -0.3), # scale2
            (2.9, 3.0),   # mu3
            (0.5, 0.6),   # sigma3
            (0.02, 0.03), # scale3
            (1.0, 1.5),   # p
            (200, 210),   # s
            ]
    else:
        exit("ERROR: no constant type specified")
    
    up_ref = np.logspace(-1, np.log10(35), 1000)
    
    yp_ref = laws.yp_Spalding(up_ref, kappa, B)
    
    rey_spalding = up_ref * yp_ref

    results = optimize.shgo(
        func=ewmlib.objfun_pwrel,
        bounds=bounds,
        args=(rey_spalding, up_ref, kappa, B, ewmlib.model3),
        constraints=shgokwargs['constraints'],
        n=shgokwargs['n'],
        iters=shgokwargs['iters'],
        callback=shgokwargs['callback'], 
        minimizer_kwargs=shgokwargs['minimizer_kwargs'], 
        options=shgokwargs['options'], 
        sampling_method=shgokwargs['sampling_method'],
        workers=shgokwargs['workers']
        )
    
    print(results)

    return kappa, B, up_ref, results

