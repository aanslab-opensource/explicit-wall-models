import numpy as np
from scipy.special import lambertw
from scipy.integrate import quad
from scipy.optimize import differential_evolution
import scipy.optimize as optimize


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

def opti_global_eqode(delta_model, objective_function, bounds):
    
    kappa, Aplus = np.meshgrid(
        np.linspace(0.38, 0.42, 20),
        np.linspace(  15,   19, 20),
        indexing='ij'
        )
    
    yp_REF = np.logspace(-1, 4, 500)
    
    results = np.empty_like(kappa, dtype=object)
    
    B = np.zeros_like(kappa, dtype=np.float64)
    
    yp_LOG = np.linspace(5000, 35000, 100)

    progress = 0
    
    kappa_size = kappa.size
    
    for idx, _ in np.ndenumerate(kappa):
        progress += 1
        
        B[idx] = np.mean(laws.up_ODE(yp_LOG, kappa[idx], Aplus[idx]) - (1 / kappa[idx]) * np.log(yp_LOG))
        
        up_REF = laws.up_ODE(yp_REF, kappa[idx], Aplus[idx])
        
        rey_REF = up_REF * yp_REF

        result = optimize.shgo(
            func=objective_function,
            bounds=bounds,
            args=(rey_REF, up_REF, kappa[idx], B[idx], delta_model),
            constraints=None,
            n=200,
            iters=1,
            callback=None, 
            minimizer_kwargs=None, 
            options=None, 
            sampling_method='simplicial', #'halton' 'sobol'
            workers=1
            )

        print(f'fun={result.fun:.3e}  residuals={np.round(result.x, 2)}')
        
        results[idx] = result

        print(f'{progress} / {kappa_size}', end='\r')

    return kappa, B, Aplus, results, yp_REF


def opti_global_reichardt_fixedB1B2(delta_model, objective_function, bounds):
    
    B1 = 11
    B2 = 3
    
    kappa, C = np.meshgrid(
        np.linspace(0.38, 0.40, 20), # reduced range
        np.linspace(6.80, 7.70, 20),
        indexing='ij'
        )
    
    yp_REF = np.logspace(-2, 4, 500)
    
    results = np.empty_like(kappa, dtype=object)
    
    B = np.zeros_like(kappa, dtype=np.float64)

    progress = 0
    
    kappa_size = kappa.size
    
    for idx, _ in np.ndenumerate(kappa):
        progress += 1
                    
        B[idx] = C[idx] + (np.log(kappa[idx]) / kappa[idx])
    
        up_REF  = laws.up_Reichardt(yp_REF, kappa[idx], B1, B2, C[idx])

        rey_REF = up_REF * yp_REF

        result = optimize.shgo(
            func=objective_function,
            bounds=bounds,
            args=(rey_REF, up_REF, kappa[idx], B[idx], delta_model),
            constraints=None,
            n=200,
            iters=1,
            callback=None, 
            minimizer_kwargs=None, 
            options=None, 
            sampling_method='simplicial', #'halton' 'sobol'
            workers=1
            )
        
        print(f'fun={result.fun:.3e}  residuals={np.round(result.x, 2)}')
        
        results[idx] = result
                
        print(f'{progress} / {kappa_size}', end='\r')
    
    return kappa, B, C, results, yp_REF


def opti_global_spalding(delta_model, objective_function, bounds):
    
    kappa, B = np.meshgrid(
        np.linspace(0.38, 0.42, 20),
        np.linspace(4.20, 5.50, 20), 
        indexing='ij'
        )
    
    up_REF = np.logspace(-1, np.log10(50), 500)
    
    results = np.empty_like(kappa, dtype=object)

    progress = 0
    
    kappa_size = kappa.size
    
    for idx, _ in np.ndenumerate(kappa):
        progress += 1
                
        yp_REF = laws.yp_Spalding(up_REF, kappa[idx], B[idx])
        
        rey_REF = up_REF * yp_REF

        result = optimize.shgo(
            func=objective_function,
            bounds=bounds,
            args=(rey_REF, up_REF, kappa[idx], B[idx], delta_model),
            constraints=None,
            n=200,
            iters=1,
            callback=None, 
            minimizer_kwargs=None, 
            options=None, 
            sampling_method='simplicial', #'halton' 'sobol'
            workers=1
            )
        
        print(f'fun={result.fun:.3e}  residuals={np.round(result.x, 2)}')
        
        results[idx] = result

        print(f'{progress} / {kappa_size}', end='\r')
        
    return kappa, B, results, up_REF

#######################################################################################################################

def opti_fixedpms_eqode(delta_model, objective_function, bounds, mode):
    
    if mode=="highre": # High-re constants
        kappa = 0.387
        Aplus = 15.2516
    elif mode=="classical": # Classical constants    
        kappa = 0.41
        Aplus = 17
    else:
        exit("ERROR: no constant type specified")
  
    yp_ref = np.logspace(-1, 4, 3000)

    yp_log = np.linspace(5000, 35000, 100)
    
    up_log = laws.up_ODE(yp_log, kappa, Aplus)       
    
    B = np.mean(up_log - (1 / kappa) * np.log(yp_log))
        
    up_eqode = laws.up_ODE(yp_ref, kappa, Aplus)
    
    rey_eqode = up_eqode * yp_ref
    
    results = differential_evolution(
                func=objective_function,
                strategy='best1bin',
                bounds=bounds,
                args=(rey_eqode, up_eqode, kappa, B, delta_model),
                workers=32,
                tol=1e-6,
                maxiter=2000,
                popsize=100,
                polish=True,
                updating='deferred'
                )
    
    print(results)

    return kappa, B, Aplus, results, yp_ref
    

def opti_fixedpms_reichardt_fixedB1B2(delta_model, objective_function, bounds, mode):

    if mode=="highre": # High-re constants
        kappa = 0.387
        B = 4.21
        C = B - (np.log(kappa) / kappa)
    elif mode=="classical": # Classical constants    
        kappa = 0.41
        C = 7.8
        B = C + np.log(kappa) / kappa
    else:
        exit("ERROR: no constant type specified")

    B1 = 11
    B2 = 3

    yp_ref = np.logspace(-1, 4.5, 1000)

    up_rh  = laws.up_Reichardt(yp_ref, kappa, B1, B2, C)

    rey_rh = up_rh * yp_ref

    results = differential_evolution(
        func=objective_function,
        strategy='best1bin',
        bounds=bounds,
        args=(rey_rh, up_rh, kappa, B, delta_model),
        workers=32,
        maxiter=2000,
        popsize=150,
        tol=1e-5,
        atol=1e-5,
        polish=True,
        updating='deferred'
    )
    
    print(results)
    
    return kappa, B, C, B1, B2, results, yp_ref, up_rh


def opti_fixedpms_spalding(delta_model, objective_function, bounds, mode):

    if mode=="highre": # High-re constants
        kappa = 0.387
        B = 4.21
    elif mode=="classical": # Classical constants    
        kappa = 0.4
        B = 5.5
    else:
        exit("ERROR: no constant type specified")
    
    up_ref = np.logspace(-1, np.log10(35), 1000)
    
    yp_ref = laws.yp_Spalding(up_ref, kappa, B)
    
    rey_spalding = up_ref * yp_ref

    results = differential_evolution(
        func=objective_function,
        strategy='best1bin',
        bounds=bounds,
        args=(rey_spalding, up_ref, kappa, B, delta_model),
        workers=10,
        tol=1e-6,
        maxiter=2000,
        popsize=100,
        polish=True,
        updating='deferred'
        )
    
    print(results)

    return kappa, B, results, up_ref, yp_ref
