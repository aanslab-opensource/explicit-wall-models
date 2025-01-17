from scipy import optimize
import numpy as np
import os
import ewmlib
from sklearn import linear_model
import pickle

bool_eqode = False
bool_reichardtB1B2 = True
bool_spalding = False

if __name__ == "__main__":
    
    os.makedirs("./results", exist_ok=True)
    
    if bool_eqode==True: # EqODE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        
        print('\nEqODE:')

        print('0 - optimization')
        kappa, B, Aplus, yp_ref, results, bounds = ewmlib.opti_global_eqode()

        print('1 - extract each optimized coefficient')
        fields = ewmlib.global_extract_opti_coeffs(results)

        print('2 - linear regression')
        fields_regs = {}
        for key in ['mu1','sigma1','scale1','p','s']:
            reg_obj = linear_model.LinearRegression()
            reg_obj.fit(np.column_stack((kappa.ravel(), Aplus.ravel())), fields[key].ravel())
            fields_regs.update({key:reg_obj})

        print('3 - a posteriori inputs')
        kappa_test, Aplus_test = np.meshgrid(
            np.linspace(kappa.min(), kappa.max(), 40), 
            np.linspace(Aplus.min(), Aplus.max(), 40), 
            indexing='ij'
            )

        B_test = np.zeros_like(kappa_test, dtype=np.float64)
        yp_log = np.linspace(5000, 35000, 100)
        for idx_test, _ in np.ndenumerate(kappa_test):
            B_test[idx_test] = np.mean(
                ewmlib.laws.up_ODE(yp_log, kappa_test[idx_test], Aplus_test[idx_test]) 
                - (1 / kappa_test[idx_test]) * np.log(yp_log)
                )

        print('4 - save results')
        with open("./results/global_eqode.pkl", "wb") as file:
            pickle.dump({
                'bounds':bounds,
                'kappa':kappa,
                'kappa_test':kappa_test,
                'B':B,
                'B_test':B_test,
                'Aplus':Aplus,
                'Aplus_test':Aplus_test,
                'yp_ref':yp_ref,
                'results':results,
                'fields':fields,
                'fields_regs':fields_regs,
                }, file)

    if bool_reichardtB1B2==True: # Reichardt fixed B1 B2 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        
        print('\nReichardt fixed B1 B2:')

        print('0 - optimization')
        kappa, B, C, B1, B2, yp_ref, results, bounds = ewmlib.opti_global_reichardt_fixedB1B2()

        print('1 - extract each optimized coefficient')
        fields = ewmlib.global_extract_opti_coeffs(results)

        print('2 - linear regression')
        fields_regs = {}
        for key in ['mu1','sigma1','scale1','p','s']:
            reg_obj = linear_model.LinearRegression()
            reg_obj.fit(np.column_stack((kappa.ravel(), C.ravel())), fields[key].ravel())
            fields_regs.update({key:reg_obj})

        print('3 - a posteriori inputs')
        
        kappa_test, C_test = np.meshgrid(
            np.linspace(0.38, 0.41, 40),
            np.linspace(6.70, 7.80, 40),
            indexing='ij'
            )

        B_test = C_test + (np.log(kappa_test) / kappa_test)

        print('4 - save results')
        with open("./results/global_reichardt-fixedB1B2.pkl", "wb") as file:
            pickle.dump({
                'bounds':bounds,
                'kappa':kappa,
                'kappa_test':kappa_test,
                'B':B,
                'B_test':B_test,
                'C':C,
                'C_test':C_test,
                'B1':B1,
                'B2':B2,
                'yp_ref':yp_ref,
                'results':results,
                'fields':fields,
                'fields_regs':fields_regs,
                }, file)

    if bool_spalding==True: # Spalding <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        
        print('\nSpalding:')

        print('0 - optimization')

        kappa, B, up_ref, results, bounds = ewmlib.opti_global_spalding()

        print('1 - extract each optimized coefficient')
        
        fields = {}
        for key in ['mu1','sigma1','scale1','p','s','success']: 
            fields.update({key:np.zeros_like(results)})

        for idx, _ in np.ndenumerate(kappa):
            for key_id, key in enumerate(fields):
                if key != 'success':
                    fields[key][idx] = results[idx].x[key_id]
                else:
                    fields['success'][idx] = results[idx].success

        print('2 - linear regression')

        fields_regs = {}
        for key in ['mu1','sigma1','scale1','p','s']:
            reg_obj = linear_model.LinearRegression()
            reg_obj.fit(np.column_stack((kappa.ravel(), B.ravel())), fields[key].ravel())
            fields_regs.update({key:reg_obj})

        print('3 - a posteriori inputs')
        
        kappa_test, B_test = np.meshgrid(
            np.linspace(0.38, 0.41, 40), 
            np.linspace(4.2, 5.5, 40), 
            indexing='ij'
            )

        print('4 - save results')
        with open("./results/global_spalding.pkl", "wb") as file:
            pickle.dump({
                'bounds':bounds,
                'kappa':kappa,
                'kappa_test':kappa_test,
                'B':B,
                'B_test':B_test,
                'up_ref':up_ref,
                'results':results,
                'fields':fields,
                'fields_regs':fields_regs,
                }, file)

