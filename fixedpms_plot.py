import os
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import ewmlib
import pickle
import pandas as pd

pd.set_option('display.float_format', '{:.4f}'.format)



plt.style.use('tableau-colorblind10')
plt.rc("text", usetex=True)
plt.rcParams.update({"font.size": 10})
plt.rcParams.update({"axes.grid" : True, "grid.color": "#DDDDDD", "grid.linestyle" : "dashed"})

columns = ['Model', 'mu1', 'sigma1', 'xi1', 'mu2', 'sigma2', 'xi2', 'mu3', 'sigma3', 'xi3', 'p', 's']

laws = ["spalding", "reichardt-fixedB1B2", "eqode"]
lawlabels=["$l=Sp$", "$l=Rh$", "$l=Eq$"]
csv_lawlabels = ['Spalding', 'Reichardt', 'EqODE']

titles=["$\mathrm{Classical}$", "$\mathrm{High\ Reynolds\ Number}$"]

fig, axs = plt.subplots(2, 1, figsize= (0.8*8.3, 4), sharex=True)

for idx_typecoeffs, typecoeffs in enumerate(['classical','highre']):
    df = pd.DataFrame(columns=columns)

    rows = []

    for idx_law, law in enumerate(laws):
        
        with open(f'./results/fixedpms_{law}_{typecoeffs}.pkl', 'rb') as file:
            data = pickle.load(file)
            
        file.close()
        
        kappa = data['kappa']
        B = data['B']

        match law:
            case 'eqode':
                yp = np.logspace(-4, 6, 1000)
                up = ewmlib.laws.up_ODE(yp, kappa, data['Aplus'])
            case 'reichardt-fixedB1B2':
                yp = np.logspace(-4, 6, 1000)
                up = ewmlib.laws.up_Reichardt(yp, kappa, 11, 3, data['C'])
            case 'spalding':
                up = np.linspace(1e-2, 35, 1000)
                yp = ewmlib.laws.yp_Spalding(up, kappa, B)
                
        rey = yp*up
        
        coeffs = data['results'].x
        
        rows.append([csv_lawlabels[idx_law]] + list(coeffs))
        
        pms_gauss = coeffs[:-2]
        p = coeffs[-2]
        s = coeffs[-1]
        
        up_caisag = ewmlib.laws.up_CaiSagaut_m2_var(rey, kappa, B, p, s)
        up_delta = ewmlib.model3(np.log10(rey), *pms_gauss)
        up_fit = up_caisag + up_delta
        
        axs[idx_typecoeffs].semilogx(rey, 100* (up - up_fit) / up, f'C{idx_law+4}', label=lawlabels[idx_law])

    for row in rows:
        df.loc[len(df)] = row

    print(f'\n{csv_lawlabels[idx_law]}')
    print(df)

    os.makedirs("./tables", exist_ok=True)

    df.to_csv(f'./tables/fixedpms_{typecoeffs}.csv', index=False)

    axs[idx_typecoeffs].set_ylabel(r"$\mathbf{err}^3_{\mathbf{\Psi}_{l}}$, \%")

    if idx_typecoeffs == 0:
        axs[idx_typecoeffs].legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=3)
    else:
        axs[idx_typecoeffs].set_xlabel(r"$Re_y$")        
        
    axs[idx_typecoeffs].text(1.5e-4, 0.026, titles[idx_typecoeffs], size=10, rotation=0.,
         ha="left", va="bottom",
         bbox=dict(boxstyle="round", ec='grey', fc='lightgrey',))
        
    axs[idx_typecoeffs].set_xlim(1e-4, 1e+6)
    axs[idx_typecoeffs].set_ylim(-0.04, 0.04)

plt.tight_layout()

os.makedirs("./figures", exist_ok=True)

plt.savefig("./figures/fixedpms.pdf", bbox_inches='tight', pad_inches=0.02)









