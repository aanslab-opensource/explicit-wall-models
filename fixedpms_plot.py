import os
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import ewmlib
import pickle


plt.style.use('tableau-colorblind10')
plt.rc("text", usetex=True)
plt.rcParams.update({"font.size": 10})
plt.rcParams.update({"axes.grid" : True, "grid.color": "#DDDDDD", "grid.linestyle" : "dashed"})

laws = ['eqode','reichardt-fixedB1B2','spalding']

labels=["$l=Sp$", "$l=Rh$", "$l=Eq$"]

titles=[r"$\mathrm{High\ Reynolds\ Number}$", r"$\mathrm{Classical}$"]

fig, axs = plt.subplots(2, 1, figsize= (0.8*8.3, 4), sharex=True)

for idx_typecoeffs, typecoeffs in enumerate(['classical','highre']):
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
        
        pms_gauss = coeffs[:-2]
        p = coeffs[-2]
        s = coeffs[-1]
        
        up_caisag = ewmlib.laws.up_CaiSagaut_m2_var(rey, kappa, B, p, s)
        up_delta = ewmlib.model3(np.log10(rey), *pms_gauss)
        up_fit = up_caisag + up_delta
        
        axs[idx_typecoeffs].semilogx(rey, 100* (up - up_fit) / up, f'C{idx_law+4}', label=labels[idx_law])
        # axs[idx_typecoeffs].set_title(titles[idx_typecoeffs])

    axs[idx_typecoeffs].set_ylabel(r"$\mathbf{err}^3_{\mathbf{\Psi}_{l}}$, \%")

    if idx_typecoeffs == 0:
        axs[idx_typecoeffs].legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=3)
    else:
        axs[idx_typecoeffs].set_xlabel(r"$Re_y$")        
        
    axs[idx_typecoeffs].text(1.5e-4, 0.035, titles[idx_typecoeffs], size=10, rotation=0.,
         ha="left", va="bottom",
         bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8),))
        
    axs[idx_typecoeffs].set_xlim(1e-4, 1e+6)
    axs[idx_typecoeffs].set_ylim(-5e-2, 5e-2)

plt.tight_layout()

# plt.show()

os.makedirs("./figures", exist_ok=True)
plt.savefig("./figures/fixedpms.pdf", bbox_inches='tight', pad_inches=0.02)