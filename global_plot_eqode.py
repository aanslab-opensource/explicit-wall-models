import os
import numpy as np
import matplotlib.pyplot as plt
import ewmlib
import matplotlib as mpl
import matplotlib.colors as mcolors
from sklearn import linear_model
import pickle


print('0 - load data')

with open('./results/global_eqode.pkl', 'rb') as file:
    data = pickle.load(file)

file.close()

kappa = data['kappa']
Aplus = data['Aplus']
B = data['B']
kappa_test = data['kappa_test']
Aplus_test = data['Aplus_test']
B_test = data['B_test']
fields = data['fields']
fields_regs = data['fields_regs']
results = data['results']




p_reg      = fields_regs['p'].predict(     np.column_stack((kappa_test.ravel(), Aplus_test.ravel()))).reshape(kappa_test.shape)
s_reg      = fields_regs['s'].predict(     np.column_stack((kappa_test.ravel(), Aplus_test.ravel()))).reshape(kappa_test.shape)
mu1_reg    = fields_regs['mu1'].predict(   np.column_stack((kappa_test.ravel(), Aplus_test.ravel()))).reshape(kappa_test.shape)
sigma1_reg = fields_regs['sigma1'].predict(np.column_stack((kappa_test.ravel(), Aplus_test.ravel()))).reshape(kappa_test.shape)
scale1_reg = fields_regs['scale1'].predict(np.column_stack((kappa_test.ravel(), Aplus_test.ravel()))).reshape(kappa_test.shape)

##############################################################################################################################

# mpl.rcParams['lines.linewidth'] = 0.5
plt.style.use('tableau-colorblind10')
plt.rc("text", usetex=True)
plt.rcParams.update({"font.size": 10})
plt.rcParams.update({"axes.grid":True, "grid.color":"#DDDDDD", "grid.linestyle":"dashed"})

##############################################################################################################################

print('1 - plot data')

fig, ax = plt.subplots(1, 1, figsize=(0.8*8.3, 2.5))

yp_ref = np.logspace(-4, 5, 500)


print('   1.0 - optimal grid')
# error on optimal points
error_values = []
for idx, _ in np.ndenumerate(kappa):
    up_eqode = ewmlib.laws.up_ODE(yp_ref, kappa[idx], Aplus[idx])
    rey_eqode = up_eqode * yp_ref
    up_caisag = ewmlib.laws.up_CaiSagaut_m2_var(rey_eqode, kappa[idx], B[idx], fields['p'][idx], fields['s'][idx])
    up_delta = ewmlib.model1(np.log10(rey_eqode), *results[idx].x[:-2])
    error_values.append(100 * (up_eqode - (up_caisag + up_delta)) / up_eqode)
error_values = np.vstack(error_values)
e_min = error_values.min(axis=0)
e_max = error_values.max(axis=0)
ax.fill_between(rey_eqode, e_min, e_max, color='k', alpha=1, label=r'$\mathbf{\tilde{\Pi}}_1(\mathbf{\Psi}_{Eq})$')


print('   1.1 - test grid')
# error on test grid
error_interp = []
for idx, _ in np.ndenumerate(kappa_test):
    up_eqode = ewmlib.laws.up_ODE(yp_ref, kappa_test[idx], Aplus_test[idx])    
    rey_eqode = up_eqode * yp_ref
    up_caisag_reg = ewmlib.laws.up_CaiSagaut_m2_var(rey_eqode, kappa_test[idx], B_test[idx], p_reg[idx], s_reg[idx])
    up_delta_reg = ewmlib.model1(np.log10(rey_eqode), mu1_reg[idx], sigma1_reg[idx], scale1_reg[idx])
    error_interp.append(100 * (up_eqode - (up_caisag_reg + up_delta_reg)) / up_eqode)
error_interp = np.vstack(error_interp)
e_min = error_interp.min(axis=0)
e_max = error_interp.max(axis=0)
ax.fill_between(rey_eqode, e_min, e_max, color='C4', alpha=0.75, label=r'$\tilde{\pi}_{Eq, 1}(\mathbf{\Psi}_{Eq})$')


ax.set_xscale("log")

ax.set_xlabel(r"$Re_y$")
ax.set_ylabel(r"$\mathbf{err}^1_{\mathbf{\Psi}_{Eq}}$, \%")

ax.set_xlim(1e-4, 1e+6)
ax.set_ylim(-0.4, 0.4)

ax.legend()

plt.tight_layout()
# plt.show()

os.makedirs("./figures", exist_ok=True)
plt.savefig("./figures/global_eqode.pdf", bbox_inches='tight', pad_inches=0.02)

##############################################################################################################################

# plt.figure(figsize=(16,7))
# cmap = 'PiYG'
# for key_id, key in enumerate(list(fields.keys())[:-1]):
#     plt.subplot(231+key_id)
#     plt.title(key)
#     im = plt.pcolormesh(kappa, Aplus, fields[key], cmap=cmap)
#     reg = fields_regs[key]
#     field_reg = reg.predict(np.column_stack((kappa_test.ravel(), Aplus_test.ravel()))).reshape(kappa_test.shape)
#     im = plt.pcolormesh(kappa, Aplus, fields[key] - field_reg, cmap=cmap)
#     plt.xlabel(r'$\kappa$')
#     plt.ylabel(r'$A^+$')
#     plt.colorbar(im)
# plt.subplot(236)
# plt.title('success')
# im_s = plt.pcolormesh(
#     kappa, Aplus, fields['success'].astype(int), 
#     cmap=plt.get_cmap('binary', 2), 
#     norm=mcolors.BoundaryNorm(boundaries=[-0.5, 0.5, 1.5], ncolors=2)
#     )
# plt.xlabel(r'$\kappa$')
# plt.ylabel(r'$A^+$')
# cbar = plt.colorbar(im_s, ticks=[0, 1])
# cbar.set_ticklabels(['False','True'])
# plt.tight_layout()

# fig = plt.figure()
# for key_id, key in enumerate(list(fields.keys())[:-1]):
#     reg = fields_regs[key]
#     field_reg = reg.predict(np.column_stack((kappa_test.ravel(), Aplus_test.ravel()))).reshape(kappa_test.shape)
#     ax = fig.add_subplot(2, 3, key_id + 1, projection='3d')
#     ax.plot_surface(kappa, Aplus, fields[key], alpha=0.5)
#     ax.plot_wireframe(kappa_test, Aplus_test, field_reg, color='r', alpha=0.5)
#     ax.set_title(key)
#     ax.set_xlabel(r'$\kappa$')
#     ax.set_ylabel(r'$A^+$')
#     ax.set_zlabel(r'parameter')
