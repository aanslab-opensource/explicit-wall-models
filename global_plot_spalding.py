import os
import numpy as np
import matplotlib.pyplot as plt
import ewmlib
import matplotlib as mpl
import matplotlib.colors as mcolors
from sklearn import linear_model
import pickle


print('0 - load data')

with open('./results/global_spalding.pkl', 'rb') as file:
    data = pickle.load(file)

file.close()

kappa = data['kappa']
B = data['B']
kappa_test = data['kappa_test']
B_test = data['B_test']
fields = data['fields']
fields_regs = data['fields_regs']
results = data['results']
up_ref = data['up_ref']





p_reg      = fields_regs['p'].predict(     np.column_stack((kappa_test.ravel(), B_test.ravel()))).reshape(kappa_test.shape)
s_reg      = fields_regs['s'].predict(     np.column_stack((kappa_test.ravel(), B_test.ravel()))).reshape(kappa_test.shape)
mu1_reg    = fields_regs['mu1'].predict(   np.column_stack((kappa_test.ravel(), B_test.ravel()))).reshape(kappa_test.shape)
sigma1_reg = fields_regs['sigma1'].predict(np.column_stack((kappa_test.ravel(), B_test.ravel()))).reshape(kappa_test.shape)
scale1_reg = fields_regs['scale1'].predict(np.column_stack((kappa_test.ravel(), B_test.ravel()))).reshape(kappa_test.shape)

##############################################################################################################################

# mpl.rcParams['lines.linewidth'] = 0.5
plt.style.use('tableau-colorblind10')
plt.rc("text", usetex=True)
plt.rcParams.update({"font.size": 10})
plt.rcParams.update({"axes.grid":True, "grid.color":"#DDDDDD", "grid.linestyle":"dashed"})

##############################################################################################################################

print('1 - plot data')

fig, ax = plt.subplots(1, 1, figsize=(0.8*8.3, 2.5))

error_values = []

up_ref = np.linspace(0.01, 35, 1000)
for idx, _ in np.ndenumerate(kappa):    
    rey_spalding = ewmlib.laws.yp_Spalding(up_ref, kappa[idx], B[idx]) * up_ref
    up_caisag = ewmlib.laws.up_CaiSagaut_m2_var(rey_spalding, kappa[idx], B[idx], fields['p'][idx], fields['s'][idx])
    up_delta = ewmlib.model1(np.log10(rey_spalding), *results[idx].x[:-2])
    error_values.append(100 * (up_ref - (up_caisag + up_delta)) / up_ref)
    
error_values = np.vstack(error_values)
e_min = error_values.min(axis=0)
e_max = error_values.max(axis=0)
ax.fill_between(rey_spalding, e_min, e_max, color='k', alpha=1, label=r'$\mathbf{\tilde{\Pi}}_1(\mathbf{\Psi}_{Sp})$')
ax.set_xscale('log')

error_interp = []
for idx_test, _ in np.ndenumerate(kappa_test):    
    rey_spalding = ewmlib.laws.yp_Spalding(up_ref, kappa_test[idx_test], B_test[idx_test]) * up_ref
    up_caisag_reg = ewmlib.laws.up_CaiSagaut_m2_var(rey_spalding, kappa_test[idx_test], B_test[idx_test], p_reg[idx_test], s_reg[idx_test])
    up_delta_reg = ewmlib.model1(np.log10(rey_spalding), mu1_reg[idx_test], sigma1_reg[idx_test], scale1_reg[idx_test])
    error_interp.append(100 * (up_ref - (up_caisag_reg + up_delta_reg)) / up_ref)

error_interp = np.vstack(error_interp)

e_min = error_interp.min(axis=0)
e_max = error_interp.max(axis=0)
ax.fill_between(rey_spalding, e_min, e_max, color='C4', alpha=0.75, label=r'$\tilde{\pi}_{Sp, 1}(\mathbf{\Psi}_{Sp})$')
ax.set_xscale('log')

ax.set_xlabel(r'$Re_y$')
ax.set_xlabel(r'$Re_y$')
ax.set_ylabel(r"$\mathbf{err}^1_{\mathbf{\Psi}_{Sp}}$, \%")
ax.set_xlim(1e-4, 1e+6)
ax.set_ylim(-0.78, 0.52)
ax.legend()

plt.tight_layout()
# plt.show()

os.makedirs("./figures", exist_ok=True)
plt.savefig("./figures/global_spalding.pdf", bbox_inches='tight', pad_inches=0.02)

##############################################################################################################################

# plt.figure(figsize=(16,7))
# cmap = 'PiYG'
# for key_id, key in enumerate(list(fields.keys())[:-1]):
#     plt.subplot(231+key_id)
#     plt.title(key)
#     im = plt.pcolormesh(kappa, B, fields[key], cmap=cmap)
#     plt.xlabel(r'$\kappa$')
#     plt.ylabel(r'$B$')
#     plt.colorbar(im)
# plt.subplot(236)
# plt.title('success')
# im_s = plt.pcolormesh(
#     kappa, B, fields['success'].astype(int), 
#     cmap=plt.get_cmap('binary', 2), 
#     norm=mcolors.BoundaryNorm(boundaries=[-0.5, 0.5, 1.5], ncolors=2)
#     )
# plt.xlabel(r'$\kappa$')
# plt.ylabel(r'$B$')
# cbar = plt.colorbar(im_s, ticks=[0, 1])
# cbar.set_ticklabels(['False','True'])
# plt.tight_layout()


# fig = plt.figure()
# for key_id, key in enumerate(list(fields.keys())[:-1]):
#     reg = fields_regs[key]    
#     field_reg = reg.predict(np.column_stack((kappa_test.ravel(), B_test.ravel()))).reshape(kappa_test.shape)
#     ax = fig.add_subplot(2, 3, key_id + 1, projection='3d')
#     ax.plot_surface(kappa, B, fields[key], alpha=0.5)
#     ax.plot_wireframe(kappa_test, B_test, field_reg, color='r', alpha=0.5)
#     ax.set_title(key)
#     ax.set_xlabel(r'$\kappa$')
#     ax.set_ylabel(r'$B$')
#     ax.set_zlabel(r'parameter')

