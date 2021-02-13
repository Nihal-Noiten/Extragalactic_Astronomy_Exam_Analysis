import	os
import	argparse
import	datetime
import	time
for		time					import sleep
import	sys	
from	sys						import stdout
import	numpy 					as np
import	matplotlib.pyplot		as plt
from	matplotlib				import rc
from	matplotlib.ticker		import FormatStrFormatter, MultipleLocator, FuncFormatter
from	matplotlib.gridspec		import GridSpec
import	matplotlib.ticker		as tck
import	matplotlib.colors		as colors
import	matplotlib.patches		as patches
from	timeit					import default_timer as timer
from	astropy.modeling 		import models, fitting
from	scipy.optimize			import curve_fit
from	scipy.integrate 		import quad, ode
# from scipy.special import erf

#######################################################################################################
#######################################################################################################

# LATEX: ON, MNRAS template

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",				# or sans-serif
    "font.serif": ["Times New Roman"]})	# or Helvetica

#######################################################################################################
#######################################################################################################

# FUNCTIONS

# The tree inverse CDFs for the homogeneous sphere:

def R_P(P,a):
	return a * np.cbrt(P)

def Ph_P(P):
	return 2 * np.pi * P

def Th_P(P):
	return np.arccos(1. - 2. * P)

# The tree PDFs for the homogeneous sphere:

def pdf_r(r,a):
	return 3. * r**2 / (a**3)

def pdf_ph(ph):
	return 0.5 / np.pi * (1. + 0. * ph)

def pdf_th(th):
	return 0.5 * np.sin(th)

# The circular velocity of the Plummer sphere, and derivative formulae

def v_circ(r):
	return r * np.sqrt( rho * 4. * np.pi / 3. )

def v_case_1(r):
	return 0.05 * v_circ(r)

def v_case_2(r):
	return 0.05 * a * v_circ(a) / r

def pdf_v_1(v):
	k = 0.05 * np.sqrt( rho * 4. * np.pi / 3. )
	return 3. / ( (k * a * V0)**3 ) * v**2 

def pdf_v_2(v):
	k = 0.05 * np.sqrt( rho * 4. * np.pi / 3. ) * (a**2) * V0 * R0
	return 3. * (k**3) / (a**3 * R0**3) / (v**4)

def pdf_l_1(l):
	k = 0.05 * np.sqrt( rho * 4. * np.pi / 3. ) * V0 / R0
	return 1.5 / (a**3 * R0**3) / (k**(3./2.)) * l**(1./2.) 

# A plotting primer for PDFs over histograms

def histo_pdf_plotter(ax, x_lim, x_step, x_bins, func, npar, x_min=0.):
	x = np.linspace(x_min, x_lim, 10000)
	if npar == 1:
		f_x = func(x, a * R0)
		ax.plot(x, f_x, color='black', lw=0.9)
	elif npar == 0:
		f_x = func(x)
		ax.plot(x, f_x, color='black', lw=0.9)
	for j in range(len(x_bins)):
		x = []
		f_x = []
		e_x = []
		x_mid = x_bins[j] + x_step / 2
		for i in range(9):
			x_temp = x_bins[j] + x_step / 2 + x_step / 15 * (i-4)
			if npar == 1:
				f_temp = func(x_mid, a * R0)
			elif npar == 0:
				f_temp = func(x_mid)
			e_temp = np.sqrt(f_temp * N) / N
			ax.vlines(x_mid, ymin=f_temp-e_temp, ymax=f_temp+e_temp, color='black', lw=0.9)
			x.append(x_temp)
			f_x.append(f_temp)
			e_x.append(e_temp)
		x = np.array(x)
		f_x = np.array(f_x)
		e_x = np.array(e_x)
		ax.plot(x, f_x + e_x, color='black', lw=0.9)
		ax.plot(x, f_x - e_x, color='black', lw=0.9)

def histo_pdf_plotter_log(ax, x_min, x_max, x_step, x_bins, func, npar):
	log_min = np.log10(x_min)
	log_max = np.log10(x_max)
	x = np.logspace(log_min, log_max, 1000)
	if npar == 1:
		f_x = func(x, a * R0) # * N
		ax.plot(x, f_x, color='black', lw=0.9)
	elif npar == 0:
		f_x = func(x) # * N
		ax.plot(x, f_x, color='black', lw=0.9)
	for j in range(len(x_bins)-1):
		x = []
		f_x = []
		e_x = []
		x_mid = (x_bins[j+1] + x_bins[j]) / 2.
		x_step = x_bins[j+1] - x_bins[j]
		for i in range(9):
			x_temp = x_mid + x_step / 15 * (i-4)
			if npar == 1:
				f_temp = func(x_mid, a * R0) #* N
			elif npar == 0:
				f_temp = func(x_mid) # * N
			e_temp = np.sqrt(f_temp * N) / N # np.sqrt(f_temp) # 
			ax.vlines(x_mid, ymin=f_temp-e_temp, ymax=f_temp+e_temp, color='black', lw=0.9)
			x.append(x_temp)
			f_x.append(f_temp)
			e_x.append(e_temp)
		x = np.array(x)
		f_x = np.array(f_x)
		e_x = np.array(e_x)
		ax.plot(x, f_x + e_x, color='black', lw=0.9)
		ax.plot(x, f_x - e_x, color='black', lw=0.9)

#######################################################################################################
#######################################################################################################

time_prog_start = timer()

lim_3d = 10.
a. = 5.

# Set up to obtain the accurate conversion factors from internal units to physical units:

G_cgs = 6.67430e-8			# cm^3 g^-1 s^-2
pc_cgs = 3.08567758e18		# cm
Msun_cgs = 1.98855e33 		# g
Myr_cgs = 31557600. * 1e6 	# s

# Conversion factors from internal units to the chosen physical units:

G0 = 1.
R0 = 5.													# pc
M0 = 1e4												# Msun
V0 = np.sqrt( G_cgs * (Msun_cgs * M0) / (pc_cgs * R0) )	# cm/s
T0 = (pc_cgs * R0) / V0									# s
T0 = T0 / Myr_cgs										# Myr
V0 = V0 / 1e5 											# km/s

file = open(plotfile, "r")
firstrow = (file.readline()).strip("\n")
NL = 1
for line in file:
	NL += 1
file.close()

I = int(firstrow)				# N_particles
L = 3+4*I						# N_lines for each timestep
NT = int(NL / L)				# N_timesteps	
X = np.zeros((I,4,NT))			# Empty array for the positions at each t
V = np.zeros((I,4,NT))			# Empty array for the velocities at each t
P = np.zeros((I,NT))			# Empty array for the potentials at each t
K = np.zeros((I,NT))			# Empty array for the kinetic energies at each t
E = np.zeros((I,NT))			# Empty array for the energies at each t
M = np.zeros((I,NT))			# Empty array for the masses at each t (should be const)
N = np.zeros(NT)				# Empty array for the N_particles at each t (should be const)
T = np.zeros(NT)				# Empty array for the times t

file = open(plotfile, "r")		# Read data!
i = 0
t = 0
for line in file:
	a = line.strip("\n")
	j = i % L
	if j == 0:
		N[t] = float(a)
	elif j == 2:
		T[t] = float(a) * T0
	elif j >= 3 and j < (3+I): 
		m = j-3
		M[m,t] = float(a) * M0
	elif j >= (3+I) and j < (3+2*I):
		m = j - (3+I)
		b = a.split()
		for k in range(len(b)):
			X[m,k+1,t] = float(b[k]) * R0
	elif j >= (3+2*I) and j < (3+3*I):
		m = j - (3+2*I)
		b = a.split()
		for k in range(len(b)):
			V[m,k+1,t] = float(b[k]) * V0
	elif j >= (3+3*I) and j < (3+4*I):
		m = j - (3+3*I)
		P[m,t] = float(a) * M0 / R0
		if (j+1) == (3+4*I):
			t += 1
			if t == NT:
				break
	i += 1
file.close()

print("Number of bodies: {:d}".format(I))
print('Conversion factors to physical units:')
print('1 r_IU = {:} pc'.format(R0))
print('1 m_IU = {:} M_sun'.format(M0))
print('1 v_IU = {:} km/s'.format(V0))
print('1 t_IU = {:} Myr'.format(T0))

M_tot = 0
for i in range(I):
	M_tot += M[i,0]

print('M_tot = {:}'.format(M_tot))
print('m_i   = {:}'.format(M[37,0]))

time_prog_load = timer()
print("Data loading time [hh/mm/ss] = {:}".format(datetime.timedelta(seconds = time_prog_load - time_prog_start)))

#######################################################################################################
#######################################################################################################

# Let us fill the 0-th component (at any t) of each particle's position and velocity with their moduli

P_tot = np.zeros((NT))
K_tot = np.zeros((NT))
E_tot = np.zeros((NT))

for t in range(NT):
	for i in range(I):
		X[i,0,t] = np.sqrt(X[i,1,t]**2 + X[i,2,t]**2 + X[i,3,t]**2)
		V[i,0,t] = np.sqrt(V[i,1,t]**2 + V[i,2,t]**2 + V[i,3,t]**2)
		P[i,t] = P[i,t] * G_cgs * Msun_cgs / pc_cgs
		K[i,t] = 0.5 * (V[i,0,t])**2 * 1e10
		E[i,t] = P[i,t] + K[i,t]
		P_tot[t] += 0.5 * P[i,t]
		K_tot[t] += K[i,t]
		E_tot[t] += 0.5 * P[i,t] + K[i,t]

# find some useful limits for the plots
t_max = np.max(T)
r_max = 0.
v_max = 0.
for i in range(I):
	for t in range(NT):
		r_max = np.amax(np.array([ np.amax( X[i,0,t] ) , r_max ]))
		v_max = np.amax(np.array([ np.amax( V[i,0,t] ) , v_max ]))
r_max = 1.1 * r_max
v_max = 1.1 * v_max

#######################################################################################################
#######################################################################################################

fig_LR , ax_LR = plt.subplots(figsize=(5,5))
factor_tr = t_max / lim_3d
ax_LR.set_aspect(factor_tr)
ax_LR.grid(linestyle=':', which='both')
ax_LR.set_xlim(0, t_max)
ax_LR.set_ylim(0, lim_3d)
ax_LR.set_title('Lagrangian radii as functions of time')
ax_LR.set_xlabel(r'$t\;$[Myr]')
ax_LR.set_ylabel(r'$r\;$[pc]') # , rotation='horizontal', horizontalalignment='right'

RL = np.zeros((9,NT))
for k in range(9):
	for t in range(NT):
		C = np.copy(X[:,0,t])
		C = np.sort(C)
		RL[k,t] = C[int(np.ceil(I/10*(k+1))-1)]
	ax_LR.plot(T , RL[k,:] , linestyle='' , marker='o' , markersize=1, label='{:d}0'.format(k+1) + r'$\%\; M_{tot}$')
	
ax_LR.legend(frameon=False, bbox_to_anchor=(1.01,1), title=r'$\begin{array}{rcl} \;\;N \!\!&\!\! = \!\!&\!\! 10^{5} \\ M_{H} \!\!&\!\! = \!\!&\!\! 10^{5} \, M_{\odot} \\ M_{P} \!\!&\!\! = \!\!&\!\! 3 \cdot 10^{3} \, M_{\odot} \\ \;\;a & = & 2.3 \; \mathrm{pc} \end{array}$'+'\n')

fig_LR.tight_layout()

##################################################################################################

fig_E , ax_E = plt.subplots(figsize=(5,5))

ax_E.grid(linestyle=':')
ax_E.set_xlim(0, t_max)
# ax_E.set_ylim(-1e17,5e17)
# ax_E.set_aspect(t_max / 6e17)
ax_E.set_title('Total energies as functions of time\n')
ax_E.set_xlabel(r'$t\;$[Myr]')
ax_E.set_ylabel(r'$E\;$[erg/g]') # , rotation='horizontal', horizontalalignment='right'

ax_E.plot(T, E_tot, linestyle=':',  color='black', markersize=1, label=r'$E_{tot}$')
ax_E.plot(T, P_tot, linestyle='-.', color='black', markersize=1, label=r'$E_{pot}$')
ax_E.plot(T, K_tot, linestyle='--', color='black', markersize=1, label=r'$E_{kin}$')
ax_E.legend(frameon=False, bbox_to_anchor=(1.01,1))
fig_E.tight_layout()

##################################################################################################

fig_Phi = plt.figure(figsize=(6,12))
gs = GridSpec(3, 1, figure=fig_Phi)
ax_phi = []
ax_phi.append(fig_Phi.add_subplot(gs[0,0]))
ax_phi.append(fig_Phi.add_subplot(gs[1,0]))
ax_phi.append(fig_Phi.add_subplot(gs[2,0]))
for i in range(3):
	ax_phi[i].set_xlim(0, t_max)
	# ax_phi[i].set_ylim(-4e12,0)
	# ax_phi[i].set_aspect(t_max / 4e12)
# rrr = np.linspace(0,lim_3d,1000)
# fff = PHI_plum(rrr,M_tot,2.3) * G_cgs * Msun_cgs / pc_cgs

ttt = int(np.floor(NT/3))
for i in range(3):
	ax_phi[i].grid(linestyle=':')
	ax_phi[i].set_xlim(0, a) # lim_3d
	ax_phi[i].set_title('\nPotential at $t$ = {:.3f} Myr'.format(T[ttt*i]))
	ax_phi[i].set_xlabel(r'$r\;$[pc]')
	ax_phi[i].set_ylabel(r'$\Phi\;$[erg/g]') #, rotation='horizontal', horizontalalignment='right'
	ax_phi[i].scatter(X[:,0,ttt*i], P[:,ttt*i], color='lightgrey', s=0.5, label=r'$\Phi(r)\,:\;simulation$')
	# ax_phi[i].plot(rrr, fff, color='black' , markersize=1, label=r'$\Phi(r)\,:\;theory$')
	ax_phi[i].legend(frameon=True, loc=4)
fig_Phi.tight_layout()

##################################################################################################
##################################################################################################

fig_LR.savefig("C1_Results_PNG/Lagrangian_Radii_{:s}.png".format(plotfile), bbox_inches='tight', dpi=400)
fig_LR.savefig("C1_Results_EPS/Lagrangian_Radii_{:s}.eps".format(plotfile), bbox_inches='tight')

fig_E.savefig("C1_Results_PNG/Energy_{:s}.png".format(plotfile), bbox_inches='tight', dpi=400)
fig_E.savefig("C1_Results_EPS/Energy_{:s}.eps".format(plotfile), bbox_inches='tight')

fig_Phi.savefig("C1_Results_PNG/Potential_t_sample_{:s}.png".format(plotfile), bbox_inches='tight', dpi=400)
fig_Phi.savefig("C1_Results_EPS/Potential_t_sample_{:s}.eps".format(plotfile), bbox_inches='tight')

#######################################################################################################
#######################################################################################################





X_cm = np.zeros((4,NT))
V_cm = np.zeros((4,NT))
M_cm = np.zeros((NT))

time_prog_CM = timer()

plt.show()