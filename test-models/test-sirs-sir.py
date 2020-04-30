import numpy as np
import pyross
import matplotlib.pyplot as plt

M = 2  # the population has two age groups
N = 1000000  # and this is the total population

beta = 0.0131  # infection rate
gIa = 1. / 7  # recovery rate of asymptomatic infectives
gIs = 1. / 7  # recovery rate of symptomatic infectives
alpha = 0  # fraction of asymptomatic infectives
fsa = 1  # the self-isolation parameter

sa = np.zeros(M)  # arrival of new susceptibles
iaa = np.zeros(M)  # daily arrival of new  asymptomatics
ep = 0  # fraction of recovered who is susceptible

fi = np.array((0.25, 0.75))  # fraction of population in age age group
Ni = fi * N  # population in each group

# set the contact structure
C = np.array(([18., 9.], [3., 12.]))

Ia_0 = np.array((1, 1))  # each age group has asymptomatic infectives
Is_0 = np.array((1, 1))  # and also symptomatic infectives
R_0 = np.array((0, 0))  # there are no recovered individuals initially
S_0 = Ni - (Ia_0 + Is_0 + R_0)

# duration of simulation and data file
Tf = 200
Nf = 2000


# the contact structure is independent of time
def contactMatrix(time):
    return C


# instantiate models
parameters = {'alpha': alpha, 'beta': beta, 'gIa': gIa, 'gIs': gIs, 'fsa': fsa,
              'ep': ep, 'sa': sa, 'iaa': iaa}

model_sir = pyross.deterministic.SIR(parameters, M, Ni)
model_sirs = pyross.deterministic.SIRS(parameters, M, Ni)

# simulate model
data_sir = model_sir.simulate(S_0, Ia_0, Is_0, contactMatrix, Tf, Nf)
data_sirs = model_sirs.simulate(S_0, Ia_0, Is_0, contactMatrix, Tf, Nf)

IK_sir = data_sir.get('X')[:, 2 * M].flatten()
IK_sirs = 1.0 * data_sirs.get('X')[:, 2 * M].flatten()

t = data_sir.get('t')

# Assert that the results are close
np.testing.assert_allclose(IK_sir, IK_sirs)

# Also plot the results for visual confirmation
fig = plt.figure(num=None, figsize=(10, 8), dpi=80, facecolor='w',
                 edgecolor='k')
plt.rcParams.update({'font.size': 22})

plt.fill_between(t, 0, IK_sir / Ni[0], color="#348ABD", alpha=0.3)
plt.plot(t, IK_sir / Ni[0], '-', color="#348ABD", label='$Children (SIR)$',
         lw=4)

plt.fill_between(t, 0, IK_sirs / Ni[0], color="#A60628", alpha=0.3)
plt.plot(t, IK_sirs / Ni[0], '--', color="#A60628", label='$Children (SIRS)$',
         lw=4)

plt.legend(fontsize=26)
plt.grid()
plt.autoscale(enable=True, axis='x', tight=True)

plt.show()
