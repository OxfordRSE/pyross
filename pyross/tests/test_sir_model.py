#
# Test using the models as SIR models, without age-structure.
#
import numpy as np
import pyross
import pytest

from simple_sir_model import simulate_simple_sir


def test_simple_sir_model():
    """
    Tests the simple SIR model used for testing.
    """
    # Run simple simulation
    t = np.arange(0, 160)
    test_data = simulate_simple_sir([999, 1, 0], 0.1, 0.2, t)

    # Check the total population is conserved
    n = np.sum(test_data, axis=1)
    np.testing.assert_allclose(n, 1000 * np.ones((160, )))

    # Check all values are within sensible bounds
    assert np.all(test_data >= 0)
    assert np.all(test_data <= 1000)


def test_sir_model_no_asymptomatic():
    """
    Test running simulations with a basic SIR model, no asymptomatic population
    and only a single age group.
    """

    # Number of age groups, size of each group, total population
    M = 1
    Ni = 1000 * np.ones(M)
    N = np.sum(Ni)

    # Asymptomatic, symptomatic, recovered, susceptible
    Ia0 = np.array([0])
    Is0 = np.array([1])
    R0  = np.array([0])
    S0  = N - (Ia0 + Is0 + R0)

    # No contact structure
    def contactMatrix(t):
        return np.identity(M)

    # Duration of simulation and data file
    Tf = 160
    Nt = 160

    # Instantiate model
    beta = 0.2
    gIs = 0.1
    parameters = {'alpha': 0, 'beta': beta, 'gIa': 0.1, 'gIs': gIs, 'fsa': 1}
    model = pyross.deterministic.SIR(parameters, M, Ni)

    # Simulate
    data = model.simulate(S0, Ia0, Is0, contactMatrix, Tf, Nt)

    # Extract data from dict
    S  = data['X'][:, 0]
    Ia = data['X'][:, 1]
    Is = data['X'][:, 2]
    t = data['t']

    # Test against simple version
    test_data = simulate_simple_sir([999, 1, 0], beta, gIs, t)

    S_ref = test_data[:, 0]
    np.testing.assert_allclose(S, S_ref, rtol=1e-6)

    I_ref = test_data[:, 1]
    np.testing.assert_allclose(Is, I_ref, rtol=1e-6)

    # No asymptomatic cases
    np.testing.assert_allclose(Ia, 0 * I_ref)


def test_sir_model():
    """
    Test running simulations with a basic SIR model, with some asympytomatic
    population and age groups (but identity contact matrices).
    """

    # Number of age groups, size of each group, total population
    M = 11
    Ni = 1000 * np.ones(M)
    N = np.sum(Ni)

    # Asymptomatic, symptomatic, recovered, susceptible
    Ia0 = np.ones(M)
    Is0 = np.ones(M)
    R0  = np.zeros(M)
    S0  = np.ones(M) * Ni - (Ia0 + Is0 - R0)

    # No contact structure
    def contactMatrix(t):
        return np.identity(M)

    # Duration of simulation and data file
    Tf = 160
    Nt = 160

    # Instantiate model
    beta = 0.2
    gI = 0.1
    fsa = 1   # No self-isolation: C in isolation = fsa * C normal
    parameters = {'alpha': 0.5, 'beta': beta, 'gIa': gI, 'gIs': gI, 'fsa': 1}
    model = pyross.deterministic.SIR(parameters, M, Ni)

    # Simulate
    data = model.simulate(S0, Ia0, Is0, contactMatrix, Tf, Nt)

    # Extract data from dict
    X = data['X']
    S = np.sum(X[:, 0:M], axis=1)
    I = np.sum(X[:, M:3 * M], axis=1)
    R = N - S - I
    t = data['t']

    # Test against simple version
    test_data = simulate_simple_sir([N - 2 * M, 2 * M, 0], beta, gI, t)

    # Compare graphically, for debugging
    if False:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10, 8))

        plt.fill_between(t, 0, S, color="#348ABD", alpha=0.3)
        plt.plot(t, S, '-', color="#348ABD", label='$S$', lw=4)
        plt.fill_between(t, 0, I, color='#A60628', alpha=0.3)
        plt.plot(t, I, '-', color='#A60628', label='$I$', lw=4)
        plt.fill_between(t, 0, R, color="dimgrey", alpha=0.3)
        plt.plot(t, R, '-', color="dimgrey", label='$R$', lw=4)
        plt.plot(t, test_data[:, 0], 'k--')
        plt.plot(t, test_data[:, 1], 'k--')
        plt.plot(t, test_data[:, 2], 'k--')
        plt.legend()
        plt.grid()
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.show()

    # Compare numerically
    S_ref = test_data[:, 0]
    np.testing.assert_allclose(S, S_ref, rtol=1e-6)
    I_ref = test_data[:, 1]
    np.testing.assert_allclose(I, I_ref, rtol=1e-6)



if __name__ == '__main__':
    test_sir_model()
