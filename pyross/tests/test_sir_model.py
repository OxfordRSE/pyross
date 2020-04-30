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
    S0  = N - (Ia0 + Is0 + R0)

    # No contact structure
    def contactMatrix(t):
        return np.identity(M)

    # Duration of simulation and data file
    Tf = 160
    Nt = 160

    # Instantiate model
    beta = 0.2
    gI = 0.1
    parameters = {'alpha': 0, 'beta': beta, 'gIa': gI, 'gIs': gI, 'fsa': 0.5}
    model = pyross.deterministic.SIR(parameters, M, Ni)

    # Simulate
    data = model.simulate(S0, Ia0, Is0, contactMatrix, Tf, Nt)

    # Extract data from dict
    S  = data['X'][:, 0]
    Ia = data['X'][:, 1]
    Is = data['X'][:, 2]
    t = data['t']

    # Test against simple version
    test_data = simulate_simple_sir([N - 2 * M, 2 * M, 0], beta, gI, t)

    # Compare graphically, for debugging



    # Compare numerically
    S_ref = test_data[:, 0]
    np.testing.assert_allclose(S, S_ref, rtol=1e-6)

    I_ref = test_data[:, 1]
    np.testing.assert_allclose(Is, I_ref, rtol=1e-6)

    # No asymptomatic cases
    np.testing.assert_allclose(Ia, 0 * I_ref)



if __name__ == '__main__':
    test_sir_model()
