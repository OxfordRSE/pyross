#
# SIR Epidemiology toy model.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import numpy as np
from scipy.integrate import odeint

def simulate_simple_sir(y0, beta, gamma, times):
    """
    Simulate starting from initial population ``y0`` with parameters ``beta``
    and ``gamma``, and logging at the specified vector of ``times``.

    Returns a numpy array of shape ``(3, len(times))``
    """
    y0 = np.array(y0)

    if len(y0) != 3:
        raise ValueError('Initial value must have size 3.')
    if np.any(y0 < 0):
        raise ValueError('Initial states can not be negative.')
    if times[0] != 0:
        # odeint requires the initial time to be included in the times vector
        raise ValueError(
            'The initial time t=0 must be included in `times`.')

    def rhs(y, t, i, g):
        dS = -i * y[0] * y[1]
        dI = +i * y[0] * y[1] - g * y[1]
        dR = g * y[1]
        return np.array([dS, dI, dR])

    return odeint(rhs, y0, times, (beta / np.sum(y0), gamma), mxstep=100000)

