# Import necessary packages
import warnings # Ignore user warnings
import itertools as it # Readable nested for loops
from pathlib import Path # Filepaths
import typing # Argument / output type checking
import numpy as np # N-dim arrays + math
import scipy.linalg as spla # Complex linear algebra
import scipy.signal as spsg # Signal processing
import matplotlib.pyplot as plt # Plots
import matplotlib.figure as figure # Figure documentation
import sympy # Symbolic math + pretty printing


def etch(sym: str, mat: np.ndarray):
    display(sympy.Eq(sympy.Symbol(sym),
                     sympy.Matrix(mat.round(5)),
                     evaluate = False))
    pass


def d2c(A: np.ndarray, B: np.ndarray,
        dt: float) \
        -> typing.Tuple[np.ndarray, np.ndarray]:
    """Convert discrete linear state space model to continuous linear state space model.

    :param np.ndarray A:
    :param np.ndarray B:
    :param float dt: Timestep duration
    :return: (A_c, B_c) Continuous-time linear state space model
    """
    A_c = spla.logm(A)/dt
    if np.linalg.cond(A - np.eye(*A.shape)) < 1/np.spacing(1):
        B_c = A_c @ spla.inv(A - np.eye(*A.shape)) @ B
    else:
        B_temp = np.zeros(A_c.shape)
        for i in range(200):
            B_temp += (1/((i + 1)*np.math.factorial(i)))*np.linalg.matrix_power(A_c, i)*(dt**(i + 1))
        B_c = B @ spla.inv(B_temp)
    return A_c, B_c


def c2d(A_c: np.ndarray, B_c: np.ndarray,
        dt: float) \
        -> typing.Tuple[np.ndarray, np.ndarray]:
    """Convert continuous linear state space model to discrete linear state space model.

    :param np.ndarray A_c:
    :param np.ndarray B_c:
    :param float dt: Timestep duration
    :return: (A, B) Discrete-time linear state space model
    """
    A = spla.expm(A_c*dt)
    if np.linalg.cond(A_c) < 1/np.spacing(1):
        B = (A - np.eye(*A.shape)) @ spla.inv(A_c) @ B_c
    else:
        B_temp = np.zeros(A_c.shape)
        for i in range(200):
            B_temp += (1/((i + 1)*np.math.factorial(i)))*np.linalg.matrix_power(A_c, i)*(dt**(i + 1))
        B = B_temp @ B_c
    return A, B


def sim_ss(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray,
           X_0: np.ndarray, U: np.ndarray,
           nt: int) \
        -> typing.Tuple[np.ndarray, np.ndarray]:
    """Simulate linear state space model via ZOH.

    :param np.ndarray A:
    :param np.ndarray B:
    :param np.ndarray C:
    :param np.ndarray D:
    :param np.ndarray X_0: Initial state condition
    :param np.ndarray U: Inputs, either impulse or continual
    :param nt: Number of timesteps to simulate
    :return: (X) State vector array over duration; (Z) Observation vector array over duration
    """
    assert D.shape == (C @ A @ B).shape
    assert X_0.shape[-2] == A.shape[-1]
    assert U.shape[-2] == B.shape[-1]
    assert A.shape[-2] == B.shape[-2]
    assert C.shape[-2] == D.shape[-2]
    assert A.shape[-1] == C.shape[-1]
    assert B.shape[-1] == D.shape[-1]
    assert (U.shape[-1] == 1) or (U.shape[-1] == nt) or (U.shape[-1] == nt - 1)

    X = np.concatenate([X_0, np.zeros([X_0.shape[-2], nt])], 1)
    Z = np.zeros([C.shape[-2], nt])
    if U.shape[-1] == 1: # Impulse
        X[:, 1] = (A @ X[:, 0]) + (B @ U[:, 0])
        Z[:, 0] = (C @ X[:, 0]) + (D @ U[:, 0])
        for i in range(1, nt):
            X[:, i + 1] = (A @ X[:, i])
            Z[:, i] = (C @ X[:, i])
    else: # Continual
        for i in range(nt):
            X[:, i + 1] = (A @ X[:, i]) + (B @ U[:, i])
            Z[:, i] = (C @ X[:, i]) + (D @ U[:, i])
    return X, Z


def markov_sim(Y: np.ndarray, U: np.ndarray) \
        -> np.ndarray:
    """Obtain observations from Markov parameters and inputs, for zero initial conditions

    :param np.ndarray Y: Markov parameter matrix
    :param np.ndarray U: Continual inputs
    :return: (Z) Observation vector array over duration
    :rtype: np.ndarray
    """
    l, m, r = Y.shape
    Y_2_Z = np.zeros([r*l, l])
    Y_2_Z[:r, :] = U
    for i in range(1, l):
        Y_2_Z[r*i:r*(i + 1), :] = np.concatenate([np.zeros([r, i]), U[:, 0:(-i)]], 1)
    Z = np.concatenate(Y, 1) @ Y_2_Z
    return Z


def ss2markov(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray,
              nt: int) \
        -> np.ndarray:
    """Get Markov parameters from state space model.

    :param np.ndarray A:
    :param np.ndarray B:
    :param np.ndarray C:
    :param np.ndarray D:
    :param nt: Number of Markov parameters to generate (i.e., length of simulation)
    :return: (Y) 3D array of Markov parameters
    :rtype: np.ndarray
    """
    assert D.shape == (C @ A @ B).shape
    Y = np.zeros([nt, *D.shape])
    Y[0] = D
    for i in range(1, nt):
        Y[i] = C @ (np.linalg.matrix_power(A, i - 1)) @ B
    return Y


def Hankel(Y: np.ndarray, alpha: int, beta: int, i: int = 0) \
        -> np.ndarray:
    """Hankel matrix.

    :param Y: Markov parameter matrix
    :param alpha: Num. of rows of Markov parameters in Hankel matrix
    :param beta: Num. of columns of Markov parameters in Hankel matrix
    :param i: Start node of Hankel matrix
    :return: Block Hankel matrix.
    :rtype: np.ndarray
    """
    assert (len(Y) - 1) >= (i + alpha + beta - 1)
    m, r = Y.shape[-2:]
    H = np.zeros([alpha*m, beta*r])
    for j in range(beta):
        H[:, (j*r):((j + 1)*r)] = Y[(i + 1 + j):(i + alpha + 1 + j)].reshape([alpha*m, r])
    return H


def era(Y: np.ndarray, alpha: int, beta: int, n: int) \
        -> typing.Tuple[np.ndarray,
                        np.ndarray,
                        np.ndarray,
                        np.ndarray,
                        np.ndarray]:
    """Eigensystem Realization Algorithm (ERA).

    :param np.ndarray Y: Markov parameter matrix
    :param int alpha: Num. of rows of Markov parameters in Hankel matrix
    :param int beta: Num. of columns of Markov parameters in Hankel matrix
    :param int n: Order of proposed linear state space system
    :returns: (A, B, C, D) - State space of proposed linear state space system; (S) - Singular Values of H(0)
    :rtype: (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    """
    assert (len(Y) - 1) >= (alpha + beta - 1)
    m, r = Y.shape[-2:]
    assert (alpha >= (n/m)) and (beta >= (n/r))
    H_0 = Hankel(Y, alpha, beta, 0)
    print(f"Rank of H(0): {np.linalg.matrix_rank(H_0)}")
    H_1 = Hankel(Y, alpha, beta, 1)
    print(f"Rank of H(1): {np.linalg.matrix_rank(H_1)}")
    U_sim, S, Vh = np.linalg.svd(H_0)
    V = Vh.T
    U_n = U_sim[:, :n]
    V_n = V[:, :n]
    S_n = S[:n]

    E_r = np.concatenate([np.eye(r), np.tile(np.zeros([r, r]), beta - 1)], 1).T
    E_m = np.concatenate([np.eye(m), np.tile(np.zeros([m, m]), alpha - 1)], 1).T
    A = np.diag(S_n**(-1/2)) @ U_n.T @ H_1 @ V_n @ np.diag(S_n**(-1/2))
    B = np.diag(S_n**(1/2)) @ V_n.T @ E_r
    C = E_m.T @ U_n @ np.diag(S_n**(1/2))
    D = Y[0]
    return A, B, C, D, S


def okid(Z: np.ndarray, U: np.ndarray,
         l_0: int,
         alpha: int, beta: int,
         n: int):
    """Observer Kalman Identification Algorithm (OKID).

    :param np.ndarray Z: Observation vector array over duration
    :param np.ndarray U: Continual inputs
    :param int l_0: Order of OKID to execute (i.e., number of Markov parameters to generate via OKID)
    :param int alpha: Num. of rows of Markov parameters in Hankel matrix
    :param int beta: Num. of columns of Markov parameters in Hankel matrix
    :param int n: Number of proposed states to use for ERA
    :return: (Y) Markov parameters
    :rtype: np.ndarray
    """
    r, l_u = U.shape
    m, l = Z.shape
    assert l == l_u
    V = np.concatenate([U, Z], 0)
    assert (max([alpha + beta, (n/m) + (n/r)]) <= l_0) and (l_0 <= (l - r)/(r + m)) # Boundary conditions

    # Form observer
    Y_2_Z = np.zeros([r + (r + m)*l_0, l])
    Y_2_Z[:r, :] = U
    for i in range(1, l_0 + 1):
        Y_2_Z[((i*r) + ((i - 1)*m)):(((i + 1)*r) + (i*m)), :] = np.concatenate([np.zeros([r + m, i]), V[:, 0:(-i)]], 1)
    # Find Observer Markov parameters via least-squares
    Y_obs = Z @ spla.pinv2(Y_2_Z)
    Y_bar_1 = np.array(list(it.chain.from_iterable([Y_obs[:, i:(i + r)]
                                                    for i in range(r, r + (r + m)*l_0, r + m)]))).reshape([l_0, m, r])
    Y_bar_2 = -np.array(list(it.chain.from_iterable([Y_obs[:, i:(i + m)]
                                                     for i in range(2*r, r + (r + m)*l_0, r + m)]))).reshape([l_0, m, m])

    # Obtain Markov parameters from Observer Markov parameters
    Y = np.zeros([l_0 + 1, m, r])
    Y[0] = Y_obs[:, :r]
    for k in range(1, l_0 + 1):
        Y[k] = Y_bar_1[k - 1] - \
               np.array([Y_bar_2[i] @ Y[k - (i + 1)]
                         for i in range(k)]).sum(axis = 0)
    # Obtain Observer Gain Markov parameters from Observer Markov parameters
    Y_og = np.zeros([l_0, m, m])
    Y_og[0] = Y_bar_2[0]
    for k in range(1, l_0):
        Y_og[k] = Y_bar_2[k] - \
                  np.array([Y_bar_2[i] @ Y_og[k - (i + 1)]
                            for i in range(k - 1)]).sum(axis = 0)
    return Y, Y_og
