import numpy as np
from scipy.optimize import minimize
from functools import partial
from scipy.special import erf, erfcx, erfinv


def erf_times_gauss(X):
    return erfcx(X / np.sqrt(2)) * np.sqrt(np.pi / 2)


def log_erf_times_gauss(X):
    m = np.zeros(X.shape)
    tmp = X < 6
    m[tmp] = (0.5 * X[tmp]**2 + np.log(1 - erf(X[tmp] / np.sqrt(2))) - np.log(2))
    m[~tmp] = (0.5 * np.log(2 / np.pi) - np.log(2) -
               np.log(X[~tmp]) + np.log(1 - 1 / X[~tmp]**2 + 3 / X[~tmp]**4))
    return m + 0.5 * np.log(2*np.pi)


def energy(Y, gamma_plus, gamma_minus, theta_plus, theta_minus):
    Y_plus = np.maximum(Y, 0)
    Y_minus = np.maximum(-Y, 0)
    return (Y_plus**2 * gamma_plus) / 2.\
        + (Y_minus**2 * gamma_minus) / 2.\
        + (Y_plus * theta_plus)\
        + (Y_minus * theta_minus)


def cgf_from_inputs(I, gamma_plus, gamma_minus, theta_plus, theta_minus):
    return np.logaddexp(
        log_erf_times_gauss((-I + theta_plus) /
                            np.sqrt(gamma_plus)) - 0.5 * np.log(gamma_plus),
        log_erf_times_gauss((I + theta_minus) /
                            np.sqrt(gamma_minus)) - 0.5 * np.log(gamma_minus)
    )


def mean12_pm_from_inputs(I, gamma_plus, gamma_minus, theta_plus, theta_minus):
    I_plus = (-I + theta_plus) / np.sqrt(gamma_plus)
    I_minus = (I + theta_minus) / np.sqrt(gamma_minus)

    etg_plus = erf_times_gauss(I_plus)
    etg_minus = erf_times_gauss(I_minus)

    p_plus = 1 / (1 + (etg_minus / np.sqrt(gamma_minus)) /
                  (etg_plus / np.sqrt(gamma_plus)))
    nans = np.isnan(p_plus)
    p_plus[nans] = 1.0 * (np.abs(I_plus[nans]) > np.abs(I_minus[nans]))
    p_minus = 1 - p_plus
    mean_pos = (-I_plus + 1 / etg_plus) / np.sqrt(gamma_plus)
    mean_neg = (I_minus - 1 / etg_minus) / np.sqrt(gamma_minus)
    mean2_pos = 1 / gamma_plus * (1 + I_plus**2 - I_plus / etg_plus)
    mean2_neg = 1 / gamma_minus * (1 + I_minus**2 - I_minus / etg_minus)
    return (p_plus * mean_pos, p_minus * mean_neg, p_plus * mean2_pos, p_minus * mean2_neg)


def mean_from_inputs(I, gamma_plus, gamma_minus, theta_plus, theta_minus):
    mu_pos, mu_neg, _, _ = mean12_pm_from_inputs(
        I, gamma_plus, gamma_minus, theta_plus, theta_minus)
    return mu_pos + mu_neg


def sample_from_inputs(I, gamma_plus, gamma_minus, theta_plus, theta_minus):
    I_plus = (-I + theta_plus) / np.sqrt(gamma_plus)
    I_minus = (I + theta_minus) / np.sqrt(gamma_minus)

    etg_plus = erf_times_gauss(I_plus)
    etg_minus = erf_times_gauss(I_minus)

    p_plus = 1 / (1 + (etg_minus / np.sqrt(gamma_minus)) /
                  (etg_plus / np.sqrt(gamma_plus)))
    nans = np.isnan(p_plus)
    p_plus[nans] = 1.0 * (np.abs(I_plus[nans]) > np.abs(I_minus[nans]))
    p_minus = 1 - p_plus

    is_pos = np.random.rand(*I.shape) < p_plus
    rmax = np.zeros(p_plus.shape)
    rmin = np.zeros(p_plus.shape)
    rmin[is_pos] = erf(I_plus[is_pos] / np.sqrt(2))
    rmax[is_pos] = 1
    rmin[~is_pos] = -1
    rmax[~is_pos] = erf(-I_minus[~is_pos] / np.sqrt(2))

    h = np.zeros(I.shape)
    tmp = (rmax - rmin > 1e-14)
    h = np.sqrt(2) * erfinv(rmin + (rmax - rmin)
                            * np.random.rand(*h.shape))
    h[is_pos] -= I_plus[is_pos]
    h[~is_pos] += I_minus[~is_pos]
    h /= np.sqrt(is_pos * gamma_plus + (1 - is_pos) * gamma_minus)
    h[np.isinf(h) | np.isnan(h) | ~tmp] = 0
    return h


class dReLU_Regression:
    def __init__(self, Nx=None, Ny=None):
        if Nx is not None:
            Nx = self.Nx
        if Ny is not None:
            Ny = self.Ny
        return

    def get_params(self, params=None):
        if params is not None:
            gamma_plus = params[:self.Ny]
            gamma_minus = params[self.Ny:2 * self.Ny]
            theta_plus = params[2 * self.Ny:3 * self.Ny]
            theta_minus = params[3 * self.Ny: 4 * self.Ny]
            weights = params[4 * self.Ny:].reshape([self.Nx, self.Ny])
        else:
            gamma_plus = self.gamma_plus
            gamma_minus = self.gamma_minus
            theta_plus = self.theta_plus
            theta_minus = self.theta_minus
            weights = self.weights
        return gamma_plus, gamma_minus, theta_plus, theta_minus, weights

    def negative_likelihood(self, X, Y, params=None):
        if X.ndim == 1:
            X = X[:, np.newaxis]
        if Y.ndim == 1:
            Y = Y[:, np.newaxis]
        gamma_plus, gamma_minus, theta_plus, theta_minus, weights = self.get_params(
            params=params)
        I = np.dot(X, weights)
        E = energy(Y, gamma_plus, gamma_minus, theta_plus, theta_minus) - I * Y
        log_partition = cgf_from_inputs(
            I, gamma_plus, gamma_minus, theta_plus, theta_minus)
        return -(-E - log_partition).sum(1).mean(0)

    def gradient_negative_likelihood(self, X, Y, params=None):
        if X.ndim == 1:
            X = X[:, np.newaxis]
        if Y.ndim == 1:
            Y = Y[:, np.newaxis]
        gamma_plus, gamma_minus, theta_plus, theta_minus, weights = self.get_params(
            params=params)

        I = np.dot(X, weights)
        mu_pos, mu_neg, mu2_pos, mu2_neg = mean12_pm_from_inputs(
            I, gamma_plus, gamma_minus, theta_plus, theta_minus)

        gradient_I = Y - (mu_pos + mu_neg)
        gradient_gamma_plus = -0.5 * (np.maximum(Y, 0)**2 - mu2_pos).mean(0)
        gradient_gamma_minus = -0.5 * (np.maximum(-Y, 0)**2 - mu2_neg).mean(0)
        gradient_theta_plus = -(np.maximum(Y, 0) - mu_pos).mean(0)
        gradient_theta_minus = (np.minimum(Y, 0) - mu_neg).mean(0)
        gradient_w = np.dot(X.T, gradient_I) / X.shape[0]
        gradient_params = np.concatenate((
            gradient_gamma_plus,
            gradient_gamma_minus,
            gradient_theta_plus,
            gradient_theta_minus,
            gradient_w.flatten()
        ), axis=0)
        return -gradient_params

    def predict(self, X, params=None):
        if X.ndim == 1:
            X = X[:, np.newaxis]
        gamma_plus, gamma_minus, theta_plus, theta_minus, weights = self.get_params(
            params=params)
        I = np.dot(X, weights)
        return mean_from_inputs(I, gamma_plus, gamma_minus, theta_plus, theta_minus)

    def predict_samples(self, X, repeats=1, params=None):
        if X.ndim == 1:
            X = X[:, np.newaxis]
        gamma_plus, gamma_minus, theta_plus, theta_minus, weights = self.get_params(
            params=params)
        I = np.dot(X, weights)
        I_ = np.repeat(I, repeats, axis=0)
        Ysamples = sample_from_inputs(
            I_, gamma_plus, gamma_minus, theta_plus, theta_minus)
        return Ysamples.reshape([X.shape[0], repeats, Ysamples.shape[-1]])

    def fit(self, X, Y):
        if X.ndim == 1:
            X = X[:, np.newaxis]
        if Y.ndim == 1:
            Y = Y[:, np.newaxis]
        self.B = X.shape[0]
        self.Nx = X.shape[1]
        self.Ny = Y.shape[1]

        self.gamma_plus = np.ones(self.Ny)
        self.gamma_minus = np.ones(self.Ny)
        self.theta_plus = np.zeros(self.Ny)
        self.theta_minus = np.zeros(self.Ny)
        self.weights = np.zeros([self.Nx, self.Ny])

        params0 = np.concatenate(
            (
                self.gamma_plus,
                self.gamma_minus,
                self.theta_plus,
                self.theta_minus,
                self.weights.flatten()
            ), axis=0)

        to_minimize = partial(self.negative_likelihood, X, Y)
        gradient_to_minimize = partial(self.gradient_negative_likelihood, X, Y)

        result = minimize(to_minimize, params0,
                          jac=gradient_to_minimize, method='L-BFGS-B')
        params_optimal = result.x
        self.gamma_plus, self.gamma_minus, self.theta_plus, self.theta_minus, self.weights = self.get_params(
            params_optimal)
        # print('Fitting Done')
        return result
