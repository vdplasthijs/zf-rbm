import numpy as np
from scipy.optimize import minimize
from functools import partial


def energy(Y, gamma, theta):
    return 0.5 * gamma * Y**2 + theta * Y


def cgf_from_inputs(I, gamma, theta):
    return 0.5 * (I - theta)**2 / gamma + 0.5 * np.log(2 * np.pi / gamma)


def mean_from_inputs(I, gamma, theta):
    return (I - theta) / gamma


def mean2_from_inputs(I, gamma, theta):
    return ((I - theta) / gamma)**2 + 1 / gamma


def sample_from_inputs(I, gamma, theta):
    return (I - theta) / gamma + 1 / np.sqrt(gamma) * np.random.randn(*I.shape)


class Gaussian_Regression:
    def __init__(self, Nx=None, Ny=None):
        if Nx is not None:
            Nx = self.Nx
        if Ny is not None:
            Ny = self.Ny
        return

    def get_params(self, params=None):
        if params is not None:
            gamma = params[:self.Ny]
            theta = params[self.Ny:2 * self.Ny]
            weights = params[2 * self.Ny:].reshape([self.Nx, self.Ny])
        else:
            gamma = self.gamma
            theta = self.theta
            weights = self.weights
        return gamma, theta, weights

    def negative_likelihood(self, X, Y, params=None):
        if X.ndim == 1:
            X = X[:, np.newaxis]
        if Y.ndim == 1:
            Y = Y[:, np.newaxis]
        gamma, theta, weights = self.get_params(
            params=params)
        I = np.dot(X, weights)
        E = energy(Y, gamma, theta) - I * Y
        log_partition = cgf_from_inputs(
            I, gamma, theta)
        return -(-E - log_partition).sum(1).mean(0)

    def gradient_negative_likelihood(self, X, Y, params=None):
        if X.ndim == 1:
            X = X[:, np.newaxis]
        if Y.ndim == 1:
            Y = Y[:, np.newaxis]
        gamma, theta, weights = self.get_params(
            params=params)

        I = np.dot(X, weights)
        mu = mean_from_inputs(I, gamma, theta)
        mu2 = mean2_from_inputs(I, gamma, theta)

        gradient_I = Y - mu
        gradient_gamma = -0.5 * (Y**2 - mu2).mean(0)
        gradient_theta = - gradient_I.mean(0)
        gradient_w = np.dot(X.T, gradient_I) / X.shape[0]
        gradient_params = np.concatenate((
            gradient_gamma,
            gradient_theta,
            gradient_w.flatten()
        ), axis=0)
        return -gradient_params

    def predict(self, X, params=None):
        if X.ndim == 1:
            X = X[:, np.newaxis]
        gamma, theta, weights = self.get_params(
            params=params)
        I = np.dot(X, weights)
        return mean_from_inputs(I, gamma, theta)

    def predict_samples(self, X, repeats=1, params=None):
        if X.ndim == 1:
            X = X[:, np.newaxis]
        gamma, theta, weights = self.get_params(
            params=params)
        I = np.dot(X, weights)
        I_ = np.repeat(I, repeats, axis=0)
        Ysamples = sample_from_inputs(
            I_, gamma, theta)
        return Ysamples.reshape([X.shape[0], repeats, Ysamples.shape[-1]])

    def fit(self, X, Y):
        if X.ndim == 1:
            X = X[:, np.newaxis]
        if Y.ndim == 1:
            Y = Y[:, np.newaxis]
        self.B = X.shape[0]
        self.Nx = X.shape[1]
        self.Ny = Y.shape[1]

        self.gamma = np.ones(self.Ny)
        self.theta = np.zeros(self.Ny)
        self.weights = np.zeros([self.Nx, self.Ny])

        params0 = np.concatenate(
            (
                self.gamma,
                self.theta,
                self.weights.flatten()
            ), axis=0)

        to_minimize = partial(self.negative_likelihood, X, Y)
        gradient_to_minimize = partial(self.gradient_negative_likelihood, X, Y)

        result = minimize(to_minimize, params0,
                          jac=gradient_to_minimize, method='L-BFGS-B')
        params_optimal = result.x
        self.gamma, self.theta, self.weights = self.get_params(
            params_optimal)
        # print('Fitting Done')
        return result
