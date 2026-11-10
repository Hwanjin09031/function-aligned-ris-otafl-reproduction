"""Conditional component-error model.

Nominal clustered/Rician components are conditional means. Given them, draw
proper independent errors on the direct, UE-RIS, and shared RIS-AP links.
The true cascaded channel retains G_true * h_r_true / sqrt(M).
Effective errors need not be Gaussian; their conditional moments are exact.
Only numpy is required. All variances use E[abs(z)**2] convention.
"""
from dataclasses import dataclass
import numpy as np


def cn(rng, shape, variance):
    if not np.isfinite(variance) or variance < 0:
        raise ValueError('variance must be finite and nonnegative')
    return np.sqrt(variance / 2) * (rng.normal(size=shape) + 1j*rng.normal(size=shape))


@dataclass(frozen=True)
class ComponentCSI:
    h_d_hat: np.ndarray                  # K, Nr
    h_r_hat: np.ndarray                  # K, M
    G_hat: np.ndarray                    # Nr, M; common to all users
    q_d: float
    q_r: float
    q_G: float
    rho: float
    mean_nominal_hr_power: float

    @property
    def C_hat(self):
        return self.G_hat[None, :, :] * self.h_r_hat[:, None, :] / np.sqrt(self.h_r_hat.shape[1])

    def effective_mean(self, theta):
        theta = self._theta(theta)
        return self.h_d_hat + np.einsum('knm,m->kn', self.C_hat, theta)

    def _theta(self, theta):
        theta = np.asarray(theta, dtype=complex)
        if theta.shape != (self.h_r_hat.shape[1],) or not np.isfinite(theta).all():
            raise ValueError('theta has an invalid shape or nonfinite entries')
        return theta

    def cross_covariance(self, theta, phi=None):
        """E[e_k(theta) e_j(phi)^H | full nominal CSI], shape K,K,Nr,Nr.

        Includes the shared RIS-AP error, which correlates users. No fresh
        errors are drawn when evaluating another codeword of the same sample.
        """
        theta = self._theta(theta)
        phi = theta if phi is None else self._theta(phi)
        d = theta * phi.conj()
        K, M = self.h_r_hat.shape
        Nr = self.G_hat.shape[0]
        eye = np.eye(Nr, dtype=complex)
        shared = self.q_G/M * ((self.h_r_hat*d) @ self.h_r_hat.conj().T)
        cov = shared[:, :, None, None] * eye[None, None, :, :]
        diagonal = (self.q_d + self.q_G*self.q_r/M*np.sum(d))*eye
        diagonal += self.q_r/M * ((self.G_hat*d) @ self.G_hat.conj().T)
        cov[np.arange(K), np.arange(K)] += diagonal
        return cov

    def user_covariance(self, theta):
        K = len(self.h_d_hat)
        cov = self.cross_covariance(theta)
        return cov[np.arange(K), np.arange(K)]


@dataclass(frozen=True)
class CascadedChannelSample:
    h_d_true: np.ndarray
    C_true: np.ndarray
    h_d_hat: np.ndarray
    C_hat: np.ndarray
    csi: ComponentCSI


def resample_truth(csi, rng):
    """Keep the entire nominal information fixed; resample true components."""
    G = csi.G_hat + cn(rng, csi.G_hat.shape, csi.q_G)
    hr = csi.h_r_hat + cn(rng, csi.h_r_hat.shape, csi.q_r)
    hd = csi.h_d_hat + cn(rng, csi.h_d_hat.shape, csi.q_d)
    C = G[None, :, :]*hr[:, None, :] / np.sqrt(hr.shape[1])
    return CascadedChannelSample(hd, C, csi.h_d_hat, csi.C_hat, csi)


def generate_clustered_samples(
    rng, environment, n_samples, K, N_r, M,
    angle_jitter_std=0.012, kappa_hr=25.0,
    direct_variance=0.03, csi_error_ratio=0.0
):
    """Legacy-compatible generator with revised CSI semantics.

    rho=csi_error_ratio is error/nominal second moment, averaged over the
    specified nominal distribution, for both direct and cascaded components.
    Split the cascaded error budget equally between the two hops using
    t=sqrt(1+rho)-1, so 2*t+t*t=rho. The actual conditional covariance varies
    with nominal CSI and is not generally sigma_e^2 I.
    """
    rho = float(csi_error_ratio)
    if not np.isfinite(rho) or rho < 0 or direct_variance < 0:
        raise ValueError('uncertainty and direct variance must be nonnegative')
    if environment.G.shape != (N_r, M) or environment.scenario_powers.shape[1] != K:
        raise ValueError('environment dimensions disagree with requested dimensions')
    beta_r = float(np.mean(environment.scenario_powers))
    beta_G = float(np.mean(abs(environment.G)**2))
    t = rho / (np.sqrt(1+rho)+1)  # stable sqrt(1+rho)-1
    q_d, q_r, q_G = rho*direct_variance, t*beta_r, t*beta_G
    output = []
    ris_index = np.arange(M)
    n_scenarios = environment.scenario_frequencies.shape[0]
    for _ in range(n_samples):
        scenario = int(rng.integers(n_scenarios))
        hr = np.empty((K, M), dtype=complex)
        for k in range(K):
            frequency = environment.scenario_frequencies[scenario, k] + rng.normal(0, angle_jitter_std)
            los = np.exp(1j*2*np.pi*frequency*ris_index)
            hr[k] = np.sqrt(environment.scenario_powers[scenario, k]) * (
                np.sqrt(kappa_hr/(1+kappa_hr))*los
                + np.sqrt(1/(1+kappa_hr))*cn(rng, (M,), 1.0))
        csi = ComponentCSI(cn(rng, (K, N_r), direct_variance), hr,
                           np.array(environment.G, copy=True), q_d, q_r, q_G,
                           rho, beta_r)
        output.append(resample_truth(csi, rng))
    return output


def effective_error_covariance(sample, csi_error_ratio=None, theta=None):
    """Return K x Nr x Nr conditional covariance; never reads realized truth.

    Default theta is all ones. For any unit-modulus theta the marginal
    covariance is the same in this component-error model. No-RIS uses zero.
    """
    if csi_error_ratio is not None and not np.isclose(csi_error_ratio, sample.csi.rho, rtol=1e-12, atol=0):
        raise ValueError('requested uncertainty disagrees with sample metadata')
    if theta is None:
        theta = np.ones(sample.C_hat.shape[-1], dtype=complex)
    return sample.csi.user_covariance(theta)


def as_user_covariance(value, K, Nr):
    """Support the old scalar argument and the new per-user matrices."""
    value = np.asarray(value)
    if value.ndim == 0:
        if not np.isfinite(value) or np.iscomplexobj(value) or value < 0:
            raise ValueError('scalar uncertainty must be real and nonnegative')
        return np.broadcast_to(float(value)*np.eye(Nr), (K, Nr, Nr))
    if value.shape != (K, Nr, Nr) or not np.isfinite(value).all():
        raise ValueError('uncertainty must have shape K,Nr,Nr')
    if not np.allclose(value, value.conj().transpose(0, 2, 1), atol=1e-12, rtol=1e-10):
        raise ValueError('uncertainty must be Hermitian')
    if np.min(np.linalg.eigvalsh(value)) < -1e-12:
        raise ValueError('uncertainty must be positive semidefinite')
    return value
