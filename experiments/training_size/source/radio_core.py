"""Normalized clustered environment and two-bit phase utilities."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ClusteredEnvironment:
    G: np.ndarray
    scenario_frequencies: np.ndarray
    scenario_powers: np.ndarray

def cn(rng, shape, variance=1.0):
    scale = np.sqrt(max(variance, 0.0) / 2.0)
    return scale * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))

def phase_alphabet(bits):
    if bits < 1:
        raise ValueError("phase bits must be >= 1")
    size = 2**bits
    return np.exp(1j * 2*np.pi*np.arange(size)/size)

def quantize_unit_modulus(theta, bits):
    theta = np.exp(1j*np.angle(theta))
    if bits is None:
        return theta
    alphabet = phase_alphabet(bits)
    phase = np.angle(theta)
    index = np.mod(np.rint(phase*len(alphabet)/(2*np.pi)).astype(int), len(alphabet))
    return alphabet[index]

def create_clustered_environment(rng, K, N_r, M, n_scenarios, kappa_G=20.0):
    ris_index = np.arange(M)
    ap_index = np.arange(N_r)

    f_ap = rng.uniform(-0.25, 0.25)
    f_ris = rng.uniform(-0.25, 0.25)
    a_ap = np.exp(1j*2*np.pi*f_ap*ap_index)/np.sqrt(N_r)
    a_ris = np.exp(1j*2*np.pi*f_ris*ris_index)/np.sqrt(M)
    G_los = np.sqrt(N_r*M)*np.outer(a_ap, np.conj(a_ris))
    G = np.sqrt(kappa_G/(1+kappa_G))*G_los + np.sqrt(1/(1+kappa_G))*cn(rng,(N_r,M))

    base = np.linspace(-0.31,0.31,K)
    scenario_frequencies = np.empty((n_scenarios,K))
    scenario_powers = np.ones((n_scenarios,K))

    for scenario in range(n_scenarios):
        shift = -0.15 + 0.30*scenario/max(1,n_scenarios-1)
        permutation = np.roll(np.arange(K), scenario % K)
        scenario_frequencies[scenario] = np.clip(base[permutation]+shift,-0.47,0.47)
        weak = scenario % K
        scenario_powers[scenario,weak] = 0.35
        scenario_powers[scenario,(weak+1)%K] = 0.65

    return ClusteredEnvironment(G, scenario_frequencies, scenario_powers)

