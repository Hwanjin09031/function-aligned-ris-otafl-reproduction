# Conditional component errors and exact effective covariance.
# Legacy FL normalization, payload accounting, AO/Lloyd, rate and GNN
# issues remain. This is NOT a publication-ready simulation runner.
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
# ============================================================
# [2] RIS channel + AirComp core
# ============================================================

from conditional_csi import (CascadedChannelSample, generate_clustered_samples,
    effective_error_covariance, as_user_covariance)

@dataclass
class AirCompSolution:
    objective: float
    b: np.ndarray
    v: np.ndarray

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

def effective_channels(sample, theta, estimated):
    h_d = sample.h_d_hat if estimated else sample.h_d_true
    C = sample.C_hat if estimated else sample.C_true
    return h_d + np.einsum("knm,m->kn",C,theta)

def aircomp_objective(g,b,v,omega,noise_variance,effective_error_var=0.0):
    coefficient = g @ np.conj(v)
    mismatch = b*coefficient - omega
    cov = as_user_covariance(effective_error_var, *g.shape)
    projected = np.einsum("a,kab,b->k", v.conj(), cov, v).real
    uncertainty = np.sum(np.abs(b)**2 * projected)
    return float(np.sum(np.abs(mismatch)**2) + noise_variance*np.linalg.norm(v)**2 + uncertainty)

def optimize_aircomp(
    g_design, omega, pmax, noise_variance,
    effective_error_var=0.0, max_iter=30, tol=1e-8, rng=None
):
    K,N_r = g_design.shape
    cov = as_user_covariance(effective_error_var, K, N_r)
    if rng is None:
        rng=np.random.default_rng(0)

    v=np.sum(np.conj(omega)[:,None]*g_design,axis=0)
    if np.linalg.norm(v)<1e-12:
        v=cn(rng,(N_r,))
    v/=max(np.linalg.norm(v),1e-12)

    b=np.zeros(K,dtype=np.complex128)
    previous=np.inf

    for _ in range(max_iter):
        v_norm_sq=float(np.linalg.norm(v)**2)
        for k in range(K):
            a_k=np.vdot(v,g_design[k])
            den=np.abs(a_k)**2+float(np.vdot(v, cov[k]@v).real)
            candidate=omega[k]*np.conj(a_k)/max(den,1e-15)
            radius=np.sqrt(pmax[k])
            if np.abs(candidate)>radius:
                candidate=radius*candidate/np.abs(candidate)
            b[k]=candidate

        R=noise_variance*np.eye(N_r,dtype=np.complex128)
        r=np.zeros(N_r,dtype=np.complex128)
        for k in range(K):
            R += np.abs(b[k])**2*(np.outer(g_design[k],np.conj(g_design[k])) + cov[k])
            r += b[k]*np.conj(omega[k])*g_design[k]
        v=np.linalg.pinv(R)@r

        current=aircomp_objective(g_design,b,v,omega,noise_variance,effective_error_var)
        if np.isfinite(previous) and abs(previous-current)<=tol*max(1.0,previous):
            break
        previous=current

    return AirCompSolution(
        aircomp_objective(g_design,b,v,omega,noise_variance,effective_error_var),
        b.copy(),v.copy()
    )

def evaluate_true_mse(sample,theta,solution,omega,noise_variance):
    g_true=effective_channels(sample,theta,estimated=False)
    return aircomp_objective(g_true,solution.b,solution.v,omega,noise_variance,0.0)

def design_solution_for_theta(
    sample,theta,rng,omega,pmax,noise_variance,
    effective_error_var,aircomp_iter
):
    return optimize_aircomp(
        effective_channels(sample,theta,estimated=True),
        omega,pmax,noise_variance,
        effective_error_var=effective_error_var,
        max_iter=aircomp_iter,rng=rng
    )

def optimize_ris_full_csi_ao(
    sample,rng,omega,pmax,noise_variance,
    design_from_true_csi,robust_effective_error_var,
    phase_bits,phase_grid_size=16,ao_iter=8,
    phase_sweeps=2,aircomp_iter=25,restarts=2
):
    M=sample.C_true.shape[-1]
    alphabet = phase_alphabet(phase_bits) if phase_bits is not None else np.exp(1j*2*np.pi*np.arange(phase_grid_size)/phase_grid_size)

    best_theta=None
    best_solution=None
    best_design_objective=np.inf

    for _ in range(restarts):
        theta=rng.choice(alphabet,size=M,replace=True)
        previous=np.inf

        for _ in range(ao_iter):
            g_design=effective_channels(sample,theta,estimated=not design_from_true_csi)
            solution=optimize_aircomp(
                g_design,omega,pmax,noise_variance,
                effective_error_var=robust_effective_error_var,
                max_iter=aircomp_iter,rng=rng
            )

            for _ in range(phase_sweeps):
                changed=False
                for m in range(M):
                    current_phase=theta[m]
                    best_phase=current_phase
                    best_value=np.inf
                    for candidate in alphabet:
                        trial=theta.copy()
                        trial[m]=candidate
                        g_trial=effective_channels(sample,trial,estimated=not design_from_true_csi)
                        value=aircomp_objective(
                            g_trial,solution.b,solution.v,omega,
                            noise_variance,robust_effective_error_var
                        )
                        if value<best_value:
                            best_value=value
                            best_phase=candidate
                    if best_phase!=current_phase:
                        theta[m]=best_phase
                        changed=True
                if not changed:
                    break

            g_design=effective_channels(sample,theta,estimated=not design_from_true_csi)
            solution=optimize_aircomp(
                g_design,omega,pmax,noise_variance,
                effective_error_var=robust_effective_error_var,
                max_iter=aircomp_iter,rng=rng
            )
            current=solution.objective
            if np.isfinite(previous) and abs(previous-current)<=1e-6*max(1.0,previous):
                break
            previous=current

        if solution.objective<best_design_objective:
            best_design_objective=solution.objective
            best_theta=theta.copy()
            best_solution=solution

    if best_theta is None or best_solution is None:
        raise RuntimeError("Full-CSI AO failed")

    true_mse=evaluate_true_mse(sample,best_theta,best_solution,omega,noise_variance)
    return best_theta,best_solution,float(true_mse)

def select_codeword_by_function(
    sample,codebook,rng,omega,pmax,noise_variance,
    effective_error_var,aircomp_iter
):
    best_index=-1
    best_solution=None
    best_design_value=np.inf
    for index,theta in enumerate(codebook):
        solution=design_solution_for_theta(
            sample,theta,rng,omega,pmax,noise_variance,
            effective_error_var,aircomp_iter
        )
        if solution.objective<best_design_value:
            best_design_value=solution.objective
            best_index=index
            best_solution=solution
    if best_solution is None:
        raise RuntimeError("Codeword selection failed")
    true_mse=evaluate_true_mse(sample,codebook[best_index],best_solution,omega,noise_variance)
    return best_index,best_solution,float(true_mse)

def zf_sum_rate(g,noise_variance,total_power):
    K,N_r=g.shape
    if N_r<K:
        gain=np.sum(np.abs(g)**2,axis=1)
        power=np.full(K,total_power/K)
        return float(np.sum(np.log2(1+power*gain/noise_variance)))

    H=np.conj(g)
    gram_inv=np.linalg.pinv(H@np.conj(H.T)+1e-10*np.eye(K))
    noise_levels=noise_variance*np.maximum(np.real(np.diag(gram_inv)),1e-12)
    order=np.argsort(noise_levels)
    sorted_n=noise_levels[order]
    water_level=0.0
    for active in range(1,K+1):
        candidate=(total_power+np.sum(sorted_n[:active]))/active
        if active==K or candidate<=sorted_n[active]:
            water_level=candidate
            break
    power_sorted=np.maximum(water_level-sorted_n,0)
    if power_sorted.sum()>0:
        power_sorted*=total_power/power_sorted.sum()
    power=np.empty_like(power_sorted)
    power[order]=power_sorted
    return float(np.sum(np.log2(1+power/noise_levels)))

def optimize_ea_rate_codeword(sample,rng,phase_bits,noise_variance,total_power,ao_iter=8):
    alphabet=phase_alphabet(phase_bits)
    M=sample.C_true.shape[-1]
    theta=rng.choice(alphabet,size=M,replace=True)
    for _ in range(ao_iter):
        changed=False
        for m in range(M):
            current=theta[m]
            best=current
            best_rate=-np.inf
            for candidate in alphabet:
                trial=theta.copy()
                trial[m]=candidate
                rate=zf_sum_rate(effective_channels(sample,trial,estimated=True),noise_variance,total_power)
                if rate>best_rate:
                    best_rate=rate
                    best=candidate
            if best!=current:
                theta[m]=best
                changed=True
        if not changed:
            break
    return theta

def generate_ea_rate_codebook(training_samples,rng,L,phase_bits,noise_variance,total_power,ao_iter=8):
    if len(training_samples)<L:
        raise ValueError("At least L training samples are required")
    selected=np.linspace(0,len(training_samples)-1,L,dtype=int)
    return np.asarray([
        optimize_ea_rate_codeword(training_samples[int(i)],rng,phase_bits,noise_variance,total_power,ao_iter)
        for i in selected
    ])

def select_codeword_by_rate(sample,codebook,noise_variance,total_power):
    rates=[
        zf_sum_rate(effective_channels(sample,theta,estimated=True),noise_variance,total_power)
        for theta in codebook
    ]
    return int(np.argmax(rates))

def fixed_transceiver_quadratic(sample,solution,omega,estimated):
    C=sample.C_hat if estimated else sample.C_true
    h_d=sample.h_d_hat if estimated else sample.h_d_true
    K,_,M=C.shape
    Q=np.zeros((M,M),dtype=np.complex128)
    q=np.zeros(M,dtype=np.complex128)
    for k in range(K):
        u=np.conj(C[k].T)@solution.v
        d=np.vdot(solution.v,h_d[k])
        Q+=np.abs(solution.b[k])**2*np.outer(u,np.conj(u))
        q+=np.conj(solution.b[k])*(omega[k]-solution.b[k]*d)*u
    return Q,q

def quadratic_value(theta,Q,q):
    return float(np.real(np.vdot(theta,Q@theta)-2*np.vdot(theta,q)))

def discrete_refine_quadratic(theta,Q,q,alphabet,sweeps):
    theta=theta.copy()
    for _ in range(sweeps):
        changed=False
        for m in range(theta.size):
            best_phase=theta[m]
            best_value=quadratic_value(theta,Q,q)
            for candidate in alphabet:
                if candidate==theta[m]:
                    continue
                trial=theta.copy()
                trial[m]=candidate
                value=quadratic_value(trial,Q,q)
                if value<best_value-1e-12:
                    best_value=value
                    best_phase=candidate
            if best_phase!=theta[m]:
                theta[m]=best_phase
                changed=True
        if not changed:
            break
    return theta

def optimize_single_function_codeword(
    sample,rng,omega,pmax,noise_variance,phase_bits,
    aircomp_iter,outer_iter=5,phase_sweeps=2,restarts=2
):
    alphabet=phase_alphabet(phase_bits)
    M=sample.C_true.shape[-1]
    best_theta=None
    best_value=np.inf
    for _ in range(restarts):
        theta=rng.choice(alphabet,size=M,replace=True)
        for _ in range(outer_iter):
            solution=optimize_aircomp(
                effective_channels(sample,theta,estimated=False),
                omega,pmax,noise_variance,max_iter=aircomp_iter,rng=rng
            )
            Q,q=fixed_transceiver_quadratic(sample,solution,omega,estimated=False)
            updated=discrete_refine_quadratic(theta,Q,q,alphabet,phase_sweeps)
            if np.array_equal(updated,theta):
                break
            theta=updated
        solution=optimize_aircomp(
            effective_channels(sample,theta,estimated=False),
            omega,pmax,noise_variance,max_iter=aircomp_iter,rng=rng
        )
        if solution.objective<best_value:
            best_value=solution.objective
            best_theta=theta.copy()
    if best_theta is None:
        raise RuntimeError("Function codeword initialization failed")
    return best_theta

def function_lloyd_codebook(
    training_samples,rng,L,omega,pmax,noise_variance,phase_bits,
    lloyd_iter=6,centroid_iter=3,phase_sweeps=2,
    aircomp_iter=15,n_initializations=2
):
    alphabet=phase_alphabet(phase_bits)
    M=training_samples[0].C_true.shape[-1]
    best_codebook=None
    best_distortion=np.inf

    for _ in range(n_initializations):
        selected_indices=rng.choice(len(training_samples),size=L,replace=False)
        codebook=np.asarray([
            optimize_single_function_codeword(
                training_samples[int(index)],rng,omega,pmax,noise_variance,
                phase_bits,aircomp_iter
            )
            for index in selected_indices
        ])

        for _ in range(lloyd_iter):
            distortion=np.empty((len(training_samples),L))
            for n,sample in enumerate(training_samples):
                for ell,theta in enumerate(codebook):
                    solution=optimize_aircomp(
                        effective_channels(sample,theta,estimated=False),
                        omega,pmax,noise_variance,max_iter=aircomp_iter,rng=rng
                    )
                    distortion[n,ell]=solution.objective

            assignment=np.argmin(distortion,axis=1)
            for ell in range(L):
                indices=np.where(assignment==ell)[0]
                if len(indices)==0:
                    worst=int(np.argmax(np.min(distortion,axis=1)))
                    codebook[ell]=optimize_single_function_codeword(
                        training_samples[worst],rng,omega,pmax,noise_variance,
                        phase_bits,aircomp_iter
                    )
                    continue

                theta=codebook[ell].copy()
                for _ in range(centroid_iter):
                    Q=np.zeros((M,M),dtype=np.complex128)
                    q=np.zeros(M,dtype=np.complex128)
                    for index in indices:
                        sample=training_samples[int(index)]
                        solution=optimize_aircomp(
                            effective_channels(sample,theta,estimated=False),
                            omega,pmax,noise_variance,max_iter=aircomp_iter,rng=rng
                        )
                        Qn,qn=fixed_transceiver_quadratic(sample,solution,omega,estimated=False)
                        Q+=Qn
                        q+=qn
                    Q/=len(indices)
                    q/=len(indices)
                    updated=discrete_refine_quadratic(theta,Q,q,alphabet,phase_sweeps)
                    if np.array_equal(updated,theta):
                        break
                    theta=updated
                codebook[ell]=theta

        total=0.0
        for sample in training_samples:
            values=[]
            for theta in codebook:
                solution=optimize_aircomp(
                    effective_channels(sample,theta,estimated=False),
                    omega,pmax,noise_variance,max_iter=aircomp_iter,rng=rng
                )
                values.append(solution.objective)
            total+=min(values)
        average=total/len(training_samples)
        if average<best_distortion:
            best_distortion=average
            best_codebook=codebook.copy()

    if best_codebook is None:
        raise RuntimeError("Function Lloyd codebook failed")
    return best_codebook