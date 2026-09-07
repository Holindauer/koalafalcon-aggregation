import sys
sys.path.insert(0, "../../lattice-estimator")
from sage.all import log, PolynomialRing, GF, ceil, sqrt, pi, e
from estimator import SIS
import math

L2_NORM = 2

# Computes the CoreFalcon+ signature standard deviation formula 
def corefalcon_sigma(q, n, alpha, epsilon):
    return (
        (alpha / math.pi)
        * math.sqrt(math.log(4 * n * (1 + 1/epsilon)) / 2)
        * math.sqrt(q)
    )


# Comptutes the CoreFalcon+ signature norm bound formula
def corefalcon_beta(n, tau, sigma):
    return tau * sigma * math.sqrt(2 * n)

# computes the CoreFalcon+ error term formula
def corefalcon_epsilon(Q_s, lambda_security):
    return 1 / math.sqrt(lambda_security * Q_s)

# encapsulates concrete parameters for CoreFalcon+ 512 w/ KoalaBear prime modulus
def parameters_512():
    lambda_security = 128
    n = 512 
    q = 2**31 - 2**24 + 1
    Q_s = 2**64
    C_s = 2**64 + 2**50
    Q_H = 2**96
    alpha = 1.17
    k = 416
    tau = 1.1
    epsilon = corefalcon_epsilon(Q_s, lambda_security)
    sigma = corefalcon_sigma(q, n, alpha, epsilon)
    beta = corefalcon_beta(n, tau, sigma)

    return {
        "lambda_security": lambda_security,
        "n": n,
        "q": q,
        "Q_s": Q_s,
        "C_s": C_s,
        "Q_H": Q_H,
        "alpha": alpha,
        "tau": tau,
        "epsilon": epsilon,
        "sigma": sigma,
        "beta": beta,
        "k": k,
    }

# Computes the concrete parameters for CoreFalcon+ 1024 w/ KoalaBear prime modulus
def parameters_1024():
    lambda_security = 256
    n = 1024
    q = 2**31 - 2**24 + 1
    Q_s = 2**64
    C_s = 2**64 + 2**50
    Q_H = 2**96
    alpha = 1.17
    tau = 1.1
    k = 416
    epsilon = corefalcon_epsilon(Q_s, lambda_security)
    sigma = corefalcon_sigma(q, n, alpha, epsilon)
    beta = corefalcon_beta(n, tau, sigma)

    return {
        "lambda_security": lambda_security,
        "n": n,
        "q": q,
        "Q_s": Q_s,
        "Q_H": Q_H,
        "C_s": C_s,
        "alpha": alpha,
        "tau": tau,
        "epsilon": epsilon,
        "sigma": sigma,
        "beta": beta,
        "k": k,
    }

# Estimates the full UF-CMA security bound for CoreFalcon+
def estimate_uf_cma_security(parameters: dict):
    epsilon = parameters["epsilon"]
    lambda_security = parameters["lambda_security"]
    Q_s = parameters["Q_s"]
    Q_H = parameters["Q_H"]
    C_s = parameters["C_s"]
    k = parameters["k"]
    alpha = parameters["alpha"]
    tau = parameters["tau"]
    beta = parameters["beta"]
    n = parameters["n"]
    q = parameters["q"]

    print("CoreFalcon+ UF-CMA security estimate: n=", n, "q=", q, "lambda=", lambda_security)

    # estimate advantage of an adversary against t-R-ISIS using 
    # lattice-estimator library under the core-SVP methodology
    sis_params = SIS.Parameters(
            n=n,
            q=q,
            length_bound=beta,
            norm=L2_NORM,
            m=2 * n,
            tag="SIS-Koala",
        )
    sis_estimate = SIS.estimate.rough(sis_params)
    sis_bits = log(sis_estimate["lattice"]["rop"], 2)

    # sampler success probability
    p_PreSmp_beta = 1 - (
        ((1 + epsilon) / (1 - epsilon))
        * (sqrt((e ** (1 - tau**2)) * tau**2) ** (2*n))
        * (1 + 2 * epsilon)
    )

    # order a_p for renyi divergence r_p
    a_p = (
        lambda_security * (epsilon ** 2) * log(4)
        + sqrt(
            8 * C_s * lambda_security * (epsilon ** 2) * log(2)
            + lambda_security**2 * (epsilon ** 4) * (log(4) ** 2)
        )
    ) / (4 * C_s * (epsilon ** 2))

    # renyi divergence r_p
    r_p = 1 + 2 * a_p * (epsilon ** 2)

    # order a_u for renyi divergence r_u
    a_u = (
        lambda_security * ((epsilon / (1 - epsilon)) ** 2) * log(4)
        + sqrt(
            8 * C_s * lambda_security * ((epsilon / (1 - epsilon)) ** 2) * log(2)
            + lambda_security**2 * ((epsilon / (1 - epsilon)) ** 4) * (log(4) ** 2)
        )
    ) / (4 * C_s * ((epsilon / (1 - epsilon)) ** 2))

    # renyi divergence r_u
    r_u = 1 + 2 * a_u * ((epsilon / (1 - epsilon)) ** 2)


    # full computational term of the UF-CMA bound, this includes 
    # the renyi-divergence factors on top of the SIS estimate
    computational_log2 = (
        C_s * log(r_u, 2)
        +
        (
            C_s * log(r_p, 2)
            -
            sis_bits
        ) * ((a_p - 1) / a_p)
    ) * ((a_u - 1) / a_u)


    binomial_log2 = (
        -2 * C_s * (p_PreSmp_beta - Q_s / C_s) ** 2
    ) / log(2)

    # Validity checks for the sampler/binomial bound
    assert 0 < epsilon < 1 / 4, (
        f"epsilon must be in (0, 1/4), got epsilon={epsilon.n()}"
    )

    assert Q_s <= C_s * p_PreSmp_beta, (
        "Hoeffding lower-tail condition failed: "
        f"Q_s={Q_s} > C_s * p_PreSmp_beta={(C_s * p_PreSmp_beta).n()}"
    )

    assert binomial_log2 <= -lambda_security, (
        "Binomial/Hoeffding failure probability is too large: "
        f"log2 bound={binomial_log2.n()} > -lambda={-lambda_security}"
    )

    salt_log2 = log(C_s, 2) + log(Q_H + C_s, 2) - k

    max_log2 = max([computational_log2, binomial_log2, salt_log2])

    uf_cma_bound_log2 = max_log2 + log(
        2 ** (computational_log2 - max_log2)
        + 2 ** (binomial_log2 - max_log2)
        + 2 ** (salt_log2 - max_log2),
        2,
    )

    uf_cma_bits = -uf_cma_bound_log2

    print()
    print(f"UF-CMA bound log2: {uf_cma_bound_log2.n()}")
    print(f"UF-CMA bits: {uf_cma_bits.n()}")
    print(f"computational_log2: {computational_log2.n()}")
    print(f"binomial_log2: {binomial_log2.n()}")
    print(f"salt_log2: {salt_log2.n()}")
    print(f"SIS estimate: {sis_estimate['lattice']['rop'].n()}")
    print(f"SIS bits: {sis_bits.n()}")
    print(f"a_p: {a_p.n()}")
    print(f"a_u: {a_u.n()}")
    print(f"r_p: {r_p.n()}")
    print(f"r_u: {r_u.n()}")
    print(f"p_PreSmp_beta: {p_PreSmp_beta.n()}")
    print() 

if __name__ == "__main__":

    # setup parameters for both instances (targeting NIST level 1, 5 security)
    params_512 = parameters_512()
    params_1024 = parameters_1024()

    print()
    print("params 512: \n", params_512)
    print()
    print("params 1024: \n", params_1024)
    print()

    # estimate UF-CMA security
    estimate_uf_cma_security(params_512)
    estimate_uf_cma_security(params_1024)