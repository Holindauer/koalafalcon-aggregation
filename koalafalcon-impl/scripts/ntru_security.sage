import sys
sys.path.insert(0, "../../lattice-estimator")
from sage.all import log
from estimator import SIS
import math

KOALA = 2**31 - 2**24 + 1
GOLDILOCKS = 2**64 - 2**32 + 1
EPS_TINY = 1e-9

def eps_max(q, n):
    return math.log(n) / math.log(q)

def sigma_of(q, n, eps):
    return n * math.sqrt(math.log(8 * n * q)) * q ** (0.5 + eps)

def s_of(q, n, eps):
    return n**1.5 * sigma_of(q, n, eps)

def beta_of(q, n, eps):
    return 2 * s_of(q, n, eps) * math.sqrt(2 * n)

def splits(q, n):
    return (q - 1) % (2 * n) == 0

def growth_ok(q, n, eps):
    return q ** (0.5 - eps) >= n**3.5

def estimate_ntrusign_security(q, n, eps):
    sigma = sigma_of(q, n, eps)
    s = s_of(q, n, eps)
    beta = beta_of(q, n, eps)

    print(f"NTRUSign (Stehle-Steinfeld Cor. 4.1): n={n}, q={q}, eps={eps}")
    print(f"sigma={sigma}, s={s}, beta={beta}")
    print(f"splits={splits(q, n)}, growth_ok={growth_ok(q, n, eps)}")
    print(f"beta / ((q-1)/2) = {beta / ((q - 1) / 2)}")

    if beta >= (q - 1) / 2:
        print("SIS bits: 0 (trivial, beta >= (q-1)/2)")
        print()
        return

    params = SIS.Parameters(
        n=n,
        q=q,
        length_bound=beta,
        norm=2,
        m=2 * n,
        tag="NTRUSign-SS",
    )
    est = SIS.estimate.rough(params)
    sis_bits = log(est["lattice"]["rop"], 2)

    print(f"SIS estimate: {est['lattice']['rop'].n()}")
    print(f"SIS bits: {sis_bits.n()}")
    print()


if __name__ == "__main__":
    for q, name in [(KOALA, "Koala"), (GOLDILOCKS, "Goldilocks")]:
        print()
        print(f"{name}: q={q}")
        print()

        for n in [512, 1024]:
            estimate_ntrusign_security(q, n, EPS_TINY)
            estimate_ntrusign_security(q, n, eps_max(q, n))