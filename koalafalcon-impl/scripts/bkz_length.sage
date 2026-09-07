from math import sqrt, log, pi

# assuming m=2n
def bkz_length_estimate(delta: float, q: int, m: int) -> float:
    return min(q, sqrt(q) * (delta ** m))

# if lattice red can find vectors of len min(q, sqrt(q) * (delta ** m)), 
# then in order to find vectors within the norm bound beta, delta would 
# need to be at most 
#
# assuming beta < q:
# beta = min(q, sqrt(q) * (delta ** m))
# beta / sqrt(q) = (delta ** m)
# (beta / sqrt(q)) ** (1 / m) = delta 
#
# Inversion only applies when beta < q, since bkz_length_estimate is capped by min(q, ...).
def max_delta(beta: float, q: int, m: int) -> float:
    return (beta / sqrt(q)) ** (1 / m)

def bkz_estimate(n: int, q: int, alpha: float, tau: float, security_lambda: int, Q_s: int) -> float:

    m = 2 * n

    epsilon = 1 / sqrt(security_lambda * Q_s)

    s = (
        (alpha / pi)
        * sqrt(log(4 * n * (1 + 1 / epsilon)) / 2)
        * sqrt(q)
    )

    beta = tau * s * sqrt(2 * n)

    md = max_delta(beta, q, m)

    print(f"n = {n}")
    print(f"m = {m}")
    print(f"q = {q}")
    print(f"epsilon = {epsilon}")
    print(f"s = {s}")
    print(f"beta = {beta}")
    print()
    for delta in [1.013, 1.011, 1.005]:
        length = bkz_length_estimate(delta, q, m)
        print(f"root-hermite = {delta}: bkz_length ≈ {length}, finds beta-length vector: {length <= beta}")
    print()
    print("maximum root-hermite to find a beta length vector: ", md)

if __name__ == "__main__":
    # CoreFalcon+ w/ KoalaBear parameters
    n_512 = 512
    n_1024 = 1024
    q = 2**31 - 2**24 + 1
    alpha = 1.17
    tau = 1.1
    lambda_128 = 128
    lambda_256 = 256
    Q_s = 2**64

    bkz_estimate(n_512, q, alpha, tau, lambda_128, Q_s)
    bkz_estimate(n_1024, q, alpha, tau, lambda_256, Q_s)