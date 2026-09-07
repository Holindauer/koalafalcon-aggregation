#![allow(non_snake_case)]
mod hash;
mod keys;
mod parameters;
mod sampling;
mod trapdoor;

pub use keys::{Signature, SigningKey, VerifyingKey};
pub use parameters::{
    FalconParameterSet, KoalaFalconParameters, N512, N1024, SALT_LEN, SecureKoalaFalcon512,
    SecureKoalaFalcon1024,
};

use crate::algebra::KoalaBearRing;
use crate::falcon::parameters::KoalaFalconParameters as Params;
use crate::falcon::trapdoor::TpdGen;
use crate::profile;
use crate::utils::Error;
use crate::utils::{center_poly_i64, i64_to_ring, l1_norm, linf_norm, squared_l2};
use rand::{CryptoRng, Rng};

fn params_for_n<const N: usize>() -> Result<Params, Error> {
    match N {
        512 => Ok(Params::falcon_512()),
        1024 => Ok(Params::falcon_1024()),
        _ => Err(Error::UnsupportedRingDimension(N)),
    }
}

/// Generate a KoalaFalcon signing/verification key pair for degree `N` ∈ {512, 1024}.
pub fn keygen<const N: usize>(
    rng: &mut impl Rng,
) -> Result<(SigningKey<N>, VerifyingKey<N>), Error> {
    keygen_with_params(&params_for_n::<N>()?, rng)
}

/// Generate keys for an explicit parameter set (experimental ladder, custom β/σ, etc.).
pub fn keygen_with_params<const N: usize>(
    params: &KoalaFalconParameters,
    rng: &mut impl Rng,
) -> Result<(SigningKey<N>, VerifyingKey<N>), Error> {
    profile!("keygen");
    if params.n != N {
        return Err(Error::UnsupportedRingDimension(N));
    }
    let tpd = TpdGen::from_params(params).map_err(Error::Trapdoor)?;
    let (sk, pk) = tpd.generate_keys::<N>(rng).map_err(Error::Trapdoor)?;
    let signing_key = SigningKey::from_trapdoor(sk, pk, params)?;
    let verifying_key = VerifyingKey::from_public(pk, params)?;
    Ok((signing_key, verifying_key))
}

/// Generate KoalaFalcon-512 keys.
pub fn keygen_512(rng: &mut impl Rng) -> Result<(SigningKey<512>, VerifyingKey<512>), Error> {
    keygen::<512>(rng)
}

/// Generate KoalaFalcon-1024 keys.
pub fn keygen_1024(rng: &mut impl Rng) -> Result<(SigningKey<1024>, VerifyingKey<1024>), Error> {
    keygen::<1024>(rng)
}

impl<const N: usize> SigningKey<N> {
    /// CoreFalcon+ `Sgn+`.
    pub fn sign(
        &self,
        message: &[u8],
        rng: &mut (impl Rng + CryptoRng),
    ) -> Result<Signature<N>, Error> {
        profile!("sign");

        const MAX_ATTEMPTS: usize = 10_000;

        for _ in 0..MAX_ATTEMPTS {
            let mut salt = [0u8; SALT_LEN];
            rng.fill_bytes(&mut salt);

            let c_ring: KoalaBearRing<N> =
                self.challenge_prefix.hash_to_point::<N>(&salt, message)?;
            let c = center_poly_i64(c_ring.coeffs());
            let Ok((s1, s2)) = self.sampler.sample(&c, rng) else {
                continue;
            };

            let norm_sq = squared_l2(&s1) + squared_l2(&s2);
            if norm_sq < self.beta_squared {
                return Ok(Signature::new(salt, i64_to_ring::<N>(&s2)));
            }
        }

        Err(Error::SigningFailed)
    }

    pub fn sign_with_challenge(
        &self,
        t: &KoalaBearRing<N>,
        rng: &mut (impl Rng + CryptoRng),
    ) -> Result<Signature<N>, Error> {
        profile!("sign_with_challenge");

        const MAX_ATTEMPTS: usize = 10_000;

        for _ in 0..MAX_ATTEMPTS {
            let c = center_poly_i64(t.coeffs());
            let Ok((s1, s2)) = self.sampler.sample(&c, rng) else {
                continue;
            };

            let norm_sq = squared_l2(&s1) + squared_l2(&s2);
            if norm_sq < self.beta_squared {
                return Ok(Signature::new([0u8; SALT_LEN], i64_to_ring::<N>(&s2)));
            }
        }

        Err(Error::SigningFailed)
    }
}

impl<const N: usize> VerifyingKey<N> {
    fn signature_preimage_centered(
        &self,
        message: &[u8],
        signature: &Signature<N>,
    ) -> Result<(Vec<i64>, Vec<i64>), Error> {
        let c: KoalaBearRing<N> = self
            .challenge_prefix
            .hash_to_point::<N>(&signature.salt, message)?;
        let s2h = self.h_mul.mul(&signature.s2);
        let s1 = c - s2h;

        Ok((
            center_poly_i64(s1.coeffs()),
            center_poly_i64(signature.s2.coeffs()),
        ))
    }

    /// Exact `L1`, squared `L2`, and `L∞` norms of the signature preimage `(s₁, s₂)`.
    pub fn signature_norms(
        &self,
        message: &[u8],
        signature: &Signature<N>,
    ) -> Result<SignatureNorms, Error> {
        let (s1_c, s2_c) = self.signature_preimage_centered(message, signature)?;
        Ok(SignatureNorms {
            l1: l1_norm(&s1_c) + l1_norm(&s2_c),
            l2_squared: squared_l2(&s1_c) + squared_l2(&s2_c),
            linf: linf_norm(&s1_c).max(linf_norm(&s2_c)),
        })
    }

    /// Exact squared Euclidean norm `‖s‖²` of the signature preimage `(s₁, s₂)`.
    pub fn signature_squared_norm(
        &self,
        message: &[u8],
        signature: &Signature<N>,
    ) -> Result<u128, Error> {
        Ok(self.signature_norms(message, signature)?.l2_squared)
    }

    /// Hash `(salt, message)` to the Falcon challenge polynomial `t`.
    pub fn challenge_point(&self, salt: &[u8], message: &[u8]) -> Result<KoalaBearRing<N>, Error> {
        self.challenge_prefix.hash_to_point::<N>(salt, message)
    }

    /// CoreFalcon+ `Ver`.
    pub fn verify(&self, message: &[u8], signature: &Signature<N>) -> Result<bool, Error> {
        profile!("verify");
        let norm_sq = self.signature_squared_norm(message, signature)?;
        Ok(norm_sq < self.beta_squared)
    }
}

/// Exact coefficient norms for a verified signature preimage `(s₁, s₂)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SignatureNorms {
    pub l1: u128,
    pub l2_squared: u128,
    pub linf: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::{
        CyclotomicRing, Field, KoalaBear, KoalaBearRing, PreparedNegacyclicMultiplier,
    };
    use crate::falcon::hash::hash_to_point;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn run_keygen_consistent<const N: usize>() {
        let mut rng = StdRng::from_os_rng();
        let params = params_for_n::<N>().expect("params");
        let tpd = TpdGen::from_params(&params).expect("tpd");
        let (sk, pk) = tpd.generate_keys::<N>(&mut rng).expect("keygen");
        let vk = VerifyingKey::from_public(pk, &params).expect("vk");
        let signing_key = SigningKey::from_trapdoor(sk, pk, &params).expect("sk");
        assert!(signing_key.is_consistent(&vk, &sk));
    }

    fn run_sign_verify_roundtrip<const N: usize>() {
        let mut rng = StdRng::from_os_rng();
        let (sk, vk) = keygen::<N>(&mut rng).expect("keygen");
        let msg = b"koala-falcon-sign-test";
        let sig = sk.sign(msg, &mut rng).expect("sign");
        assert!(vk.verify(msg, &sig).expect("verify"));
        assert!(!vk.verify(b"other", &sig).expect("verify fail"));
    }

    #[test]
    fn norm_boundary_exact_integer() {
        let params = Params::falcon_512();
        let beta_squared = params.beta_squared;
        let accepts = |norm_sq: u128, bound: u128| norm_sq < bound;
        assert!(accepts(beta_squared - 1, beta_squared));
        assert!(!accepts(beta_squared, beta_squared));
        assert!(!accepts(beta_squared + 1, beta_squared));
    }

    #[test]
    fn keygen_rejects_unsupported_n() {
        let mut rng = StdRng::from_os_rng();
        assert!(matches!(
            keygen::<256>(&mut rng),
            Err(Error::UnsupportedRingDimension(256))
        ));
    }

    #[test]
    fn keygen_consistent_512() {
        run_keygen_consistent::<512>();
    }

    #[test]
    fn keygen_consistent_1024() {
        run_keygen_consistent::<1024>();
    }

    #[test]
    fn challenge_prefix_and_prepared_h_match_oneshot() {
        let mut rng = StdRng::from_os_rng();
        let (_sk, vk) = keygen::<512>(&mut rng).expect("keygen");
        let salt = b"cached-salt";
        let msg = b"cached-msg";
        let a = hash_to_point(&vk.h, salt, msg).unwrap();
        let b = vk.challenge_prefix.hash_to_point::<512>(salt, msg).unwrap();
        assert_eq!(a, b);
        let h_mul = PreparedNegacyclicMultiplier::new(vk.h).unwrap();
        let probe = KoalaBearRing::ONE;
        assert_eq!(h_mul.mul(&probe), probe * vk.h);
    }

    #[test]
    fn expand_and_sample_stats_512() {
        let mut rng = StdRng::from_os_rng();
        let params = Params::falcon_512();
        let tpd = TpdGen::from_params(&params).expect("tpd");
        let (sk, pk) = tpd.generate_keys::<512>(&mut rng).expect("keygen");
        let signing_key = SigningKey::<512>::from_trapdoor(sk, pk, &params).expect("expand");
        let vk = VerifyingKey::<512>::from_public(pk, &params).expect("vk");

        let c_ring = hash_to_point::<512>(&vk.h, b"salt", b"m").unwrap();
        let c = center_poly_i64(c_ring.coeffs());
        let (s1, s2) = signing_key.sampler.sample(&c, &mut rng).expect("sample");
        let norm = ((squared_l2(&s1) + squared_l2(&s2)) as f64).sqrt();
        let beta = params.beta;
        println!("Fourier PreSmp ‖s‖₂={norm} β={beta}");

        let s1_r = i64_to_ring::<512>(&s1);
        let s2_r = i64_to_ring::<512>(&s2);
        let lhs = s1_r + s2_r * vk.h;
        assert_eq!(lhs, c_ring, "preimage must satisfy s1 + s2·h = c");
    }

    #[test]
    fn sign_verify_roundtrip_512() {
        run_sign_verify_roundtrip::<512>();
    }

    #[test]
    fn sign_verify_roundtrip_1024() {
        run_sign_verify_roundtrip::<1024>();
    }

    #[test]
    fn verify_detects_tampering() {
        let mut rng = StdRng::from_os_rng();
        let (sk, vk) = keygen::<512>(&mut rng).expect("keygen");
        let msg = b"m0";
        let mut sig = sk.sign(msg, &mut rng).expect("sign");
        assert!(vk.verify(msg, &sig).unwrap());

        assert!(!vk.verify(b"XX", &sig).unwrap());

        sig.salt[0] ^= 0x5A;
        assert!(!vk.verify(msg, &sig).unwrap());

        let mut sig = sk.sign(msg, &mut rng).expect("sign");
        let mut coeffs = *sig.s2.coeffs();
        coeffs[0] += KoalaBear::ONE;
        sig.s2 = KoalaBearRing::from_coeffs(&coeffs);
        assert!(!vk.verify(msg, &sig).unwrap());
    }

    #[test]
    fn signature_encoding_roundtrip_and_rejects_wrong_length() {
        let mut rng = StdRng::from_os_rng();
        let (sk, vk) = keygen::<512>(&mut rng).expect("keygen");
        let sig = sk.sign(b"wire", &mut rng).expect("sign");
        let bytes = sig.to_bytes();
        assert_eq!(bytes.len(), Signature::<512>::ENCODED_LEN);
        let decoded = Signature::<512>::from_bytes(&bytes).expect("decode");
        assert_eq!(decoded, sig);
        assert!(vk.verify(b"wire", &decoded).unwrap());

        let short = &bytes[..bytes.len() - 1];
        assert!(matches!(
            Signature::<512>::from_bytes(short),
            Err(Error::InsufficientInputBytes { .. })
        ));
        let mut long = bytes.to_vec();
        long.push(0);
        assert!(matches!(
            Signature::<512>::from_bytes(&long),
            Err(Error::InsufficientInputBytes { .. })
        ));
    }

    #[test]
    fn fourier_tree_builds_512() {
        let mut rng = StdRng::from_os_rng();
        let params = Params::falcon_512();
        let tpd = TpdGen::from_params(&params).expect("tpd");
        let (trap_sk, pk) = tpd.generate_keys::<512>(&mut rng).expect("keygen");
        let signing_key = SigningKey::<512>::from_trapdoor(trap_sk, pk, &params).expect("expand");
        let vk = VerifyingKey::<512>::from_public(pk, &params).expect("vk");
        let fp = &signing_key.sampler;
        let stored = fp.tree_stored_complex_values();
        println!(
            "Fourier tree L10 complexes={stored} (n log n ~ {})",
            512 * 9
        );
        assert!(
            stored < 512 * 512 / 4,
            "tree should be far below dense O(n²)"
        );

        let beta_squared = params.beta_squared;
        let mut ok = 0usize;
        let mut fail = 0usize;
        let mut short = 0usize;
        for i in 0..32 {
            let salt = format!("salt{i}");
            let c_ring = hash_to_point::<512>(&vk.h, salt.as_bytes(), b"m").unwrap();
            let c = center_poly_i64(c_ring.coeffs());
            match signing_key.sampler.sample(&c, &mut rng) {
                Ok((s1, s2)) => {
                    let s1_r = i64_to_ring::<512>(&s1);
                    let s2_r = i64_to_ring::<512>(&s2);
                    assert_eq!(s1_r + s2_r * vk.h, c_ring);
                    let n2 = squared_l2(&s1) + squared_l2(&s2);
                    if n2 < beta_squared {
                        short += 1;
                    }
                    ok += 1;
                }
                Err(_) => fail += 1,
            }
        }
        println!("Fourier exact-preimage ok={ok} fail={fail}/32 short_under_β={short}");
        assert_eq!(fail, 0, "exact ring reconstruction must succeed");
        assert_eq!(ok, 32);
        assert!(short >= 1, "at least one sample should meet β");

        let msg = b"fourier-sign-verify";
        let sig = signing_key.sign(msg, &mut rng).expect("fourier sign");
        assert!(vk.verify(msg, &sig).expect("fourier verify"));
    }
}
