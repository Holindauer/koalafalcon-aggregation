//! Signing and verification keys and signatures for KoalaFalcon.

#![allow(non_snake_case)]

use crate::algebra::{Field, KoalaBear, KoalaBearRing, PreparedNegacyclicMultiplier};
use crate::falcon::hash::ChallengePrefix;
use crate::falcon::parameters::{KoalaFalconParameters, SALT_LEN};
use crate::falcon::sampling::FourierPreSmp;
use crate::profile;
use crate::utils::Error;

/// Compact NTRU trapdoor for the secret basis (internal to key generation).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PrivateKey<const N: usize> {
    pub f: KoalaBearRing<N>,
    pub g: KoalaBearRing<N>,
    pub F: KoalaBearRing<N>,
    pub G: KoalaBearRing<N>,
}

impl<const N: usize> PrivateKey<N> {
    /// Same q-ary lattice as `pk`: `g ≡ f·h` and `G ≡ F·h` (mod q).
    #[cfg(test)]
    pub fn matches_public(&self, pk: &PublicKey<N>) -> bool {
        self.f * pk.h == self.g && self.F * pk.h == self.G
    }

    pub(crate) fn expand_fourier_presmp(
        &self,
        pk: &PublicKey<N>,
        params: &KoalaFalconParameters,
    ) -> Result<FourierPreSmp<N>, Error> {
        profile!("expand_fourier_presmp");
        debug_assert_eq!(params.n, N);

        let f_i = crate::utils::center_poly_i64(self.f.coeffs());
        let g_i = crate::utils::center_poly_i64(self.g.coeffs());
        let F_i = crate::utils::center_poly_i64(self.F.coeffs());
        let G_i = crate::utils::center_poly_i64(self.G.coeffs());
        let h_i = crate::utils::center_poly_i64(pk.h.coeffs());
        let neg_f: Vec<i64> = f_i.iter().map(|&x| -x).collect();
        let neg_F: Vec<i64> = F_i.iter().map(|&x| -x).collect();

        let sigmin = {
            let bound = params.s / params.target_gs_norm;
            (bound * 0.995).max(1.001)
        };

        FourierPreSmp::from_ntru_i64(&g_i, &neg_f, &G_i, &neg_F, &h_i, params.s, sigmin)
    }
}

/// Public key material from trapdoor generation (internal).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PublicKey<const N: usize> {
    pub h: KoalaBearRing<N>,
}

/// Secret signing key: Fourier PreSmp and cached challenge prefix.
#[derive(Debug, Clone)]
pub struct SigningKey<const N: usize> {
    pub(crate) sampler: FourierPreSmp<N>,
    pub(crate) challenge_prefix: ChallengePrefix,
    pub(crate) beta_squared: u128,
}

impl<const N: usize> SigningKey<N> {
    pub(crate) fn from_trapdoor(
        sk: PrivateKey<N>,
        pk: PublicKey<N>,
        params: &KoalaFalconParameters,
    ) -> Result<Self, Error> {
        profile!("expand_keys");
        let sampler = sk.expand_fourier_presmp(&pk, params)?;
        Ok(Self {
            challenge_prefix: ChallengePrefix::from_h(&pk.h)?,
            beta_squared: params.beta_squared,
            sampler,
        })
    }

    /// Check trapdoor consistency (test helper).
    #[cfg(test)]
    pub(crate) fn is_consistent(&self, vk: &VerifyingKey<N>, sk: &PrivateKey<N>) -> bool {
        sk.matches_public(&PublicKey { h: vk.h })
    }
}

/// Public verification key.
#[derive(Debug, Clone)]
pub struct VerifyingKey<const N: usize> {
    pub h: KoalaBearRing<N>,
    pub(crate) h_mul: PreparedNegacyclicMultiplier<N>,
    pub(crate) challenge_prefix: ChallengePrefix,
    pub(crate) beta_squared: u128,
}

impl<const N: usize> VerifyingKey<N> {
    pub(crate) fn from_public(
        pk: PublicKey<N>,
        params: &KoalaFalconParameters,
    ) -> Result<Self, Error> {
        Ok(Self {
            h: pk.h,
            h_mul: PreparedNegacyclicMultiplier::new(pk.h)?,
            challenge_prefix: ChallengePrefix::from_h(&pk.h)?,
            beta_squared: params.beta_squared,
        })
    }

    /// Build a verification key from a public ring element
    pub fn from_public_key_ring(
        h: KoalaBearRing<N>,
        params: &KoalaFalconParameters,
    ) -> Result<Self, Error> {
        Self::from_public(PublicKey { h }, params)
    }
}

/// CoreFalcon+ signature `σ = (r, s₂)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Signature<const N: usize> {
    pub salt: [u8; SALT_LEN],
    pub s2: KoalaBearRing<N>,
}

impl<const N: usize> Signature<N> {
    pub fn new(salt: [u8; SALT_LEN], s2: KoalaBearRing<N>) -> Self {
        Self { salt, s2 }
    }
}

macro_rules! signature_encoding {
    ($n:expr) => {
        impl Signature<$n> {
            pub const ENCODED_LEN: usize = SALT_LEN + $n * KoalaBear::N_BYTES;

            /// Encode `σ = (r, s₂)` as `r ‖ LE32(s₂[0]) ‖ … ‖ LE32(s₂[N-1])`.
            pub fn to_bytes(&self) -> [u8; Self::ENCODED_LEN] {
                let mut out = [0u8; Self::ENCODED_LEN];
                out[..SALT_LEN].copy_from_slice(&self.salt);
                let mut off = SALT_LEN;
                for coeff in self.s2.coeffs() {
                    out[off..off + KoalaBear::N_BYTES]
                        .copy_from_slice(&coeff.as_canonical_u32().to_le_bytes());
                    off += KoalaBear::N_BYTES;
                }
                out
            }

            /// Decode a canonical signature, rejecting wrong lengths.
            pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
                if bytes.len() != Self::ENCODED_LEN {
                    return Err(Error::InsufficientInputBytes {
                        expected: Self::ENCODED_LEN,
                        got: bytes.len(),
                    });
                }
                let mut salt = [0u8; SALT_LEN];
                salt.copy_from_slice(&bytes[..SALT_LEN]);
                let s2 = KoalaBearRing::from_bytes(&bytes[SALT_LEN..])?;
                Ok(Self { salt, s2 })
            }
        }
    };
}

signature_encoding!(512);
signature_encoding!(1024);
