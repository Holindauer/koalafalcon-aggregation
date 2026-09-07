//! KoalaFalcon V3 challenge hash \(H(\mathrm{pk}, r, m)\).
//!
//! ## Transcript
//!
//! Each component is tagged and length-delimited. Tags are encoded as
//! `enc_u64(|tag|) || tag bytes`. Every `u64` is `enc_u64(x)` = four 16-bit
//! limbs. Byte strings use `enc_u64(|bytes|) || one field per byte`.
//!
//! ```text
//! tag("suite")        || enc_u64(|SUITE|) || SUITE bytes
//! tag("public-key")   || enc_u64(N)       || h[0], …, h[N-1]
//! tag("salt")         || enc_u64(|salt|)  || salt bytes
//! tag("message")      || enc_u64(|msg|)   || message bytes
//! ```
//!
//! ## Challenge polynomial
//!
//! \[
//! (c_0,\ldots,c_{N-1}) \leftarrow \mathrm{PoseidonXOF}_{\mathsf{suite}}(\mathsf{pk}, r, m)
//! \]
//!
//! Clone the cached prefix (suite + public key), absorb salt and message,
//! finalize the Pad10 sponge, then squeeze exactly `N` field elements in
//! natural XOF order:
//!
//! ```text
//! state[0], …, state[RATE-1], permute, state[0], …
//! ```
//!
//! Map each canonical field value directly into `KoalaBearRing<N>`. There is
//! no seed-and-counter expansion phase.
//!
//! ## Parameter suites
//!
//! | `N`   | Rate | Capacity | Generic sponge capacity (approx.) |
//! |-------|------|----------|-----------------------------------|
//! | 512   | 15   | 9        | ~139 bits                         |
//! | 1024  | 7    | 17       | ~263 bits                         |
//!
//! Rate 7 for `N=1024` addresses the generic sponge capacity target; it does
//! **not** alone prove 256-bit security of Poseidon1 (permutation analysis is
//! separate).

use crate::algebra::{CyclotomicRing, Field, KoalaBear, KoalaBearRing};
use crate::profile;
use crate::utils::Error;
use p3_field::PrimeCharacteristicRing;
use p3_koala_bear::KoalaBear as P3KoalaBear;

use super::poseidon_xof::{
    PoseidonAbsorbState, PoseidonXof, RATE_512, RATE_1024, encode_u64_limbs,
};

const SUITE_512: &[u8] = b"KOALAFALCON_CHALLENGE_V3_N512_POSEIDON1_W24_R15";
const SUITE_1024: &[u8] = b"KOALAFALCON_CHALLENGE_V3_N1024_POSEIDON1_W24_R7";

const TAG_SUITE: &[u8] = b"suite";
const TAG_PUBLIC_KEY: &[u8] = b"public-key";
const TAG_SALT: &[u8] = b"salt";
const TAG_MESSAGE: &[u8] = b"message";

#[derive(Clone, Debug)]
enum ChallengePrefixState {
    N512(PoseidonAbsorbState<RATE_512>),
    N1024(PoseidonAbsorbState<RATE_1024>),
}

/// Cached mid-transcript state after absorbing suite + public key.
#[derive(Clone, Debug)]
pub(crate) struct ChallengePrefix {
    state: ChallengePrefixState,
}

struct Transcript512(PoseidonAbsorbState<RATE_512>);
struct Transcript1024(PoseidonAbsorbState<RATE_1024>);

impl Transcript512 {
    fn new() -> Self {
        Self(PoseidonAbsorbState::new())
    }

    fn from_state(state: PoseidonAbsorbState<RATE_512>) -> Self {
        Self(state)
    }

    fn absorb_tag(&mut self, tag: &[u8]) {
        for limb in encode_u64_limbs(tag.len() as u64) {
            self.0.absorb_field(limb);
        }
        for &b in tag {
            self.0.absorb_field(P3KoalaBear::new(b as u32));
        }
    }

    fn absorb_u64(&mut self, value: u64) {
        self.0.absorb_fields(&encode_u64_limbs(value));
    }

    fn absorb_bytes_component(&mut self, tag: &[u8], bytes: &[u8]) {
        self.absorb_tag(tag);
        self.absorb_u64(bytes.len() as u64);
        for &b in bytes {
            self.0.absorb_field(P3KoalaBear::new(b as u32));
        }
    }

    fn absorb_ring_component<const N: usize>(&mut self, tag: &[u8], ring: &KoalaBearRing<N>) {
        self.absorb_tag(tag);
        self.absorb_u64(N as u64);
        for c in ring.coeffs() {
            self.0.absorb_field(c.0);
        }
    }

    fn into_state(self) -> PoseidonAbsorbState<RATE_512> {
        self.0
    }
}

impl Transcript1024 {
    fn new() -> Self {
        Self(PoseidonAbsorbState::new())
    }

    fn from_state(state: PoseidonAbsorbState<RATE_1024>) -> Self {
        Self(state)
    }

    fn absorb_tag(&mut self, tag: &[u8]) {
        for limb in encode_u64_limbs(tag.len() as u64) {
            self.0.absorb_field(limb);
        }
        for &b in tag {
            self.0.absorb_field(P3KoalaBear::new(b as u32));
        }
    }

    fn absorb_u64(&mut self, value: u64) {
        self.0.absorb_fields(&encode_u64_limbs(value));
    }

    fn absorb_bytes_component(&mut self, tag: &[u8], bytes: &[u8]) {
        self.absorb_tag(tag);
        self.absorb_u64(bytes.len() as u64);
        for &b in bytes {
            self.0.absorb_field(P3KoalaBear::new(b as u32));
        }
    }

    fn absorb_ring_component<const N: usize>(&mut self, tag: &[u8], ring: &KoalaBearRing<N>) {
        self.absorb_tag(tag);
        self.absorb_u64(N as u64);
        for c in ring.coeffs() {
            self.0.absorb_field(c.0);
        }
    }

    fn into_state(self) -> PoseidonAbsorbState<RATE_1024> {
        self.0
    }
}

fn squeeze_ring<const N: usize, const RATE: usize>(
    xof: &mut PoseidonXof<RATE>,
) -> KoalaBearRing<N> {
    let mut p3 = [P3KoalaBear::ZERO; N];
    xof.squeeze_fields(&mut p3);
    let mut coeffs = [KoalaBear::ZERO; N];
    for (c, p) in coeffs.iter_mut().zip(p3.iter()) {
        *c = KoalaBear(*p);
    }
    KoalaBearRing::from_coeffs(&coeffs)
}

impl ChallengePrefix {
    pub(crate) fn from_h<const N: usize>(h: &KoalaBearRing<N>) -> Result<Self, Error> {
        match N {
            512 => {
                let mut t = Transcript512::new();
                t.absorb_bytes_component(TAG_SUITE, SUITE_512);
                t.absorb_ring_component(TAG_PUBLIC_KEY, h);
                Ok(Self {
                    state: ChallengePrefixState::N512(t.into_state()),
                })
            }
            1024 => {
                let mut t = Transcript1024::new();
                t.absorb_bytes_component(TAG_SUITE, SUITE_1024);
                t.absorb_ring_component(TAG_PUBLIC_KEY, h);
                Ok(Self {
                    state: ChallengePrefixState::N1024(t.into_state()),
                })
            }
            _ => Err(Error::UnsupportedRingDimension(N)),
        }
    }

    pub(crate) fn hash_to_point<const N: usize>(
        &self,
        salt: &[u8],
        message: &[u8],
    ) -> Result<KoalaBearRing<N>, Error> {
        profile!("hash_to_point");
        match (&self.state, N) {
            (ChallengePrefixState::N512(state), 512) => {
                let mut t = Transcript512::from_state(state.clone());
                t.absorb_bytes_component(TAG_SALT, salt);
                t.absorb_bytes_component(TAG_MESSAGE, message);
                let mut xof = t.into_state().finalize();
                Ok(squeeze_ring::<N, RATE_512>(&mut xof))
            }
            (ChallengePrefixState::N1024(state), 1024) => {
                let mut t = Transcript1024::from_state(state.clone());
                t.absorb_bytes_component(TAG_SALT, salt);
                t.absorb_bytes_component(TAG_MESSAGE, message);
                let mut xof = t.into_state().finalize();
                Ok(squeeze_ring::<N, RATE_1024>(&mut xof))
            }
            _ => Err(Error::UnsupportedRingDimension(N)),
        }
    }
}

/// One-shot V3 challenge (tests and vector generation).
#[cfg(test)]
pub(crate) fn hash_to_point<const N: usize>(
    h: &KoalaBearRing<N>,
    salt: &[u8],
    message: &[u8],
) -> Result<KoalaBearRing<N>, Error> {
    match N {
        512 => {
            let mut t = Transcript512::new();
            t.absorb_bytes_component(TAG_SUITE, SUITE_512);
            t.absorb_ring_component(TAG_PUBLIC_KEY, h);
            t.absorb_bytes_component(TAG_SALT, salt);
            t.absorb_bytes_component(TAG_MESSAGE, message);
            let mut xof = t.into_state().finalize();
            Ok(squeeze_ring::<N, RATE_512>(&mut xof))
        }
        1024 => {
            let mut t = Transcript1024::new();
            t.absorb_bytes_component(TAG_SUITE, SUITE_1024);
            t.absorb_ring_component(TAG_PUBLIC_KEY, h);
            t.absorb_bytes_component(TAG_SALT, salt);
            t.absorb_bytes_component(TAG_MESSAGE, message);
            let mut xof = t.into_state().finalize();
            Ok(squeeze_ring::<N, RATE_1024>(&mut xof))
        }
        _ => Err(Error::UnsupportedRingDimension(N)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::KoalaBear;

    fn ring_from_u32<const N: usize>(f: impl Fn(usize) -> u32) -> KoalaBearRing<N> {
        let coeffs: [KoalaBear; N] = std::array::from_fn(|i| KoalaBear::new(f(i)));
        KoalaBearRing::from_coeffs(&coeffs)
    }

    fn squeeze_prefix<const N: usize>(
        prefix: &ChallengePrefix,
        salt: &[u8],
        message: &[u8],
        count: usize,
    ) -> Vec<u32> {
        let ring = prefix.hash_to_point::<N>(salt, message).unwrap();
        ring.coeffs()
            .iter()
            .take(count)
            .map(|c| c.as_canonical_u32())
            .collect()
    }

    fn squeeze_one_shot<const N: usize>(
        h: &KoalaBearRing<N>,
        salt: &[u8],
        message: &[u8],
        count: usize,
    ) -> Vec<u32> {
        let ring = hash_to_point(h, salt, message).unwrap();
        ring.coeffs()
            .iter()
            .take(count)
            .map(|c| c.as_canonical_u32())
            .collect()
    }

    #[test]
    fn prefix_matches_one_shot_512() {
        let h = ring_from_u32::<512>(|_| 1);
        let prefix = ChallengePrefix::from_h(&h).unwrap();
        let salt = b"test-salt";
        let message = b"hello";
        assert_eq!(
            squeeze_prefix::<512>(&prefix, salt, message, 512),
            squeeze_one_shot::<512>(&h, salt, message, 512)
        );
    }

    #[test]
    fn prefix_matches_one_shot_1024() {
        let h = ring_from_u32::<1024>(|_| 2);
        let prefix = ChallengePrefix::from_h(&h).unwrap();
        let salt = b"x";
        let message = b"y";
        assert_eq!(
            squeeze_prefix::<1024>(&prefix, salt, message, 1024),
            squeeze_one_shot::<1024>(&h, salt, message, 1024)
        );
    }

    #[test]
    fn transcript_ambiguity_salt_message() {
        let h = ring_from_u32::<512>(|_| 3);
        let a = hash_to_point(&h, b"a", b"b").unwrap();
        let b = hash_to_point(&h, b"ab", b"").unwrap();
        assert_ne!(a, b);

        let c = hash_to_point(&h, b"", b"ab").unwrap();
        let d = hash_to_point(&h, b"a", b"b").unwrap();
        assert_ne!(c, d);
    }

    #[test]
    fn suite_separation_512_vs_1024() {
        let h512 = ring_from_u32::<512>(|i| (i as u32).wrapping_mul(3));
        let h1024 = ring_from_u32::<1024>(|i| (i as u32).wrapping_mul(3));
        let salt = b"shared";
        let message = b"payload";
        let p512 =
            squeeze_prefix::<512>(&ChallengePrefix::from_h(&h512).unwrap(), salt, message, 16);
        let p1024 =
            squeeze_prefix::<1024>(&ChallengePrefix::from_h(&h1024).unwrap(), salt, message, 16);
        assert_ne!(p512, p1024);
    }

    #[test]
    fn wrong_dimension_rejected() {
        let h512 = ring_from_u32::<512>(|_| 1);
        let prefix = ChallengePrefix::from_h(&h512).unwrap();
        assert!(matches!(
            prefix.hash_to_point::<1024>(b"s", b"m"),
            Err(Error::UnsupportedRingDimension(1024))
        ));
        assert!(matches!(
            ChallengePrefix::from_h(&h512)
                .unwrap()
                .hash_to_point::<256>(b"s", b"m"),
            Err(Error::UnsupportedRingDimension(256))
        ));
    }

    #[test]
    fn fixed_vector_512() {
        let h = ring_from_u32::<512>(|i| (i as u32).wrapping_mul(12345).wrapping_add(7));
        let ring = hash_to_point(&h, b"vec-salt-512", b"vec-message").unwrap();
        let first16: Vec<u32> = ring
            .coeffs()
            .iter()
            .take(16)
            .map(|c| c.as_canonical_u32())
            .collect();
        assert_eq!(
            first16,
            [
                88_904_561,
                1_078_292_760,
                1_625_224_472,
                369_434_128,
                1_600_669_485,
                1_323_736_207,
                1_873_003_549,
                1_739_362_997,
                1_415_926_166,
                1_917_671_940,
                1_884_310_331,
                1_897_126_628,
                175_141_304,
                1_530_361_406,
                1_613_869_184,
                801_935_587,
            ]
        );
    }

    #[test]
    fn fixed_vector_1024() {
        let h = ring_from_u32::<1024>(|i| (i as u32).wrapping_mul(54321).wrapping_add(11));
        let ring = hash_to_point(&h, b"vec-salt-1024", b"vec-message-1024").unwrap();
        let first16: Vec<u32> = ring
            .coeffs()
            .iter()
            .take(16)
            .map(|c| c.as_canonical_u32())
            .collect();
        assert_eq!(
            first16,
            [
                1_887_758_501,
                764_982_074,
                332_558_559,
                1_616_012_332,
                1_168_192_893,
                818_805_855,
                1_308_441_211,
                988_123_929,
                496_444_375,
                487_452_502,
                1_339_859_099,
                1_824_119_959,
                1_757_881_193,
                191_523_736,
                2_002_410_439,
                818_233_516,
            ]
        );
    }
}
