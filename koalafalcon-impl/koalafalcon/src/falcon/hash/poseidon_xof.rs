//! Resumable Poseidon1 sponge absorption and XOF squeezing over KoalaBear.
//!
//! Width is 24. Rate is a const-generic parameter (15 for ~128-bit capacity
//! targets, 7 for ~256-bit capacity targets). This module does **not** know
//! about Falcon transcripts, public keys, or ring types.
//!
//! Pad10 semantics match Plonky3 `Pad10Sponge`:
//! - partial final absorb block: rate-domain padding marker at `rate_filled`,
//!   clear the rest of the rate, permute;
//! - exact-full final absorb block: padding marker in the first capacity
//!   position, permute;
//! - a full absorb block is not permuted until more input follows or
//!   finalization occurs.
//!
//! After finalization, the XOF yields `state[0..RATE]`, then permutes and
//! repeats. Squeezed fields are **not** re-hashed.

use p3_koala_bear::{KoalaBear as P3KoalaBear, default_koalabear_poseidon1_24};
use p3_symmetric::{Increment, Permutation};
use std::sync::OnceLock;

/// Poseidon1 state width.
pub(crate) const WIDTH: usize = 24;

/// Rate for the KoalaFalcon-512 challenge suite (capacity = 9).
pub(crate) const RATE_512: usize = 15;

/// Rate for the KoalaFalcon-1024 challenge suite (capacity = 17).
///
/// Capacity 17 KoalaBear elements gives a generic sponge-capacity bound of
/// roughly \(17 \cdot \log_2 q / 2 \approx 263\) bits. This addresses the
/// sponge capacity requirement only; it does **not** by itself establish
/// 256-bit security of the Poseidon1 permutation (round counts and algebraic
/// attacks require a separate review).
pub(crate) const RATE_1024: usize = 7;

type PoseidonPerm = p3_koala_bear::Poseidon1KoalaBear<WIDTH>;

fn permutation() -> &'static PoseidonPerm {
    static PERM: OnceLock<PoseidonPerm> = OnceLock::new();
    PERM.get_or_init(default_koalabear_poseidon1_24)
}

fn pad10_derangement() -> Increment<P3KoalaBear> {
    Increment(P3KoalaBear::new(1))
}

/// Mid-absorption Poseidon state (overwrite Pad10 sponge, no padding applied yet).
#[derive(Clone, Debug)]
pub(crate) struct PoseidonAbsorbState<const RATE: usize> {
    state: [P3KoalaBear; WIDTH],
    rate_filled: usize,
}

impl<const RATE: usize> PoseidonAbsorbState<RATE> {
    pub(crate) fn new() -> Self {
        Self {
            state: [P3KoalaBear::new(0); WIDTH],
            rate_filled: 0,
        }
    }

    pub(crate) fn absorb_field(&mut self, value: P3KoalaBear) {
        if self.rate_filled == RATE {
            permutation().permute_mut(&mut self.state);
            self.rate_filled = 0;
        }
        self.state[self.rate_filled] = value;
        self.rate_filled += 1;
    }

    pub(crate) fn absorb_fields(&mut self, values: &[P3KoalaBear]) {
        for &v in values {
            self.absorb_field(v);
        }
    }

    pub(crate) fn finalize(self) -> PoseidonXof<RATE> {
        let derange = pad10_derangement();
        let mut state = self.state;
        if self.rate_filled == RATE {
            state[RATE] = derange.permute(state[RATE]);
        } else {
            state[self.rate_filled] = derange.permute(P3KoalaBear::new(0));
            for slot in state.iter_mut().take(RATE).skip(self.rate_filled + 1) {
                *slot = P3KoalaBear::new(0);
            }
        }
        permutation().permute_mut(&mut state);
        PoseidonXof {
            state,
            squeeze_pos: 0,
        }
    }
}

impl<const RATE: usize> Default for PoseidonAbsorbState<RATE> {
    fn default() -> Self {
        Self::new()
    }
}

/// Finalized Poseidon sponge used as an XOF.
#[derive(Clone, Debug)]
pub(crate) struct PoseidonXof<const RATE: usize> {
    state: [P3KoalaBear; WIDTH],
    squeeze_pos: usize,
}

impl<const RATE: usize> PoseidonXof<RATE> {
    pub(crate) fn squeeze_field(&mut self) -> P3KoalaBear {
        if self.squeeze_pos == RATE {
            permutation().permute_mut(&mut self.state);
            self.squeeze_pos = 0;
        }
        let out = self.state[self.squeeze_pos];
        self.squeeze_pos += 1;
        out
    }

    pub(crate) fn squeeze_fields(&mut self, output: &mut [P3KoalaBear]) {
        for slot in output.iter_mut() {
            *slot = self.squeeze_field();
        }
    }
}

/// Encode a `u64` as four 16-bit limbs (injective below the KoalaBear modulus).
pub(crate) fn encode_u64_limbs(value: u64) -> [P3KoalaBear; 4] {
    [
        P3KoalaBear::new((value & 0xffff) as u32),
        P3KoalaBear::new(((value >> 16) & 0xffff) as u32),
        P3KoalaBear::new(((value >> 32) & 0xffff) as u32),
        P3KoalaBear::new(((value >> 48) & 0xffff) as u32),
    ]
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use p3_field::PrimeField32;
    use p3_symmetric::{CryptographicHasher, Pad10Sponge};
    use rand::{Rng, SeedableRng, rngs::StdRng};

    fn reference_first_block<const RATE: usize>(input: &[P3KoalaBear]) -> [P3KoalaBear; RATE] {
        let sponge: Pad10Sponge<_, _, _, WIDTH, RATE, RATE> =
            Pad10Sponge::new(default_koalabear_poseidon1_24(), pad10_derangement());
        sponge.hash_slice(input)
    }

    fn custom_first_block<const RATE: usize>(input: &[P3KoalaBear]) -> [P3KoalaBear; RATE] {
        let mut absorb = PoseidonAbsorbState::<RATE>::new();
        absorb.absorb_fields(input);
        let mut xof = absorb.finalize();
        let mut out = [P3KoalaBear::new(0); RATE];
        xof.squeeze_fields(&mut out);
        out
    }

    fn custom_squeeze<const RATE: usize>(input: &[P3KoalaBear], count: usize) -> Vec<P3KoalaBear> {
        let mut absorb = PoseidonAbsorbState::<RATE>::new();
        absorb.absorb_fields(input);
        let mut xof = absorb.finalize();
        let mut out = vec![P3KoalaBear::new(0); count];
        xof.squeeze_fields(&mut out);
        out
    }

    #[test]
    fn first_block_matches_pad10_sponge_rate15() {
        for len in [
            0,
            1,
            RATE_512 - 1,
            RATE_512,
            RATE_512 + 1,
            2 * RATE_512 - 1,
            2 * RATE_512,
            2 * RATE_512 + 1,
            3 * RATE_512,
        ] {
            let input: Vec<_> = (0..len)
                .map(|i| P3KoalaBear::new((i as u32).wrapping_mul(17).wrapping_add(3)))
                .collect();
            let got = custom_first_block::<RATE_512>(&input);
            let expect = reference_first_block::<RATE_512>(&input);
            assert_eq!(got, expect, "len={len}");
        }
    }

    #[test]
    fn first_block_matches_pad10_sponge_rate7() {
        for len in [
            0,
            1,
            RATE_1024 - 1,
            RATE_1024,
            RATE_1024 + 1,
            2 * RATE_1024 - 1,
            2 * RATE_1024,
            2 * RATE_1024 + 1,
            3 * RATE_1024,
        ] {
            let input: Vec<_> = (0..len)
                .map(|i| P3KoalaBear::new((i as u32).wrapping_mul(31).wrapping_add(5)))
                .collect();
            let got = custom_first_block::<RATE_1024>(&input);
            let expect = reference_first_block::<RATE_1024>(&input);
            assert_eq!(got, expect, "len={len}");
        }
    }

    #[test]
    fn every_prefix_split_matches_one_shot_rate15() {
        let input: Vec<_> = (0..40u32)
            .map(|i| P3KoalaBear::new(i.wrapping_mul(13).wrapping_add(7)))
            .collect();
        let full = custom_squeeze::<RATE_512>(&input, 32);
        for split in 0..=input.len() {
            let mut absorb = PoseidonAbsorbState::<RATE_512>::new();
            absorb.absorb_fields(&input[..split]);
            let mut resumed = absorb.clone();
            resumed.absorb_fields(&input[split..]);
            let mut xof = resumed.finalize();
            let got: Vec<_> = (0..32).map(|_| xof.squeeze_field()).collect();
            assert_eq!(got, full, "split={split}");
        }
    }

    #[test]
    fn every_prefix_split_matches_one_shot_rate7() {
        let input: Vec<_> = (0..25u32)
            .map(|i| P3KoalaBear::new(i.wrapping_mul(11).wrapping_add(2)))
            .collect();
        let full = custom_squeeze::<RATE_1024>(&input, 32);
        for split in 0..=input.len() {
            let mut absorb = PoseidonAbsorbState::<RATE_1024>::new();
            absorb.absorb_fields(&input[..split]);
            let mut resumed = absorb.clone();
            resumed.absorb_fields(&input[split..]);
            let mut xof = resumed.finalize();
            let got: Vec<_> = (0..32).map(|_| xof.squeeze_field()).collect();
            assert_eq!(got, full, "split={split}");
        }
    }

    #[test]
    fn random_chunking_matches_one_shot() {
        let mut rng = StdRng::from_os_rng();
        let input: Vec<_> = (0..100)
            .map(|_| P3KoalaBear::new(rng.random_range(0..1_000_000)))
            .collect();
        let full = custom_squeeze::<RATE_512>(&input, 64);

        let mut absorb = PoseidonAbsorbState::<RATE_512>::new();
        let mut pos = 0usize;
        while pos < input.len() {
            let chunk = rng.random_range(1..=7);
            let end = (pos + chunk).min(input.len());
            absorb.absorb_fields(&input[pos..end]);
            pos = end;
        }
        let mut xof = absorb.finalize();
        let got: Vec<_> = (0..64).map(|_| xof.squeeze_field()).collect();
        assert_eq!(got, full);
    }

    #[test]
    fn xof_prefix_consistency() {
        let input: Vec<_> = (0..50u32).map(|i| P3KoalaBear::new(i + 1)).collect();
        let short = custom_squeeze::<RATE_512>(&input, 32);
        let long = custom_squeeze::<RATE_512>(&input, 128);
        assert_eq!(short, long[..32]);
    }

    #[test]
    fn encode_u64_limbs_distinct() {
        use crate::algebra::KOALA_BEAR_PRIME;
        let q = KOALA_BEAR_PRIME as u64;
        let values = [
            0u64,
            1,
            65535,
            65536,
            q - 1,
            q,
            u32::MAX as u64,
            u32::MAX as u64 + 1,
            u64::MAX,
        ];
        let mut seen = std::collections::HashSet::new();
        for v in values {
            let limbs = encode_u64_limbs(v);
            let key: [u32; 4] = [
                limbs[0].as_canonical_u32(),
                limbs[1].as_canonical_u32(),
                limbs[2].as_canonical_u32(),
                limbs[3].as_canonical_u32(),
            ];
            assert!(seen.insert(key), "duplicate encoding for {v}");
        }
    }

    #[test]
    fn poseidon1_plonky3_test_vectors() {
        use p3_koala_bear::default_koalabear_poseidon1_16;
        use p3_symmetric::Permutation as _;

        let perm16 = default_koalabear_poseidon1_16();
        let mut input16 =
            P3KoalaBear::new_array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
        let expected16 = P3KoalaBear::new_array([
            610090613, 935319874, 1893335292, 796792199, 356405232, 552237741, 55134556,
            1215104204, 1823723405, 1133298033, 1780633798, 1453946561, 710069176, 1128629550,
            1917333254, 1175481618,
        ]);
        perm16.permute_mut(&mut input16);
        assert_eq!(input16, expected16);

        let perm24 = default_koalabear_poseidon1_24();
        let mut input24 = P3KoalaBear::new_array([
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
        ]);
        let expected24 = P3KoalaBear::new_array([
            511672087, 215882318, 237782537, 740528428, 712760904, 54615367, 751514671, 110231969,
            1905276435, 992525666, 918312360, 18628693, 749929200, 1916418953, 691276896,
            1112901727, 1163558623, 882867603, 673396520, 1480278156, 1402044758, 1693467175,
            1766273044, 433841551,
        ]);
        perm24.permute_mut(&mut input24);
        assert_eq!(input24, expected24);
    }
}
