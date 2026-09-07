#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(dead_code)]
#![allow(clippy::all)]

// Adapted from Pornin rust-fn-dsa fn-dsa-kgen/ntru.rs (public domain).

use super::fxp::*;
use super::mp31::*;
use super::poly_mp::*;
use super::vect::*;
use super::zint31::*;

// ========================================================================
// Solving the NTRU equation
// ========================================================================

// Check that (f,g) has an acceptable orthogonalized norm.
// If this function returns false, then the (f,g) pair should be
// rejected.
// tmp min size: 2.5*n
pub(crate) fn check_ortho_norm(logn: u32, f: &[i32], g: &[i32], tmp: &mut [FXR]) -> bool {
    let n = 1usize << logn;
    let (fx, tmp) = tmp.split_at_mut(n);
    let (gx, rt3) = tmp.split_at_mut(n);
    vect_to_fxr(logn, fx, f);
    vect_to_fxr(logn, gx, g);
    vect_FFT(logn, fx);
    vect_FFT(logn, gx);
    vect_invnorm_fft(logn, rt3, fx, gx, 0);
    vect_adj_fft(logn, fx);
    vect_adj_fft(logn, gx);
    vect_mul_realconst(logn, fx, FXR::from_i32(Q as i32));
    vect_mul_realconst(logn, gx, FXR::from_i32(Q as i32));
    vect_mul_selfadj_fft(logn, fx, rt3);
    vect_mul_selfadj_fft(logn, gx, rt3);
    vect_iFFT(logn, fx);
    vect_iFFT(logn, gx);
    let mut sn = FXR::ZERO;
    for i in 0..n {
        sn += fx[i].sqr() + gx[i].sqr();
    }
    sn < FXR::from_u64_scaled32(72251709809335)
}

const Q: u32 = crate::algebra::KOALA_BEAR_PRIME;

// At recursion depth d, with:
//   slen = MOD_SMALL_BL[d]
//   llen = MOD_LARGE_BL[d]
//   tlen = MOD_SMALL_BL[d + 1]
// then:
//   (f, g) at this level use slen words for each coefficient
//   (F', G') from deeper level use tlen words for each coefficient
//   unreduced (F, G) at this level use llen words for each coefficient
//   output (F, G) use slen words for each coefficient
// Depth 1 uses 4 limbs: KoalaBear Babai residuals are ~90 bits. Deeper levels
// are ~×3 vs Falcon-12289 for larger resultants. Depth 0 outputs 1 limb.
const MOD_SMALL_BL: [usize; 11] = [1, 6, 6, 9, 12, 24, 42, 81, 159, 312, 621];
const MOD_LARGE_BL: [usize; 10] = [18, 9, 9, 18, 33, 63, 120, 234, 465, 924];

// Minimum depth for which intermediate (f,g) values are saved.
const MIN_SAVE_FG: [u32; 11] = [0, 0, 1, 2, 2, 2, 2, 2, 2, 3, 3];

// When log(n) >= MIN_LOGN_FGNTT, we use the NTT to subtract (k*f,k*g)
// from (F,G) during the reduction.
const MIN_LOGN_FGNTT: u32 = 4;

// Number of top words to consider during Babai reduction.
// Falcon: [1,1,2,2,2,3,3,4,5,7]. KoalaBear limb tables are ~3× Falcon’s, so
// keep a similar fraction of the top words (must stay ≤ MOD_SMALL_BL[depth]).
const WORD_WIN: [usize; 10] = [1, 2, 6, 6, 6, 9, 9, 12, 15, 21];

// Number of bits gained per each round of reduction.
// Falcon uses 13/11 for logn 9/10; with KoalaBear-sized limbs that claim is
// optimistic (Babai then drops high limbs too early and stops short of `slen`).
const REDUCE_BITS: [u32; 11] = [16, 16, 16, 16, 16, 16, 16, 16, 16, 4, 3];

// Given polynomials f and g (modulo X^n+1 with n = 2^logn), find
// polynomials F and G such that:
//    -127 <= F[i], G[i] <= +127   for all i in [0, n-1]
//    f*G - g*F = q  mod X^n+1     (with q = 12289)
// Returned value is true on success, false on error. If the function does
// not succeed, then the contents of F and G are not modified.
// All four slices f, g, F and G must have length exactly 2^logn.
// tmp_u32 min size: 6*n
// tmp_fxr min size: 2.5*n
pub(crate) fn solve_NTRU(
    logn: u32,
    f: &[i32],
    g: &[i32],
    F: &mut [i32],
    G: &mut [i32],
    tmp_u32: &mut [u32],
    tmp_fxr: &mut [FXR],
) -> bool {
    match solve_NTRU_staged(logn, f, g, F, G, tmp_u32, tmp_fxr) {
        Ok(()) => true,
        Err(_stage) => false,
    }
}

#[cfg(test)]
pub(crate) fn solve_NTRU_fail_stage(
    logn: u32,
    f: &[i32],
    g: &[i32],
    F: &mut [i32],
    G: &mut [i32],
    tmp_u32: &mut [u32],
    tmp_fxr: &mut [FXR],
) -> Option<&'static str> {
    match solve_NTRU_staged(logn, f, g, F, G, tmp_u32, tmp_fxr) {
        Ok(()) => None,
        Err(stage) => Some(stage),
    }
}

fn solve_NTRU_staged(
    logn: u32,
    f: &[i32],
    g: &[i32],
    F: &mut [i32],
    G: &mut [i32],
    tmp_u32: &mut [u32],
    tmp_fxr: &mut [FXR],
) -> Result<(), &'static str> {
    assert!(1 <= logn && logn <= 10);
    let n = 1usize << logn;
    assert!(f.len() == n && g.len() == n);
    assert!(F.len() == n && G.len() == n);

    if let Err(e) = solve_NTRU_deepest_staged(logn, f, g, tmp_u32) {
        return Err(e);
    }
    // Depth 0 uses the same Babai path: KoalaBear keeps multi-limb (F,G) after
    // depth 1 (MOD_SMALL_BL[1] > 1), so the Falcon single-word depth0 helper
    // cannot be used as-is.
    for depth in (0..logn).rev() {
        match solve_NTRU_intermediate(logn, f, g, depth, tmp_u32, tmp_fxr) {
            Ok(()) => {}
            Err(why) => {
                return Err(match (depth, why) {
                    (0, InterFail::PreBabaiEq) => "d0_pre_babai_eq",
                    (0, InterFail::PostBabaiEq) => "d0_post_babai_eq",
                    (8, InterFail::PreBabaiEq) => "d8_pre_babai_eq",
                    (8, InterFail::PostBabaiEq) => "d8_post_babai_eq",
                    (7, InterFail::PreBabaiEq) => "d7_pre_babai_eq",
                    (7, InterFail::PostBabaiEq) => "d7_post_babai_eq",
                    (d, InterFail::PreBabaiEq) if d <= 9 => "intermediate_pre_eq",
                    (d, InterFail::PostBabaiEq) if d <= 9 => "intermediate_post_eq",
                    _ => "intermediate",
                });
            }
        }
    }

    for i in 0..(2 * n) {
        // Intermediate Babai leaves 31-bit signed limbs (sign in bit 30).
        let z = (tmp_u32[i] | ((tmp_u32[i] & 0x40000000) << 1)) as i32;
        if z < -((1 << 24) - 1) || z > ((1 << 24) - 1) {
            return Err("coeff_bound");
        }
    }

    for i in 0..n {
        F[i] = (tmp_u32[i] | ((tmp_u32[i] & 0x40000000) << 1)) as i32;
        G[i] = (tmp_u32[i + n] | ((tmp_u32[i + n] & 0x40000000) << 1)) as i32;
    }
    Ok(())
}

// Solving the NTRU equation, deepest level.
// This computes the integers F and G such that:
//   Res(f,X^n+1)*G - Res(g,X^n+1)*F = q
// The two integers are written into tmp[], over MOD_SMALL_BL[logn]
// words each.
fn solve_NTRU_deepest(logn: u32, f: &[i32], g: &[i32], tmp: &mut [u32]) -> bool {
    solve_NTRU_deepest_staged(logn, f, g, tmp).is_ok()
}

fn solve_NTRU_deepest_staged(
    logn: u32,
    f: &[i32],
    g: &[i32],
    tmp: &mut [u32],
) -> Result<(), &'static str> {
    let slen = MOD_SMALL_BL[logn as usize];

    // Get (f,g) at the deepest level. Obtained (f,g) are in RNS+NTT;
    // since degree is 1 at the deepest level, then NTT is a no-op and
    // we have (f,g) in RNS.
    if !make_fg_deepest(logn, f, g, tmp) {
        // f is not invertible modulo X^n+1 and modulo P0, we reject
        // that case.
        return Err("deepest_make_fg");
    }

    // Reorganize work area:
    //   Fp   output F (slen)
    //   Gp   output G (slen)
    //   fp   Res(f, X^n+1) (slen)
    //   gp   Res(g, X^n+1) (slen)
    //   t1   rest of temporary
    tmp.copy_within(0..(2 * slen), 2 * slen);
    let (Fp, tmp) = tmp.split_at_mut(slen);
    let (Gp, tmp) = tmp.split_at_mut(slen);
    let (fgp, t1) = tmp.split_at_mut(2 * slen);

    // Convert the resultants into plain integers. The resultants are always
    // non-negative, hence we do not normalize to signed.
    zint_rebuild_CRT(fgp, slen, 1, 2, false, t1);
    let (fp, gp) = fgp.split_at_mut(slen);

    // Apply the binary GCD to get (F, G).
    if zint_bezout(Gp, Fp, fp, gp, t1) != 0xFFFFFFFF {
        // Resultants are not coprime to each other; we reject that case
        // (note: we also reject the case where the GCD is exactly q,
        // even though that case could be handled.
        return Err("deepest_bezout");
    }

    // Multiply the obtained (F,G) by q to get a solution f*G - g*F = q.
    if zint_mul_small(Fp, Q) != 0 || zint_mul_small(Gp, Q) != 0 {
        // If either multiplication overflows, we reject.
        return Err("deepest_mul_q");
    }

    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InterFail {
    PreBabaiEq,
    PostBabaiEq,
}

/// Montgomery check: `f*G - g*F ≡ q (mod p0)` with plain RNS limbs.
fn ntru_eq_holds_mont(
    logn: u32,
    Ft: &[u32],
    Gt: &[u32],
    FGlen: usize,
    ft: &[u32],
    gt: &[u32],
    fglen: usize,
    fg_already_ntt: bool,
    scratch: &mut [u32],
) -> bool {
    let n = 1usize << logn;
    let (t_ft, scratch) = scratch.split_at_mut(n);
    let (t_gt, scratch) = scratch.split_at_mut(n);
    let (t1, scratch) = scratch.split_at_mut(n);
    let (t2, gm) = scratch.split_at_mut(n);

    let p = P0.p;
    let p0i = P0.p0i;
    let R2 = P0.R2;
    let Rx_f = mp_Rx31(fglen as u32, p, p0i, R2);
    let Rx_F = mp_Rx31(FGlen as u32, p, p0i, R2);
    mp_mkgm(logn, P0.g, p, p0i, gm);

    if fg_already_ntt {
        t_ft.copy_from_slice(&ft[..n]);
        t_gt.copy_from_slice(&gt[..n]);
    } else {
        for i in 0..n {
            t_ft[i] = zint_mod_small_signed(&ft[i..], fglen, n, p, p0i, R2, Rx_f);
            t_gt[i] = zint_mod_small_signed(&gt[i..], fglen, n, p, p0i, R2, Rx_f);
        }
        mp_NTT(logn, t_ft, gm, p, p0i);
        mp_NTT(logn, t_gt, gm, p, p0i);
    }
    for i in 0..n {
        t1[i] = zint_mod_small_signed(&Ft[i..], FGlen, n, p, p0i, R2, Rx_F);
        t2[i] = zint_mod_small_signed(&Gt[i..], FGlen, n, p, p0i, R2, Rx_F);
    }
    mp_NTT(logn, t1, gm, p, p0i);
    mp_NTT(logn, t2, gm, p, p0i);

    let rv = mp_mmul(Q, 1, p, p0i);
    for i in 0..n {
        let x = mp_mmul(t_ft[i], t2[i], p, p0i);
        let y = mp_mmul(t_gt[i], t1[i], p, p0i);
        if rv != mp_sub(x, y, p) {
            return false;
        }
    }
    true
}

// Solving the NTRU equation, intermediate level.
fn solve_NTRU_intermediate(
    logn_top: u32,
    f: &[i32],
    g: &[i32],
    depth: u32,
    tmp_u32: &mut [u32],
    tmp_fxr: &mut [FXR],
) -> Result<(), InterFail> {
    let logn = logn_top - depth;
    let n = 1usize << logn;
    let hn = n >> 1;

    // slen   size for (f,g) at this level (and also output (F,G))
    // llen   size for unreduced (F,G) at this level
    // tlen   size for (F,G) from the deeper level
    // Note: we always have llen >= tlen
    let slen = MOD_SMALL_BL[depth as usize];
    let llen = MOD_LARGE_BL[depth as usize];
    let tlen = MOD_SMALL_BL[(depth + 1) as usize];

    // Input layout:
    //   Fd   F from deeper level (tlen * hn)
    //   Gd   G from deeper level (tlen * hn)
    // Fd and Gd are in plain representation.

    // Get (f,g) for this level.
    let min_sav = MIN_SAVE_FG[logn_top as usize];
    if depth < min_sav {
        // (f,g) were not saved previously, recompute them.
        make_fg_intermediate(logn_top, f, g, depth, &mut tmp_u32[(2 * tlen * hn)..]);
    } else {
        // (f,g) were saved previously, get them.
        let mut sav_off = tmp_u32.len();
        for d in min_sav..(depth + 1) {
            sav_off -= MOD_SMALL_BL[d as usize] << (logn_top + 1 - d);
        }
        tmp_u32.copy_within(sav_off..(sav_off + 2 * slen * n), 2 * tlen * hn);
    }

    // Current layout:
    //   Fd   F from deeper level (tlen * hn)
    //   Gd   G from deeper level (tlen * hn)
    //   ft   f from this level (slen * n)
    //   gt   g from this level (slen * n)
    // We now move things to this layout:
    //   Ft   F from this level (unreduced) (llen * n)
    //   Gt   G from this level (unreduced) (llen * n)
    //   ft   f from this level (slen * n) (RNS+NTT)
    //   gt   g from this level (slen * n) (RNS+NTT)
    //   Fd   F from deeper level (tlen * hn) (plain)
    //   Gd   G from deeper level (tlen * hn) (plain)
    tmp_u32.copy_within(0..(2 * tlen * hn), 2 * (llen + slen) * n);
    tmp_u32.copy_within(
        (2 * tlen * hn)..(2 * tlen * hn + 2 * slen * n),
        2 * llen * n,
    );

    // Convert Fd and Gd to RNS, with output temporarily stored in (Ft, Gt).
    // Fd and Gd have degree hn only; we store the values for each modulus p
    // in the _last_ hn slots of the n-word line for that modulus.
    {
        let (Ft, work) = tmp_u32[..].split_at_mut(llen * n);
        let (Gt, work) = work.split_at_mut(llen * n);
        let (_, work) = work.split_at_mut(2 * slen * n); // ft and gt
        let (Fd, work) = work.split_at_mut(tlen * hn);
        let (Gd, _) = work.split_at_mut(tlen * hn);
        for i in 0..llen {
            let p = PRIMES[i].p;
            let p0i = PRIMES[i].p0i;
            let R2 = PRIMES[i].R2;
            let Rx = mp_Rx31(tlen as u32, p, p0i, R2);
            let kt = i * n + hn;
            for j in 0..hn {
                Ft[kt + j] = zint_mod_small_signed(&Fd[j..], tlen, hn, p, p0i, R2, Rx);
                Gt[kt + j] = zint_mod_small_signed(&Gd[j..], tlen, hn, p, p0i, R2, Rx);
            }
        }
    }

    // Fd and Gd are no longer needed.

    // Compute (F,G) (unreduced) modulo sufficiently many small primes.
    // We also un-NTT (f,g) as we go; when slen primes have been processed,
    // we have (f,g) in RNS, and we apply the CRT to get (f,g) in plain
    // representation.
    {
        let (FGt, work) = tmp_u32[..].split_at_mut(2 * llen * n);
        let (fgt, work) = work.split_at_mut(2 * slen * n); // ft and gt
        for i in 0..llen {
            let p = PRIMES[i].p;
            let p0i = PRIMES[i].p0i;
            let R2 = PRIMES[i].R2;

            // Memory layout:
            //   Ft    (n * llen)
            //   Gt    (n * llen)
            //   ft    (n * slen)
            //   gt    (n * slen)
            //   gm    NTT support (n)
            //   igm   iNTT support (n)
            //   fx    temporary f mod p (NTT) (n)
            //   gx    temporary g mod p (NTT) (n)
            {
                let (Ft, Gt) = FGt.split_at_mut(llen * n);
                let (ft, gt) = fgt.split_at_mut(slen * n);
                let (gm, work) = work.split_at_mut(n);
                let (igm, work) = work.split_at_mut(n);
                let (fx, work) = work.split_at_mut(n);
                let (gx, _) = work.split_at_mut(n);

                mp_mkgmigm(logn, PRIMES[i].g, PRIMES[i].ig, p, p0i, gm, igm);
                if i < slen {
                    fx.copy_from_slice(&ft[(i * n)..((i + 1) * n)]);
                    gx.copy_from_slice(&gt[(i * n)..((i + 1) * n)]);
                    mp_iNTT(logn, &mut ft[(i * n)..((i + 1) * n)], igm, p, p0i);
                    mp_iNTT(logn, &mut gt[(i * n)..((i + 1) * n)], igm, p, p0i);
                } else {
                    let Rx = mp_Rx31(slen as u32, p, p0i, R2);
                    for j in 0..n {
                        fx[j] = zint_mod_small_signed(&ft[j..], slen, n, p, p0i, R2, Rx);
                        gx[j] = zint_mod_small_signed(&gt[j..], slen, n, p, p0i, R2, Rx);
                    }
                    mp_NTT(logn, fx, gm, p, p0i);
                    mp_NTT(logn, gx, gm, p, p0i);
                }

                // We have (F,G) in RNS in Ft and Gt; we apply the NTT
                // modulo p. Note that we can use gm (generated for degree
                // n) for an NTT with degree hn = n/2.
                let kt = i * n + hn;
                mp_NTT(logn - 1, &mut Ft[kt..(kt + hn)], gm, p, p0i);
                mp_NTT(logn - 1, &mut Gt[kt..(kt + hn)], gm, p, p0i);

                // Compute F and G (unreduced) modulo p.
                let kt = i * n;
                for j in 0..hn {
                    let fa = fx[2 * j + 0];
                    let fb = fx[2 * j + 1];
                    let ga = gx[2 * j + 0];
                    let gb = gx[2 * j + 1];
                    let mFp = mp_mmul(Ft[kt + hn + j], R2, p, p0i);
                    let mGp = mp_mmul(Gt[kt + hn + j], R2, p, p0i);
                    Ft[kt + 2 * j + 0] = mp_mmul(gb, mFp, p, p0i);
                    Ft[kt + 2 * j + 1] = mp_mmul(ga, mFp, p, p0i);
                    Gt[kt + 2 * j + 0] = mp_mmul(fb, mGp, p, p0i);
                    Gt[kt + 2 * j + 1] = mp_mmul(fa, mGp, p, p0i);
                }
                mp_iNTT(logn, &mut Ft[kt..(kt + n)], igm, p, p0i);
                mp_iNTT(logn, &mut Gt[kt..(kt + n)], igm, p, p0i);
            }

            if (i + 1) == slen {
                // (f,g) are now in RNS, convert them to plain.
                zint_rebuild_CRT(fgt, slen, n, 2, true, work);
            }
        }

        // (Ft, Gt) are in RNS, we want them in plain representation.
        zint_rebuild_CRT(FGt, llen, n, 2, true, work);
    }

    // Current memory lauout:
    //   Ft   F from this level (unreduced) (llen * n) (plain)
    //   Gt   G from this level (unreduced) (llen * n) (plain)
    //   ft   f from this level (slen * n) (plain)
    //   gt   g from this level (slen * n) (plain)

    // Unreduced (F,G) must already satisfy the NTRU equation.
    {
        let (Ft, work) = tmp_u32.split_at_mut(llen * n);
        let (Gt, work) = work.split_at_mut(llen * n);
        let (ft, work) = work.split_at_mut(slen * n);
        let (gt, scratch) = work.split_at_mut(slen * n);
        if scratch.len() >= 5 * n
            && !ntru_eq_holds_mont(logn, Ft, Gt, llen, ft, gt, slen, false, scratch)
        {
            return Err(InterFail::PreBabaiEq);
        }
    }

    // We now reduce these (F,G) with Babai's nearest plane algorithm.
    // The reduction conceptually goes as follows:
    //   k <- round((F*adj(f) + G*adj(g))/(f*adj(f) + g*adj(g)))
    //   (F, G) <- (F - k*f, G - k*g)
    // We use fixed-point approximations of (f,g) and (F, G) to get
    // a value k as a small polynomial with scaling; we then apply
    // k on the full-width polynomial. Each iteration "shaves" a
    // a few bits off F and G.
    //
    // We apply the process sufficiently many times to reduce (F, G)
    // to the size of (f, g) with a reasonable probability of success.
    // Since we want full constant-time processing, the number of
    // iterations and the accessed slots work on some assumptions on
    // the sizes of values (sizes have been measured over many samples,
    // and a margin of 5 times the standard deviation).

    // If depth is at least 2, and we will use the NTT to subtract
    // (k*f,k*g) from (F,G), then we will need to convert (f,g) to
    // NTT over slen+1 words, which requires an extra word to ft and gt.
    // KoalaBear also keeps (f,g) at depth 1: poly_sub_kfg_scaled_depth1
    // only supports FGlen ∈ {1,2}, but MOD_LARGE_BL[1] can exceed that.
    let use_sub_ntt = logn >= MIN_LOGN_FGNTT;
    if use_sub_ntt {
        tmp_u32[..].copy_within(
            (2 * llen * n + slen * n)..(2 * llen * n + 2 * slen * n),
            2 * llen * n + (slen + 1) * n,
        );
    }
    let slen_adj = if use_sub_ntt { slen + 1 } else { slen };

    // Current memory layout:
    //   Ft    F from this level (unreduced) (llen * n) (plain)
    //   Gt    G from this level (unreduced) (llen * n) (plain)
    //   ft    f from this level (slen * n, +n if use_sub_ntt) (plain)
    //   gt    g from this level (slen * n, +n if use_sub_ntt) (plain)

    // For the reduction, we will consider only the top rlen words
    // of (f,g) — but that window must sit on the *actual* MSBs. Falcon’s
    // blen = slen - rlen assumes (f,g) nearly fill `slen` limbs; after
    // KoalaBear field-norm steps they often don’t, which made scale_x=0
    // and blew up Babai (k overflow).
    let rlen = WORD_WIN[depth as usize].min(slen);
    let (scale_fg, blen) = {
        let (_, work) = tmp_u32.split_at_mut(2 * n * llen);
        let (ft, gt) = work.split_at_mut(n * slen_adj);
        let bf = poly_max_bitlength(logn, ft, slen);
        let bg = poly_max_bitlength(logn, gt, slen);
        let bits = bf ^ ((bf ^ bg) & tbmask(bf.wrapping_sub(bg)));
        let words = (((bits + 30) / 31) as usize).clamp(rlen, slen);
        let blen = words - rlen;
        (31 * (blen as u32), blen)
    };
    let scale_x;

    {
        let (_, work) = tmp_u32.split_at_mut(2 * n * llen);
        let (ft, gt) = work.split_at_mut(n * slen_adj);

        // FXR values:
        //   rt3   n
        //   rt4   n
        //   rt1   n/2
        // TODO: share (rt3,rt4,rt1) with space just after gt
        let (rt3, rttmp) = tmp_fxr.split_at_mut(n);
        let (rt4, rt1) = rttmp.split_at_mut(n);

        // scale_x is the maximum bit length of f and g (beyond scale_fg)
        let scale_xf = poly_max_bitlength(logn, &ft[(n * blen)..], rlen);
        let scale_xg = poly_max_bitlength(logn, &gt[(n * blen)..], rlen);
        scale_x = scale_xf ^ ((scale_xf ^ scale_xg) & tbmask(scale_xf.wrapping_sub(scale_xg)));

        // scale_t is from logn, but not greater than scale_x
        let scale_t = 15 - logn;
        let scale_t = scale_t ^ ((scale_t ^ scale_x) & tbmask(scale_x.wrapping_sub(scale_t)));
        let scdiff = scale_x - scale_t;

        // Extract the approximations of f and g (scaled).
        poly_big_to_fixed(logn, &ft[(n * blen)..], rlen, scdiff, rt3);
        poly_big_to_fixed(logn, &gt[(n * blen)..], rlen, scdiff, rt4);

        // Compute adj(f)/(f*adj(f) + g*adj(g)) into rt3 (FFT).
        // Compute adj(g)/(f*adj(f) + g*adj(g)) into rt4 (FFT).
        vect_FFT(logn, rt3);
        vect_FFT(logn, rt4);
        vect_norm_fft(logn, rt1, rt3, rt4);
        vect_mul2e(logn, rt3, scale_t);
        vect_mul2e(logn, rt4, scale_t);
        for i in 0..hn {
            // Note: four independent divisions; we do not mutualize the
            // inversion of rt1[i] since that would lose too much precision.
            rt3[i] /= rt1[i];
            rt3[i + hn] = (-rt3[i + hn]) / rt1[i];
            rt4[i] /= rt1[i];
            rt4[i + hn] = (-rt4[i + hn]) / rt1[i];
        }
    }

    // New layout:
    //   Ft    F from this level (unreduced) (llen * n)
    //   Gt    G from this level (unreduced) (llen * n)
    //   ft    f from this level (slen_adj * n)
    //   gt    g from this level (slen_adj * n)
    //   k     n
    //   t2    3*n
    //
    //   rt3   n (FXR)
    //   rt4   n (FXR)
    //   rt1   n (FXR)
    //   rt2   n (FXR)
    //
    // TODO: merge the FXR space with the u32 space:
    //   rt3 starts right after gt
    //   k,t2 can share the same space as rt1,rt2
    //   at depth 1 we should also remove ft and gt
    {
        let (Ft, work) = tmp_u32.split_at_mut(llen * n);
        let (Gt, work) = work.split_at_mut(llen * n);
        let fgt_size = 2 * slen_adj * n;
        let (fgt, work) = work.split_at_mut(fgt_size);
        let (k, t2) = work.split_at_mut(n);

        let (rt3, work) = tmp_fxr.split_at_mut(n);
        let (rt4, work) = work.split_at_mut(n);
        let (rt1, work) = work.split_at_mut(n);
        let (rt2, _) = work.split_at_mut(n);

        // Ft, Gt, ft, gt, rt3 and rt4 are already set.
        // If we use poly_sub_scaled_ntt(), then we convert f and g to
        // NTT.
        if use_sub_ntt {
            let (ft, gt) = fgt.split_at_mut(slen_adj * n);
            let (gm, tn) = t2.split_at_mut(n);
            for i in 0..slen_adj {
                let p = PRIMES[i].p;
                let p0i = PRIMES[i].p0i;
                let R2 = PRIMES[i].R2;
                let Rx = mp_Rx31(slen as u32, p, p0i, R2);
                mp_mkgm(logn, PRIMES[i].g, p, p0i, gm);
                for j in 0..n {
                    tn[(i << logn) + j] = zint_mod_small_signed(&ft[j..], slen, n, p, p0i, R2, Rx);
                }
                mp_NTT(logn, &mut tn[(i << logn)..], gm, p, p0i);
            }
            ft.copy_from_slice(&tn[..(slen_adj * n)]);
            for i in 0..slen_adj {
                let p = PRIMES[i].p;
                let p0i = PRIMES[i].p0i;
                let R2 = PRIMES[i].R2;
                let Rx = mp_Rx31(slen as u32, p, p0i, R2);
                mp_mkgm(logn, PRIMES[i].g, p, p0i, gm);
                for j in 0..n {
                    tn[(i << logn) + j] = zint_mod_small_signed(&gt[j..], slen, n, p, p0i, R2, Rx);
                }
                mp_NTT(logn, &mut tn[(i << logn)..], gm, p, p0i);
            }
            gt.copy_from_slice(&tn[..(slen_adj * n)]);
        }

        // Reduce F and G repeatedly.
        // Each iteration is expected to reduce the size of the coefficients
        // by reduce_bits.
        //
        // FGlen must never drop below the *actual* coefficient length: Falcon
        // shrinks FGlen on claimed scale progress and then only mutates that
        // many limbs in poly_sub_*. With KoalaBear, k≈0 for many early
        // rounds (leading zero padding), so a premature FGlen shrink freezes
        // the still-large high limbs forever while “finishing” Babai.
        let reduce_bits = REDUCE_BITS[logn_top as usize];
        let mut FGlen = llen;
        let mut scale_FG = {
            let bf = poly_max_bitlength(logn, Ft, llen);
            let bg = poly_max_bitlength(logn, Gt, llen);
            let bits = bf ^ ((bf ^ bg) & tbmask(bf.wrapping_sub(bg)));
            bits.saturating_add(31).max(scale_fg + 31)
        };
        let mut max_abs_k: i32 = 0;
        let mut iters: u32 = 0;
        loop {
            iters += 1;
            let _ = iters;
            let (sch, coff) = divrem31(scale_FG);
            let clen = sch as usize;
            if clen >= FGlen {
                if scale_FG <= scale_fg {
                    break;
                }
                scale_FG = scale_FG.saturating_sub(reduce_bits.max(1));
                continue;
            }
            poly_big_to_fixed(logn, &Ft[(clen * n)..], FGlen - clen, scale_x + coff, rt1);
            poly_big_to_fixed(logn, &Gt[(clen * n)..], FGlen - clen, scale_x + coff, rt2);

            // rt2 <- (F*adj(f) + G*adj(g)) / (f*adj(f) + g*adj(g))
            vect_FFT(logn, rt1);
            vect_FFT(logn, rt2);
            vect_mul_fft(logn, rt1, rt3);
            vect_mul_fft(logn, rt2, rt4);
            vect_add(logn, rt2, rt1);
            vect_iFFT(logn, rt2);

            // k <- round(rt2)  (i32 elements, stored in u32 slice)
            let mut iter_max_k: i32 = 0;
            for i in 0..n {
                let ki = rt2[i].round();
                iter_max_k = iter_max_k.max(ki.saturating_abs());
                k[i] = ki as u32;
            }
            max_abs_k = max_abs_k.max(iter_max_k);
            // Bail if k left the intended small-integer range (FXR overflow).
            if iter_max_k > (1 << 20) {
                break;
            }

            // (f,g) are scaled by scale_fg + scale_x
            // (F,G) are scaled by scale_FG + scale_x
            // Thus, k is scaled by scale_FG - scale_fg, which is public.
            let scale_k = scale_FG - scale_fg;

            if use_sub_ntt {
                let (ft, gt) = fgt.split_at_mut(slen_adj * n);
                poly_sub_scaled_ntt(logn, Ft, FGlen, ft, slen, k, scale_k, t2);
                poly_sub_scaled_ntt(logn, Gt, FGlen, gt, slen, k, scale_k, t2);
            } else if depth == 1 {
                poly_sub_kfg_scaled_depth1(logn_top, Ft, Gt, FGlen, k, scale_k, f, g, t2);
            } else {
                let (ft, gt) = fgt.split_at_mut(slen_adj * n);
                poly_sub_scaled(logn, Ft, FGlen, ft, slen, k, scale_k);
                poly_sub_scaled(logn, Gt, FGlen, gt, slen, k, scale_k);
            }

            // We now assume that F and G have shrunk by at least
            // reduce_bits — but only drop FGlen when the measured size agrees.
            if scale_FG <= scale_fg {
                break;
            }
            if scale_FG <= (scale_fg + reduce_bits) {
                scale_FG = scale_fg;
            } else {
                scale_FG -= reduce_bits;
            }
            let bf = poly_max_bitlength(logn, Ft, llen);
            let bg = poly_max_bitlength(logn, Gt, llen);
            let bits = bf ^ ((bf ^ bg) & tbmask(bf.wrapping_sub(bg)));
            let live = (((bits + 30) / 31) as usize).clamp(slen, llen);
            while FGlen > live && 31 * ((FGlen - slen) as u32) > scale_FG - scale_fg + 30 {
                FGlen -= 1;
            }
            // Keep mutating at least `live` limbs so high words can clear.
            FGlen = FGlen.max(live);
        }
        let _ = max_abs_k;

        // After Babai: check the reduced (`slen`) projection of (F,G).
        {
            let (ft, gt) = fgt.split_at_mut(slen_adj * n);
            let fg_ntt = use_sub_ntt;
            let fg_len = if use_sub_ntt { 1 } else { slen };
            let mut scratch = vec![0u32; 5 * n];
            let eq_small =
                ntru_eq_holds_mont(logn, Ft, Gt, slen, ft, gt, fg_len, fg_ntt, &mut scratch);
            if !eq_small {
                return Err(InterFail::PostBabaiEq);
            }
            if llen > slen {
                for i in slen..llen {
                    for j in 0..n {
                        Ft[i * n + j] = 0;
                        Gt[i * n + j] = 0;
                    }
                }
            }
        }
    }

    // Output F is already in the right place; G must be moved.
    tmp_u32.copy_within((llen * n)..((llen + slen) * n), slen * n);

    // Reduction is done. We test the current solution modulo a single
    // prime.
    // Exception: this is not done if depth == 1 (the reference C code
    // did not keep (ft,gt) in that case). In any case, the depth-0
    // test will cover it.
    // If use_sub_ntt is true, then ft and gt are already in NTT
    // representation.
    if depth == 1 {
        return Ok(());
    }

    // Move (ft,gt) right after the reduced G.
    // If use_sub_ntt is false, then slen_adj == slen.
    // If use_sub_ntt is true, then slen_adj == slen + 1, but (ft,gt) are
    // already in NTT representation and we only need the first coefficient.
    if use_sub_ntt {
        // ft mod p0 (NTT)
        tmp_u32.copy_within(((2 * llen) * n)..((2 * llen + 1) * n), 2 * slen * n);
        // gt mod p0 (NTT)
        tmp_u32.copy_within(
            ((2 * llen + slen_adj) * n)..((2 * llen + slen_adj + 1) * n),
            (2 * slen + slen) * n,
        );
    } else {
        tmp_u32.copy_within((2 * llen * n)..(2 * (llen + slen) * n), 2 * slen * n);
    }

    {
        let (Ft, work) = tmp_u32.split_at_mut(slen * n);
        let (Gt, work) = work.split_at_mut(slen * n);
        let (ft, work) = work.split_at_mut(slen * n);
        let (gt, scratch) = work.split_at_mut(slen * n);
        // Prefer NTT-based check when (ft,gt) are already NTT(mod p0).
        if !ntru_eq_holds_mont(
            logn,
            Ft,
            Gt,
            slen,
            ft,
            gt,
            if use_sub_ntt { 1 } else { slen },
            use_sub_ntt,
            scratch,
        ) {
            return Err(InterFail::PostBabaiEq);
        }
    }

    Ok(())
}

// Solving the NTRU equation, top-level.
fn solve_NTRU_depth0(
    logn: u32,
    f: &[i32],
    g: &[i32],
    tmp_u32: &mut [u32],
    tmp_fxr: &mut [FXR],
) -> bool {
    let n = 1usize << logn;
    let hn = n >> 1;

    // Normally, (F,G) from depth 1 should use one word per coefficient.
    // The code in this function assumes it.
    assert!(MOD_SMALL_BL[1] == 1);

    // At depth 0, all values fit on 30 bits, so we work with a single
    // modulus p.
    let p = P0.p;
    let p0i = P0.p0i;
    let R2 = P0.R2;

    {
        // Layout:
        //   Fd   F from upper level (hn)
        //   Gd   G from upper level (hn)
        //   ft   f (n)
        //   gt   g (n)
        //   gm   helper for NTT
        let (Fd, work) = tmp_u32.split_at_mut(hn);
        let (Gd, work) = work.split_at_mut(hn);
        let (ft, work) = work.split_at_mut(n);
        let (gt, work) = work.split_at_mut(n);
        let (gm, _) = work.split_at_mut(n);

        // Load f and g, convert to RNS+NTT
        mp_mkgm(logn, P0.g, p, p0i, gm);
        poly_mp_set_small(logn, f, p, ft);
        poly_mp_set_small(logn, g, p, gt);
        mp_NTT(logn, ft, gm, p, p0i);
        mp_NTT(logn, gt, gm, p, p0i);

        // Convert Fd and Gd to RNS+NTT
        poly_mp_set(logn - 1, Fd, p);
        poly_mp_set(logn - 1, Gd, p);
        mp_NTT(logn - 1, Fd, gm, p, p0i);
        mp_NTT(logn - 1, Gd, gm, p, p0i);

        // Build the unreduced (F,G) into ft and gt
        for i in 0..hn {
            let fa = ft[(i << 1) + 0];
            let fb = ft[(i << 1) + 1];
            let ga = gt[(i << 1) + 0];
            let gb = gt[(i << 1) + 1];
            let mFd = mp_mmul(Fd[i], R2, p, p0i);
            let mGd = mp_mmul(Gd[i], R2, p, p0i);
            ft[(i << 1) + 0] = mp_mmul(gb, mFd, p, p0i);
            ft[(i << 1) + 1] = mp_mmul(ga, mFd, p, p0i);
            gt[(i << 1) + 0] = mp_mmul(fb, mGd, p, p0i);
            gt[(i << 1) + 1] = mp_mmul(fa, mGd, p, p0i);
        }
    }

    // Reorganize buffers:
    //   Fp   unreduced F (n) (RNS+NTT)
    //   Gp   unreduced G (n) (RNS+NTT)
    //   t1   free (n)
    //   t2   NTT support (gm) (n)
    //   t3   free (n)
    //   t4   free (n)
    tmp_u32.copy_within(n..(3 * n), 0);

    {
        let (Fp, work) = tmp_u32.split_at_mut(n);
        let (Gp, work) = work.split_at_mut(n);
        let (t1, work) = work.split_at_mut(n);
        let (t2, work) = work.split_at_mut(n);
        let (t3, t4) = work.split_at_mut(n);

        // t4 <- f (RNS+NTT)
        poly_mp_set_small(logn, f, p, t4);
        mp_NTT(logn, t4, t2, p, p0i);

        // t1 <- F*adj(f) (RNS+NTT)
        // t3 <- f*adj(f) (RNS+NTT)
        for i in 0..n {
            let w = mp_mmul(t4[(n - 1) - i], R2, p, p0i);
            t1[i] = mp_mmul(w, Fp[i], p, p0i);
            t3[i] = mp_mmul(w, t4[i], p, p0i);
        }

        // t4 <- g (RNS+NTT)
        poly_mp_set_small(logn, g, p, t4);
        mp_NTT(logn, t4, t2, p, p0i);

        // t1 <- t1 + G*adj(g) (RNS+NTT)
        // t3 <- t3 + g*adj(g) (RNS+NTT)
        for i in 0..n {
            let w = mp_mmul(t4[(n - 1) - i], R2, p, p0i);
            t1[i] = mp_add(t1[i], mp_mmul(w, Gp[i], p, p0i), p);
            t3[i] = mp_add(t3[i], mp_mmul(w, t4[i], p, p0i), p);
        }

        // Convert back F*adj(f) + G*adj(g) and f*adj(f) + g*adj(g) to
        // plain representation, and also move f*adj(f) + g*adj(g) to t2.
        mp_mkigm(logn, P0.ig, p, p0i, t4);
        mp_iNTT(logn, t1, t4, p, p0i);
        mp_iNTT(logn, t3, t4, p, p0i);
        for i in 0..n {
            // Note: we do not truncate to 31 bits.
            t1[i] = mp_norm(t1[i], p) as u32;
            t2[i] = mp_norm(t3[i], p) as u32;
        }
    }

    // Current layout:
    //   Fp   unreduced F (RNS+NTT) (n)
    //   Gp   unreduced G (RNS+NTT) (n)
    //   t1   F*adj(f) + G*adj(g) (plain, 32-bit) (n)
    //   t2   f*adj(f) + g*adj(g) (plain, 32-bit) (n)

    // We need to divide t1 by t2, and round the result. We convert
    // them to FFT representation, downscaled by 2^10 (to avoid overflows).
    // We first convert f*adj(f) + g*adj(g), which is self-adjoint;
    // this, its FFT representation only has half-size.
    {
        let (_, work) = tmp_u32.split_at_mut(n);
        let (_, work) = work.split_at_mut(n);
        let (t1, t2) = work.split_at_mut(n);
        let (rt2, rt3) = tmp_fxr.split_at_mut(hn);

        // rt2 <- f*adj(f) + g*adj(g) (FFT, self-adjoint, scaled)
        for i in 0..n {
            let x = ((t2[i] as i32) as i64) << 22;
            rt3[i] = FXR::from_u64_scaled32(x as u64);
        }
        vect_FFT(logn, rt3);
        rt2.copy_from_slice(&rt3[..hn]);

        // rt3 <- F*adj(f) + G*adj(g) (FFT, scaled)
        for i in 0..n {
            let x = ((t1[i] as i32) as i64) << 22;
            rt3[i] = FXR::from_u64_scaled32(x as u64);
        }
        vect_FFT(logn, rt3);

        // Divide F*adj(f) + G*adj(g) by f*adj(f) + g*adj(g), and round
        // the result into t1, with conversion to RNS.
        vect_div_selfadj_fft(logn, rt3, rt2);
        vect_iFFT(logn, rt3);
        for i in 0..n {
            t1[i] = mp_set(rt3[i].round(), p);
        }
    }

    // Current layout:
    //   Fp   unreduced F (RNS+NTT) (n)
    //   Gp   unreduced G (RNS+NTT) (n)
    //   t1   k (RNS) (n)
    //   t2   free (n)
    //   t3   free (n)
    //   t4   free (n)

    {
        let (Fp, work) = tmp_u32.split_at_mut(n);
        let (Gp, work) = work.split_at_mut(n);
        let (t1, work) = work.split_at_mut(n);
        let (t2, work) = work.split_at_mut(n);
        let (t3, t4) = work.split_at_mut(n);

        // Convert k to RNS+NTT.
        mp_mkgm(logn, P0.g, p, p0i, t4);
        mp_NTT(logn, t1, t4, p, p0i);

        // Subtract k*f from F and k*G from G.
        // We also compute f*G - g*F (in RNS+NTT) to check that the solution
        // is correct.
        poly_mp_set_small(logn, f, p, t2);
        poly_mp_set_small(logn, g, p, t3);
        mp_NTT(logn, t2, t4, p, p0i);
        mp_NTT(logn, t3, t4, p, p0i);
        let rv = mp_mmul(Q, 1, p, p0i);
        for i in 0..n {
            let kv = mp_mmul(t1[i], R2, p, p0i);
            Fp[i] = mp_sub(Fp[i], mp_mmul(kv, t2[i], p, p0i), p);
            Gp[i] = mp_sub(Gp[i], mp_mmul(kv, t3[i], p, p0i), p);
            let x = mp_sub(
                mp_mmul(t2[i], Gp[i], p, p0i),
                mp_mmul(t3[i], Fp[i], p, p0i),
                p,
            );
            if x != rv {
                return false;
            }
        }

        // Convert back F and G into normal representation.
        mp_mkigm(logn, P0.ig, p, p0i, t4);
        mp_iNTT(logn, Fp, t4, p, p0i);
        mp_iNTT(logn, Gp, t4, p, p0i);
        poly_mp_norm(logn, Fp, p);
        poly_mp_norm(logn, Gp, p);
    }

    return true;
}

// Inject (f,g) at the top-level: f and g are converted to NTT and
// written into the first 2*n words of tmp[].
fn make_fg_depth0(logn: u32, f: &[i32], g: &[i32], tmp: &mut [u32]) {
    let n = 1usize << logn;
    let p = P0.p;
    let p0i = P0.p0i;
    let (ft, tmp) = tmp.split_at_mut(n);
    let (gt, tmp) = tmp.split_at_mut(n);
    let (gm, _) = tmp.split_at_mut(n);
    poly_mp_set_small(logn, f, p, ft);
    poly_mp_set_small(logn, g, p, gt);
    mp_mkgm(logn, P0.g, p, p0i, gm);
    mp_NTT(logn, ft, gm, p, p0i);
    mp_NTT(logn, gt, gm, p, p0i);
}

// One step of computing (f,g) at a given depth.
// Input: (f,g) of degree 2^(logn_top - depth)
// Output: (f',g') of degree 2^(logn_top - (depth+1))
fn make_fg_step(logn_top: u32, depth: u32, work: &mut [u32]) {
    let logn = logn_top - depth;
    let n = 1usize << logn;
    let hn = n >> 1;
    let slen = MOD_SMALL_BL[depth as usize];
    let tlen = MOD_SMALL_BL[(depth + 1) as usize];

    // Prepare buffers:
    //   fd, gd: output polynomials
    //   fs, gs: source polynomials
    //   gm, igm: buffers for NTT support arrays
    //   data: remaining slots (used for CRT)
    let data = work;
    data.copy_within(0..(2 * n * slen), 2 * hn * tlen);
    let (fd, data) = data.split_at_mut(hn * tlen);
    let (gd, data) = data.split_at_mut(hn * tlen);
    let (fgs, data) = data.split_at_mut(2 * n * slen);

    // First slen words: we use the input values directly, and apply
    // inverse NTT as we go, so that we get the sources in RNS (non-NTT).
    {
        let (fs, gs) = fgs.split_at_mut(n * slen);
        let (igm, _) = data.split_at_mut(n);
        for i in 0..slen {
            let p = PRIMES[i].p;
            let p0i = PRIMES[i].p0i;
            let R2 = PRIMES[i].R2;
            let ks = i * n;
            let kd = i * hn;
            for j in 0..hn {
                fd[kd + j] = mp_mmul(
                    mp_mmul(fs[ks + 2 * j], fs[ks + 2 * j + 1], p, p0i),
                    R2,
                    p,
                    p0i,
                );
                gd[kd + j] = mp_mmul(
                    mp_mmul(gs[ks + 2 * j], gs[ks + 2 * j + 1], p, p0i),
                    R2,
                    p,
                    p0i,
                );
            }
            mp_mkigm(logn, PRIMES[i].ig, p, p0i, igm);
            mp_iNTT(logn, &mut fs[ks..], igm, p, p0i);
            mp_iNTT(logn, &mut gs[ks..], igm, p, p0i);
        }
    }

    // Remaining output words.
    if tlen > slen {
        // fs and gs are in RNS, rebuild them into plain integer coefficients.
        zint_rebuild_CRT(fgs, slen, n, 2, true, data);

        let (fs, gs) = fgs.split_at_mut(n * slen);
        let (gm, data) = data.split_at_mut(n);
        let (t2, _) = data.split_at_mut(n);
        for i in slen..tlen {
            let p = PRIMES[i].p;
            let p0i = PRIMES[i].p0i;
            let R2 = PRIMES[i].R2;
            let Rx = mp_Rx31(slen as u32, p, p0i, R2);
            mp_mkgm(logn, PRIMES[i].g, p, p0i, gm);
            let kd = i * hn;

            for j in 0..n {
                t2[j] = zint_mod_small_signed(&fs[j..], slen, n, p, p0i, R2, Rx);
            }
            mp_NTT(logn, t2, gm, p, p0i);
            for j in 0..hn {
                fd[kd + j] = mp_mmul(mp_mmul(t2[2 * j], t2[2 * j + 1], p, p0i), R2, p, p0i);
            }

            for j in 0..n {
                t2[j] = zint_mod_small_signed(&gs[j..], slen, n, p, p0i, R2, Rx);
            }
            mp_NTT(logn, t2, gm, p, p0i);
            for j in 0..hn {
                gd[kd + j] = mp_mmul(mp_mmul(t2[2 * j], t2[2 * j + 1], p, p0i), R2, p, p0i);
            }
        }
    }
}

// Recompute (f,g) at a given depth.
fn make_fg_intermediate(logn_top: u32, f: &[i32], g: &[i32], depth: u32, work: &mut [u32]) {
    make_fg_depth0(logn_top, f, g, work);
    for d in 0..depth {
        make_fg_step(logn_top, d, work);
    }
}

// Recompute (f, g) at the deepest level. Intermediate (f,g) values
// (below the save threshold) are copied at the end of the work area.
//
// If f is not invertible modulo X^n+1 and modulo p = 2147473409,
// then this function returns false (but everything else is still
// computed); otherwise, this function returns true. There is no such
// test on g.
fn make_fg_deepest(logn: u32, f: &[i32], g: &[i32], mut work: &mut [u32]) -> bool {
    make_fg_depth0(logn, f, g, work);

    // f is now in RNS+NTT; we can test its invertibility by checking
    // that all its NTT coefficients are non-zero.
    let n = 1usize << logn;
    let mut b = 0;
    for i in 0..n {
        b |= work[i].wrapping_sub(1);
    }
    let r = (b >> 31) == 0;

    // Compute all the reduced (f,g) values, saving the intermediate
    // values (except that the highest levels).
    for d in 0..logn {
        make_fg_step(logn, d, work);
        let d2 = d + 1;
        if d2 < logn && d2 >= MIN_SAVE_FG[logn as usize] {
            let slen = MOD_SMALL_BL[d2 as usize];
            let fglen = slen << (logn + 1 - d2);
            let sav_off = work.len() - fglen;
            work.copy_within(0..fglen, sav_off);
            work = &mut work[..sav_off];
        }
    }

    r
}
