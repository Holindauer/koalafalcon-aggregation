use std::fmt::{self, Display, Formatter};

#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    NotInvertible,
    UnsupportedRingDimension(usize),
    InvalidSignatureCount(usize),
    VerificationFailed,
    InsufficientInputBytes { expected: usize, got: usize },
    Trapdoor(TrapdoorError),
    SigningFailed,
    FourierSampling(FourierSamplingError),
    InvalidParameterSet(String),
    ParameterOverflow,
}

/// Errors from FFT / ffLDL / ffSampling scaffolding.
#[derive(Debug, Clone, PartialEq)]
pub enum FourierSamplingError {
    InvalidDegree(usize),
    BadTreeShape {
        expected_leaves: usize,
        got_leaves: usize,
    },
    NonFinite,
    NearZeroDenominator,
    InvalidLeafSigma(f64),
    LeafBelowSigmin {
        sigma: f64,
        sigmin: f64,
    },
    LeafAboveSamplerMax {
        sigma: f64,
        max_sigma: f64,
    },
    InvalidSamplerParameters {
        sigma: f64,
        sigmin: f64,
    },
    LeafConversionFailed,
    OutputOverflow,
    ExactPreimageFailed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrapdoorError {
    UnsupportedDegree(usize),
    NotInvertible,
    SolveFailed,
    ExceededAttempts(usize),
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Error::NotInvertible => write!(f, "element is not invertible"),
            Error::UnsupportedRingDimension(n) => {
                write!(f, "unsupported ring dimension {n}; expected 512 or 1024")
            }
            Error::InvalidSignatureCount(n) => {
                write!(f, "signature count must be a nonzero power of two, got {n}")
            }
            Error::VerificationFailed => write!(f, "signature verification failed"),
            Error::InsufficientInputBytes { expected, got } => {
                write!(f, "expected {expected} input bytes, got {got}")
            }
            Error::Trapdoor(e) => write!(f, "trapdoor generation failed: {e}"),
            Error::SigningFailed => {
                write!(f, "signing failed: exceeded rejection sampling attempts")
            }
            Error::FourierSampling(e) => write!(f, "Fourier sampling error: {e}"),
            Error::InvalidParameterSet(msg) => write!(f, "{msg}"),
            Error::ParameterOverflow => write!(f, "benchmark parameter overflow"),
        }
    }
}

impl std::error::Error for Error {}

impl Display for FourierSamplingError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDegree(n) => write!(f, "invalid FFT degree {n}"),
            Self::BadTreeShape {
                expected_leaves,
                got_leaves,
            } => write!(
                f,
                "bad Falcon tree shape: expected {expected_leaves} leaves, got {got_leaves}"
            ),
            Self::NonFinite => write!(f, "non-finite FFT/LDL value"),
            Self::NearZeroDenominator => write!(f, "near-zero LDL denominator"),
            Self::InvalidLeafSigma(s) => write!(f, "invalid leaf sigma {s}"),
            Self::LeafBelowSigmin { sigma, sigmin } => {
                write!(f, "leaf sigma {sigma} below sigmin {sigmin}")
            }
            Self::LeafAboveSamplerMax { sigma, max_sigma } => {
                write!(f, "leaf sigma {sigma} above SamplerZ max {max_sigma}")
            }
            Self::InvalidSamplerParameters { sigma, sigmin } => {
                write!(
                    f,
                    "invalid SamplerZ parameters: sigma={sigma}, sigmin={sigmin} (need 1 < sigmin ≤ sigma ≤ 1.8205)"
                )
            }
            Self::LeafConversionFailed => write!(f, "leaf sigma conversion to f64 failed"),
            Self::OutputOverflow => write!(f, "sample coefficient overflow"),
            Self::ExactPreimageFailed => {
                write!(f, "exact ring preimage check failed (s1+s2*h != point)")
            }
        }
    }
}

impl std::error::Error for FourierSamplingError {}

impl From<FourierSamplingError> for Error {
    fn from(value: FourierSamplingError) -> Self {
        Self::FourierSampling(value)
    }
}

impl Display for TrapdoorError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedDegree(n) => {
                write!(f, "unsupported trapdoor degree {n} (need power of two)")
            }
            Self::NotInvertible => write!(f, "f is not invertible modulo q"),
            Self::SolveFailed => write!(f, "NTRU equation has no solution for sampled f, g"),
            Self::ExceededAttempts(n) => write!(f, "exceeded {n} sampling attempts"),
        }
    }
}

impl std::error::Error for TrapdoorError {}

impl From<TrapdoorError> for Error {
    fn from(value: TrapdoorError) -> Self {
        Self::Trapdoor(value)
    }
}
