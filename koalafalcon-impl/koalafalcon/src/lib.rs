#![allow(unexpected_cfgs)]

mod profiling {
    /// Add a call-stack span to Perfetto traces when the `profile` feature is enabled.
    macro_rules! profile {
        ($name:expr) => {
            #[cfg(feature = "profile")]
            let _span = tracing::debug_span!($name).entered();
        };
    }
    pub(crate) use profile;
}
pub(crate) use profiling::profile;

mod algebra;
mod falcon;
mod utils;

pub use algebra::{CyclotomicRing, Field, KOALA_BEAR_PRIME, KoalaBear, KoalaBearRing};
pub use falcon::*;
pub use utils::{Error, bit_reverse};
