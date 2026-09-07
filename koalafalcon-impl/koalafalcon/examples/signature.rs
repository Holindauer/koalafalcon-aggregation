use std::thread;
use std::time::{Duration, Instant};

use clap::Parser;
use koalafalcon::{Signature, keygen};
use rand::{SeedableRng, rngs::StdRng};

const DEFAULT_SIGNATURES: usize = 32_768;

#[derive(Debug, Parser)]
#[command(about = "Sign and verify one message many times with KoalaFalcon")]
struct Args {
    #[arg(long, default_value_t = 512, value_parser = parse_n)]
    n: usize,
    #[arg(long, default_value_t = DEFAULT_SIGNATURES, value_parser = parse_positive_usize)]
    signatures: usize,
    #[arg(long, value_parser = parse_positive_usize)]
    threads: Option<usize>,
    #[arg(long, default_value = "hello, koala falcon")]
    message: String,
}

fn parse_positive_usize(value: &str) -> Result<usize, String> {
    let value = value
        .parse::<usize>()
        .map_err(|_| format!("expected a positive integer, got {value:?}"))?;
    (value > 0)
        .then_some(value)
        .ok_or_else(|| "value must be greater than zero".to_owned())
}

fn parse_n(value: &str) -> Result<usize, String> {
    match value {
        "512" => Ok(512),
        "1024" => Ok(1024),
        _ => Err(format!("expected 512 or 1024, got {value:?}")),
    }
}

fn display_duration(duration: Duration) -> String {
    format!("{:.3} s", duration.as_secs_f64())
}

fn run_benchmark<const N: usize>(args: &Args, workers: usize) {
    let message = args.message.as_bytes();

    let mut keygen_rng = StdRng::from_os_rng();
    let keygen_started = Instant::now();
    let (signing_key, verifying_key) = keygen::<N>(&mut keygen_rng).expect("keygen");
    let keygen_time = keygen_started.elapsed();

    let base = args.signatures / workers;
    let remainder = args.signatures % workers;
    let sign_started = Instant::now();
    let signatures: Vec<Signature<N>> = thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        for worker in 0..workers {
            let worker_count = base + usize::from(worker < remainder);
            let signing_key = &signing_key;
            handles.push(scope.spawn(move || {
                let mut rng = StdRng::from_os_rng();
                let mut signatures = Vec::with_capacity(worker_count);
                for _ in 0..worker_count {
                    signatures.push(signing_key.sign(message, &mut rng).expect("sign"));
                }
                signatures
            }));
        }

        handles
            .into_iter()
            .flat_map(|handle| handle.join().expect("signing worker"))
            .collect()
    });
    let sign_time = sign_started.elapsed();

    let verify_started = Instant::now();
    let all_valid = thread::scope(|scope| {
        let chunk_len = signatures.len().div_ceil(workers);
        let handles: Vec<_> = signatures
            .chunks(chunk_len)
            .map(|chunk| {
                let verifying_key = &verifying_key;
                scope.spawn(move || {
                    chunk
                        .iter()
                        .all(|signature| verifying_key.verify(message, signature).unwrap_or(false))
                })
            })
            .collect();
        handles
            .into_iter()
            .all(|handle| handle.join().expect("verification worker"))
    });
    let verify_time = verify_started.elapsed();
    assert!(all_valid);

    println!("parameter set: Falcon-{N}");
    println!(
        "message:       {:?} ({} bytes)",
        args.message,
        message.len()
    );
    println!("signatures:    {}", signatures.len());
    println!("threads:       {workers}");
    println!("keygen:        {}", display_duration(keygen_time));
    println!("sign total:    {}", display_duration(sign_time));
    println!(
        "sign rate:     {:.0} signatures/s",
        signatures.len() as f64 / sign_time.as_secs_f64()
    );
    println!("verify total:  {}", display_duration(verify_time));
    println!(
        "verify rate:   {:.0} verifies/s",
        signatures.len() as f64 / verify_time.as_secs_f64()
    );
}

fn main() {
    let args = Args::parse();

    #[cfg(feature = "profile")]
    let _guard = tracing_profile::init_tracing().expect("initialize Perfetto tracing");

    let available_threads = thread::available_parallelism().map_or(1, usize::from);
    let workers = args
        .threads
        .unwrap_or(available_threads)
        .min(args.signatures);

    match args.n {
        512 => run_benchmark::<512>(&args, workers),
        1024 => run_benchmark::<1024>(&args, workers),
        _ => unreachable!("clap rejects unsupported --n values"),
    }
}
