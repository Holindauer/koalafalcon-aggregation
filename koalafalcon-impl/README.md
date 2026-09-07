# KoalaFalcon

This workspace has two crates:

- `koalafalcon` — Falcon signatures over the KoalaBear prime with Poseidon hashing
- `aggregation` — SNARK-based aggregation of KoalaFalcon signatures

Trapdoor sampling is modified from
[pornin/rust-fn-dsa](https://github.com/pornin/rust-fn-dsa). 

## Build / test

```bash
cargo test -p koalafalcon --release
```

## Examples

```bash
cargo run -p koalafalcon --example signature --release
cargo run -p koalafalcon --example fn_dsa_signature --release
cargo run -p aggregation --example multisig --release
```

## Benchmarks

```bash
# Signature benches
cd koalafalcon/scripts
./bench_signature.sh
./bench_fn_dsa_signature.sh

# # Aggregation 
# cd aggregation/scripts
# ./bench_snark_multisig.sh
```

## Features

- `--features profile` — Perfetto tracing
- `--features parallel` — rayon parallelism 
