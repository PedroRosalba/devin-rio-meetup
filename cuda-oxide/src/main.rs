//! Host entrypoint — kernels live in `src/kernels/` (CUDA-Oxide `#[kernel]` modules).
//!
//! Build on Ampere+ Linux:
//!   cargo oxide doctor
//!   cargo oxide run -- --manifest ../common/weights/gpt2_manifest.json --check-weights

mod host;

#[allow(dead_code)]
mod kernels;

fn main() {
    if let Err(e) = host::run(std::env::args().skip(1).collect()) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
