//! Manifest + weight validation (CPU host, no GPU required for this step).

use std::fs;
use std::path::PathBuf;

pub fn run(args: Vec<String>) -> Result<(), String> {
    let mut manifest = PathBuf::from("../common/weights/gpt2_manifest.json");
    let mut check = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--manifest" => {
                i += 1;
                manifest = PathBuf::from(args.get(i).ok_or("--manifest needs path")?);
            }
            "--check-weights" => check = true,
            "--self-test" => {
                return Err(
                    "CUDA-Oxide self-tests run via `cargo oxide run vecadd` after kernels land"
                        .into(),
                );
            }
            _ => return Err(format!("unknown arg: {}", args[i])),
        }
        i += 1;
    }

    let json = fs::read_to_string(&manifest).map_err(|e| e.to_string())?;
    if !json.contains("\"tensor_list\"") {
        return Err("manifest missing tensor_list — run tools/export_gpt2_weights.py".into());
    }
    let bin = manifest
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .join("gpt2_weights_fp32.bin");
    let meta = fs::metadata(&bin).map_err(|e| format!("missing {bin:?}: {e}"))?;
    println!(
        "gpt2-cuda-oxide host OK: manifest={}, weights={} bytes",
        manifest.display(),
        meta.len()
    );
    if check {
        println!("(full tensor parse in Rust — next milestone; mirror cuda-cpp loader)");
    }
    Ok(())
}
