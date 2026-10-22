//! pearl-oracle: verify a Pearl share (PlainProof) with the official verifier.
//!   pearl-oracle verify <header76_hex> <nbits_hex> <proof_base64_file>
//! Prints "OK" or "FAIL <reason>". Exit code 0 only on OK.
use anyhow::{Context, Result, bail};
use base64::Engine;
use zk_pow::api::proof::IncompleteBlockHeader;
use zk_pow::api::seed::SeedDerivation;
use zk_pow::api::verify::verify_plain_proof;
use zk_pow::ffi::plain_proof::PlainProof;

fn unhex(s: &str) -> Result<Vec<u8>> {
    let s = s.trim();
    if s.len() % 2 != 0 { bail!("odd hex length"); }
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).context("bad hex")).collect()
}

fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 || args[1] != "verify" {
        bail!("usage: pearl-oracle verify <header76_hex> <nbits_hex> <proof_base64_file>");
    }
    let header = IncompleteBlockHeader::from_bytes(&unhex(&args[2])?)?;
    let nbits = u32::from_str_radix(args[3].trim_start_matches("0x"), 16)?;
    let b64 = std::fs::read_to_string(&args[4])?;
    let bytes = base64::engine::general_purpose::STANDARD.decode(b64.trim())?;
    let proof = PlainProof::deserialize_compat(&bytes)?;
    verify_plain_proof(&header, &proof, Some(nbits), SeedDerivation::Salted)?;
    Ok(())
}

fn main() {
    match run() {
        Ok(()) => println!("OK"),
        Err(e) => {
            println!("FAIL {e:#}");
            std::process::exit(1);
        }
    }
}
