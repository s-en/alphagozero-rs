# alphagozero-rsa

wasm-pack build

cargo build --target=wasm32-unknown-unknown --release --lib


rustc --crate-type="dylib" src\lib.rs --out-dir="../game"