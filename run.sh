#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if ! command -v cargo >/dev/null 2>&1; then
    echo 'Rust/Cargo is required. Install it, then run this script again.' >&2
    exit 1
fi
exec cargo run --release --locked -- "$@"
