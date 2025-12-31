<h1 align="center">
     atomic-plus
</h1>
<h4 align="center">
type extensions for the atomic standard library.
</h4>
<p align="center">
  <a href="https://github.com/0xhappyboy/atomic-plus/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache2.0-d1d1f6.svg?style=flat&labelColor=1C2C2E&color=BEC5C9&logo=googledocs&label=license&logoColor=BEC5C9" alt="License"></a>
    <a href="https://crates.io/crates/atomic-plus">
<img src="https://img.shields.io/badge/crates-atomic-plus-20B2AA.svg?style=flat&labelColor=0F1F2D&color=FFD700&logo=rust&logoColor=FFD700">
</a>
</p>
<p align="center">
<a href="./README_zh-CN.md">简体中文</a> | <a href="./README.md">English</a>
</p>

## Features

- `AtomicF64`: Atomic operations for 64-bit floating point numbers
- `AtomicF32`: Atomic operations for 32-bit floating point numbers
- `AtomicBoolArray`: Space-efficient atomic boolean array (1 bit per boolean)
- `AtomicShortString`: Fixed-length atomic string (max 32 ASCII characters)
- `AtomicTimestamp`: Nanosecond-precision atomic timestamp
- `AtomicPtr`: Generic atomic pointer wrapper
- `AtomicRc`: Atomic reference counting (similar to `std::rc::Rc` but with atomic operations)

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
atomic-ext = "0.1.0"
```

## Usage

```rust
use atomic_ext::{AtomicF64, AtomicBoolArray};
use std::sync::atomic::Ordering;

// Create an atomic float
let atomic_float = AtomicF64::new(10.5);
atomic_float.store(20.5, Ordering::Relaxed);
let value = atomic_float.load(Ordering::Relaxed);

// Create an atomic boolean array
let bool_array = AtomicBoolArray::new(100);
bool_array.set(42, true, Ordering::Relaxed);
let is_true = bool_array.get(42, Ordering::Relaxed);
```

## Features

- **Thread-safe**: All types are designed for concurrent access
- **No unsafe code required**: Safe API surface
- **High performance**: Built on standard library atomics
- **Comprehensive tests**: Includes concurrent test scenarios

## Notes

- `AtomicF64` and `AtomicF32` store floats using their bit patterns
- `AtomicBoolArray` uses 1/64th the memory of `Vec<AtomicBool>`
- `AtomicRc` is not `Send`/`Sync` by default (contains raw pointers)
- `AtomicTimestamp` provides nanosecond precision

## Testing

Run tests with:

```bash
cargo test
```
