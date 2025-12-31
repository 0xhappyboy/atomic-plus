<h1 align="center">
     atomic-plus
</h1>
<h4 align="center">
Atomic标准库的类型扩展.
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

## 功能特性

- `AtomicF64`: 64 位浮点数的原子操作
- `AtomicF32`: 32 位浮点数的原子操作
- `AtomicBoolArray`: 空间高效的原子布尔数组（每个布尔值占用 1 比特）
- `AtomicShortString`: 固定长度的原子字符串（最多 32 个 ASCII 字符）
- `AtomicTimestamp`: 纳秒精度的原子时间戳
- `AtomicPtr`: 泛型原子指针包装器
- `AtomicRc`: 原子引用计数（类似于 `std::rc::Rc` 但支持原子操作）

## 安装

在 `Cargo.toml` 中添加：

```toml
[dependencies]
atomic-ext = "0.1.0"
```

## 使用示例

```rust
use atomic_ext::{AtomicF64, AtomicBoolArray};
use std::sync::atomic::Ordering;

// 创建原子浮点数
let atomic_float = AtomicF64::new(10.5);
atomic_float.store(20.5, Ordering::Relaxed);
let value = atomic_float.load(Ordering::Relaxed);

// 创建原子布尔数组
let bool_array = AtomicBoolArray::new(100);
bool_array.set(42, true, Ordering::Relaxed);
let is_true = bool_array.get(42, Ordering::Relaxed);
```

## 特性

- **线程安全**: 所有类型都设计用于并发访问
- **无需 unsafe 代码**: 安全的 API 接口
- **高性能**: 基于标准库原子操作构建
- **完整测试**: 包含并发测试场景

## 注意事项

- `AtomicF64` 和 `AtomicF32` 使用浮点数的位模式进行存储
- `AtomicBoolArray` 的内存使用量仅为 `Vec<AtomicBool>` 的 1/64
- `AtomicRc` 默认不是 `Send`/`Sync`（包含裸指针）
- `AtomicTimestamp` 提供纳秒级精度

## 测试

运行测试：

```bash
cargo test
```
