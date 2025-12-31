use std::sync::atomic::{AtomicI32, AtomicU32};
use std::sync::atomic::{AtomicI64, AtomicU64, AtomicUsize, Ordering};

/// Atomic floating point number for 64-bit floats (f64)
/// Uses AtomicU64 internally to store the bit pattern of the float
#[derive(Debug)]
pub struct AtomicF64 {
    inner: AtomicU64,
}

impl AtomicF64 {
    /// Creates a new AtomicF64 with the given initial value
    pub fn new(value: f64) -> Self {
        Self {
            inner: AtomicU64::new(value.to_bits()),
        }
    }

    /// Loads the value from the atomic float
    pub fn load(&self, ordering: Ordering) -> f64 {
        f64::from_bits(self.inner.load(ordering))
    }

    /// Stores a value into the atomic float
    pub fn store(&self, value: f64, ordering: Ordering) {
        self.inner.store(value.to_bits(), ordering)
    }

    /// Atomically compares and exchanges the value
    pub fn compare_exchange(
        &self,
        current: f64,
        new: f64,
        success: Ordering,
        failure: Ordering,
    ) -> Result<f64, f64> {
        self.inner
            .compare_exchange(current.to_bits(), new.to_bits(), success, failure)
            .map(f64::from_bits)
            .map_err(f64::from_bits)
    }

    /// Atomically adds to the current value and returns the previous value
    pub fn fetch_add(&self, value: f64, ordering: Ordering) -> f64 {
        let mut current = self.load(ordering);
        loop {
            let new = current + value;
            match self.compare_exchange(current, new, ordering, Ordering::Relaxed) {
                Ok(_) => return current,
                Err(actual) => current = actual,
            }
        }
    }

    /// Atomically subtracts from the current value and returns the previous value
    pub fn fetch_sub(&self, value: f64, ordering: Ordering) -> f64 {
        self.fetch_add(-value, ordering)
    }

    /// Atomically multiplies the current value and returns the previous value
    pub fn fetch_mul(&self, value: f64, ordering: Ordering) -> f64 {
        let mut current = self.load(ordering);
        loop {
            let new = current * value;
            match self.compare_exchange(current, new, ordering, Ordering::Relaxed) {
                Ok(_) => return current,
                Err(actual) => current = actual,
            }
        }
    }

    /// Atomically divides the current value and returns the previous value
    pub fn fetch_div(&self, value: f64, ordering: Ordering) -> f64 {
        let mut current = self.load(ordering);
        loop {
            let new = current / value;
            match self.compare_exchange(current, new, ordering, Ordering::Relaxed) {
                Ok(_) => return current,
                Err(actual) => current = actual,
            }
        }
    }

    /// Swaps the value and returns the previous value
    pub fn swap(&self, value: f64, ordering: Ordering) -> f64 {
        f64::from_bits(self.inner.swap(value.to_bits(), ordering))
    }
}

/// Atomic floating point number for 32-bit floats (f32)
/// Uses AtomicU32 internally to store the bit pattern of the float
#[derive(Debug)]
pub struct AtomicF32 {
    inner: AtomicU32,
}

impl AtomicF32 {
    /// Creates a new AtomicF32 with the given initial value
    pub fn new(value: f32) -> Self {
        Self {
            inner: AtomicU32::new(value.to_bits()),
        }
    }

    /// Loads the value from the atomic float
    pub fn load(&self, ordering: Ordering) -> f32 {
        f32::from_bits(self.inner.load(ordering))
    }

    /// Stores a value into the atomic float
    pub fn store(&self, value: f32, ordering: Ordering) {
        self.inner.store(value.to_bits(), ordering)
    }

    /// Atomically compares and exchanges the value
    pub fn compare_exchange(
        &self,
        current: f32,
        new: f32,
        success: Ordering,
        failure: Ordering,
    ) -> Result<f32, f32> {
        self.inner
            .compare_exchange(current.to_bits(), new.to_bits(), success, failure)
            .map(f32::from_bits)
            .map_err(f32::from_bits)
    }

    /// Atomically adds to the current value and returns the previous value
    pub fn fetch_add(&self, value: f32, ordering: Ordering) -> f32 {
        let mut current = self.load(ordering);
        loop {
            let new = current + value;
            match self.compare_exchange(current, new, ordering, Ordering::Relaxed) {
                Ok(_) => return current,
                Err(actual) => current = actual,
            }
        }
    }
}

/// Atomic array of boolean values using bit-level operations
/// Each boolean occupies 1 bit, 64 booleans per AtomicU64
pub struct AtomicBoolArray {
    inner: Vec<AtomicU64>,
    len: usize,
}

impl AtomicBoolArray {
    /// Creates a new AtomicBoolArray with the specified length
    pub fn new(len: usize) -> Self {
        let chunks = (len + 63) / 64;
        Self {
            inner: (0..chunks).map(|_| AtomicU64::new(0)).collect(),
            len,
        }
    }

    /// Returns the length of the boolean array
    pub fn len(&self) -> usize {
        self.len
    }

    /// Sets the boolean value at the specified index
    pub fn set(&self, index: usize, value: bool, ordering: Ordering) {
        let (chunk_idx, bit_idx) = self.calculate_position(index);

        let chunk = &self.inner[chunk_idx];
        let mut current = chunk.load(Ordering::Relaxed);

        loop {
            let new = if value {
                current | (1 << bit_idx)
            } else {
                current & !(1 << bit_idx)
            };

            match chunk.compare_exchange(current, new, ordering, Ordering::Relaxed) {
                Ok(_) => break,
                Err(actual) => current = actual,
            }
        }
    }

    /// Gets the boolean value at the specified index
    pub fn get(&self, index: usize, ordering: Ordering) -> bool {
        let (chunk_idx, bit_idx) = self.calculate_position(index);
        let chunk = self.inner[chunk_idx].load(ordering);
        (chunk >> bit_idx) & 1 == 1
    }

    /// Toggles the boolean value at the specified index
    pub fn toggle(&self, index: usize, ordering: Ordering) -> bool {
        let (chunk_idx, bit_idx) = self.calculate_position(index);
        let chunk = &self.inner[chunk_idx];
        let mut current = chunk.load(Ordering::Relaxed);

        loop {
            let new = current ^ (1 << bit_idx);
            match chunk.compare_exchange(current, new, ordering, Ordering::Relaxed) {
                Ok(_) => return (new >> bit_idx) & 1 == 1,
                Err(actual) => current = actual,
            }
        }
    }

    /// Calculates the chunk index and bit index for a given array index
    fn calculate_position(&self, index: usize) -> (usize, usize) {
        assert!(index < self.len, "Index out of bounds");
        (index / 64, index % 64)
    }
}

/// Generic atomic pointer wrapper
#[derive(Debug)]
pub struct AtomicPtr<T> {
    inner: AtomicU64,
    _marker: std::marker::PhantomData<*mut T>,
}

impl<T> AtomicPtr<T> {
    /// Creates a new AtomicPtr with the given pointer
    pub fn new(ptr: *mut T) -> Self {
        Self {
            inner: AtomicU64::new(ptr as u64),
            _marker: std::marker::PhantomData,
        }
    }

    /// Loads the pointer value
    pub fn load(&self, ordering: Ordering) -> *mut T {
        self.inner.load(ordering) as *mut T
    }

    /// Stores a pointer value
    pub fn store(&self, ptr: *mut T, ordering: Ordering) {
        self.inner.store(ptr as u64, ordering)
    }

    /// Atomically compares and exchanges the pointer
    pub fn compare_exchange(
        &self,
        current: *mut T,
        new: *mut T,
        success: Ordering,
        failure: Ordering,
    ) -> Result<*mut T, *mut T> {
        self.inner
            .compare_exchange(current as u64, new as u64, success, failure)
            .map(|val| val as *mut T)
            .map_err(|val| val as *mut T)
    }

    /// Swaps the pointer value and returns the previous value
    pub fn swap(&self, ptr: *mut T, ordering: Ordering) -> *mut T {
        self.inner.swap(ptr as u64, ordering) as *mut T
    }
}

/// Atomic short string with fixed maximum length (32 ASCII characters)
#[derive(Debug)]
pub struct AtomicShortString {
    inner: [AtomicU64; 4], // 256 bits, stores 32 ASCII characters
}

impl AtomicShortString {
    /// Creates a new empty AtomicShortString
    pub fn new() -> Self {
        Self {
            inner: [
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
            ],
        }
    }

    /// Stores a string into the atomic short string
    /// Returns an error if the string is too long (max 32 characters)
    pub fn store(&self, s: &str, ordering: Ordering) -> Result<(), String> {
        if s.len() > 32 {
            return Err("String too long".to_string());
        }

        let mut bytes = s.as_bytes().to_vec();
        bytes.resize(32, 0);

        for i in 0..4 {
            let chunk_bytes = &bytes[i * 8..(i + 1) * 8];
            let mut value = 0u64;
            for (j, &byte) in chunk_bytes.iter().enumerate() {
                value |= (byte as u64) << (j * 8);
            }
            self.inner[i].store(value, ordering);
        }

        Ok(())
    }

    /// Loads the string from the atomic short string
    pub fn load(&self, ordering: Ordering) -> String {
        let mut bytes = Vec::with_capacity(32);

        for i in 0..4 {
            let value = self.inner[i].load(ordering);
            for j in 0..8 {
                let byte = ((value >> (j * 8)) & 0xFF) as u8;
                bytes.push(byte);
            }
        }

        // Find the first null byte as string terminator
        let len = bytes.iter().position(|&b| b == 0).unwrap_or(32);
        String::from_utf8_lossy(&bytes[..len]).to_string()
    }
}

/// Atomic reference count similar to std::rc::Rc but with atomic operations
#[derive(Debug)]
pub struct AtomicRc<T> {
    ptr: AtomicPtr<T>,
    count: AtomicUsize,
}

impl<T> AtomicRc<T> {
    /// Creates a new AtomicRc with the given value
    pub fn new(value: T) -> Self {
        let boxed = Box::new(value);
        let ptr = Box::into_raw(boxed);

        Self {
            ptr: AtomicPtr::new(ptr),
            count: AtomicUsize::new(1),
        }
    }

    /// Creates a new reference to the same value
    pub fn clone(&self) -> Self {
        self.count.fetch_add(1, Ordering::Relaxed);
        Self {
            ptr: AtomicPtr::new(self.ptr.load(Ordering::Relaxed)),
            count: AtomicUsize::new(1),
        }
    }

    /// Returns the strong reference count
    pub fn strong_count(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }

    /// Returns a reference to the contained value if it exists
    pub fn get(&self) -> Option<&T> {
        let ptr = self.ptr.load(Ordering::Relaxed);
        if ptr.is_null() {
            None
        } else {
            unsafe { Some(&*ptr) }
        }
    }
}

impl<T> Drop for AtomicRc<T> {
    fn drop(&mut self) {
        let count = self.count.fetch_sub(1, Ordering::Release);
        if count == 1 {
            let ptr = self.ptr.swap(std::ptr::null_mut(), Ordering::Relaxed);
            if !ptr.is_null() {
                unsafe {
                    Box::from_raw(ptr);
                }
            }
        }
    }
}

/// Atomic timestamp with nanosecond precision
#[derive(Debug)]
pub struct AtomicTimestamp {
    inner: AtomicU64,
}

impl AtomicTimestamp {
    /// Creates a new AtomicTimestamp initialized to 0
    pub fn new() -> Self {
        Self {
            inner: AtomicU64::new(0),
        }
    }

    /// Updates the timestamp to the current time and returns the previous value
    pub fn now(&self) -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        let duration = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
        let nanos = duration.as_nanos() as u64;
        let old = self.inner.swap(nanos, Ordering::Relaxed);
        old
    }

    /// Gets the current timestamp value
    pub fn get(&self) -> u64 {
        self.inner.load(Ordering::Relaxed)
    }

    /// Updates the timestamp to the given value and returns the previous value
    pub fn update(&self, timestamp: u64) -> u64 {
        self.inner.swap(timestamp, Ordering::Relaxed)
    }

    /// Calculates the elapsed nanoseconds since the stored timestamp
    pub fn elapsed_ns(&self) -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        let current = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
        let current_ns = current.as_nanos() as u64;
        let last = self.inner.load(Ordering::Relaxed);
        if current_ns > last {
            current_ns - last
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::Ordering;

    #[test]
    fn test_atomic_f64_basic() {
        let atomic = AtomicF64::new(10.5);
        assert_eq!(atomic.load(Ordering::Relaxed), 10.5);
        atomic.store(20.5, Ordering::Relaxed);
        assert_eq!(atomic.load(Ordering::Relaxed), 20.5);
        let prev = atomic.fetch_add(5.0, Ordering::Relaxed);
        assert_eq!(prev, 20.5);
        assert_eq!(atomic.load(Ordering::Relaxed), 25.5);
    }

    #[test]
    fn test_atomic_f32_basic() {
        let atomic = AtomicF32::new(5.5);
        assert_eq!(atomic.load(Ordering::Relaxed), 5.5);
        atomic.store(15.5, Ordering::Relaxed);
        assert_eq!(atomic.load(Ordering::Relaxed), 15.5);
    }

    #[test]
    fn test_atomic_bool_array_basic() {
        let array = AtomicBoolArray::new(10);
        array.set(3, true, Ordering::Relaxed);
        assert!(array.get(3, Ordering::Relaxed));
        array.set(7, false, Ordering::Relaxed);
        assert!(!array.get(7, Ordering::Relaxed));
    }

    #[test]
    fn test_atomic_short_string_basic() {
        let atomic = AtomicShortString::new();
        atomic.store("hello", Ordering::Relaxed).unwrap();
        assert_eq!(atomic.load(Ordering::Relaxed), "hello");
    }

    #[test]
    fn test_atomic_timestamp_basic() {
        let timestamp = AtomicTimestamp::new();
        assert_eq!(timestamp.get(), 0);
        let prev = timestamp.now();
        assert_eq!(prev, 0);
        assert!(timestamp.get() > 0);
    }

    #[tokio::test]
    async fn test_atomic_f64_concurrent_simple() {
        let atomic = Arc::new(AtomicF64::new(0.0));
        let mut tasks = Vec::new();
        for i in 0..10 {
            let atomic_clone = atomic.clone();
            tasks.push(tokio::spawn(async move {
                atomic_clone.fetch_add(1.0, Ordering::Relaxed);
                i
            }));
        }
        let mut results = Vec::new();
        for task in tasks {
            results.push(task.await.unwrap());
        }
        assert_eq!(results.len(), 10);
        assert_eq!(atomic.load(Ordering::Relaxed), 10.0);
    }

    #[tokio::test]
    async fn test_atomic_f32_concurrent_simple() {
        let atomic = Arc::new(AtomicF32::new(0.0));
        let mut tasks = Vec::new();
        for _ in 0..5 {
            let atomic_clone = atomic.clone();

            tasks.push(tokio::spawn(async move {
                atomic_clone.fetch_add(2.0, Ordering::Relaxed);
            }));
        }
        for task in tasks {
            task.await.unwrap();
        }
        assert_eq!(atomic.load(Ordering::Relaxed), 10.0);
    }

    #[tokio::test]
    async fn test_atomic_bool_array_concurrent_simple() {
        let array = Arc::new(AtomicBoolArray::new(20));
        let mut tasks = Vec::new();
        for i in 0..10 {
            let array_clone = array.clone();
            tasks.push(tokio::spawn(async move {
                array_clone.set(i, true, Ordering::Relaxed);
                array_clone.set(i + 10, false, Ordering::Relaxed);
            }));
        }
        for task in tasks {
            task.await.unwrap();
        }
        for i in 0..10 {
            assert!(array.get(i, Ordering::Relaxed));
            assert!(!array.get(i + 10, Ordering::Relaxed));
        }
    }

    #[tokio::test]
    async fn test_atomic_short_string_concurrent() {
        let atomic = Arc::new(AtomicShortString::new());
        atomic.store("initial", Ordering::Relaxed).unwrap();
        let mut tasks = Vec::new();
        for _ in 0..5 {
            let atomic_clone = atomic.clone();
            tasks.push(tokio::spawn(async move {
                let value = atomic_clone.load(Ordering::Relaxed);
                assert_eq!(value, "initial");
                value
            }));
        }
        for task in tasks {
            task.await.unwrap();
        }
        assert_eq!(atomic.load(Ordering::Relaxed), "initial");
    }

    #[tokio::test]
    async fn test_atomic_timestamp_concurrent() {
        let timestamp = Arc::new(AtomicTimestamp::new());
        let mut tasks = Vec::new();
        for i in 0..10 {
            let timestamp_clone = timestamp.clone();
            tasks.push(tokio::spawn(async move {
                timestamp_clone.update((i + 1) as u64 * 1_000_000);
            }));
        }
        for task in tasks {
            task.await.unwrap();
        }
        let final_value = timestamp.get();
        assert!(final_value >= 1_000_000 && final_value <= 10_000_000);
    }

    #[tokio::test]
    async fn test_atomic_f64_race_condition() {
        let atomic = Arc::new(AtomicF64::new(0.0));
        let mut tasks = Vec::new();
        for _ in 0..100 {
            let atomic_clone = atomic.clone();
            tasks.push(tokio::spawn(async move {
                atomic_clone.fetch_add(1.0, Ordering::Relaxed);
            }));
        }
        for task in tasks {
            task.await.unwrap();
        }
        assert_eq!(atomic.load(Ordering::Relaxed), 100.0);
    }

    #[tokio::test]
    async fn test_atomic_ordering() {
        let atomic = Arc::new(AtomicF64::new(0.0));
        let mut tasks = Vec::new();
        tasks.push(tokio::spawn({
            let atomic_clone = atomic.clone();
            async move {
                atomic_clone.store(10.0, Ordering::Relaxed);
            }
        }));
        tasks.push(tokio::spawn({
            let atomic_clone = atomic.clone();
            async move {
                atomic_clone.fetch_add(5.0, Ordering::Relaxed);
            }
        }));
        tasks.push(tokio::spawn({
            let atomic_clone = atomic.clone();
            async move {
                atomic_clone.fetch_sub(3.0, Ordering::Relaxed);
            }
        }));
        for task in tasks {
            task.await.unwrap();
        }
        let final_value = atomic.load(Ordering::Relaxed);
        assert!(final_value >= 0.0);
    }

    #[test]
    fn test_atomic_ptr_basic() {
        let mut value = 42;
        let ptr = &mut value as *mut i32;

        let atomic = AtomicPtr::new(ptr);
        assert_eq!(atomic.load(Ordering::Relaxed), ptr);
    }

    #[test]
    fn test_atomic_rc_basic() {
        let atomic = AtomicRc::new(42);
        assert_eq!(atomic.strong_count(), 1);
        let clone = atomic.clone();
        assert_eq!(atomic.strong_count(), 2);
        assert_eq!(atomic.get(), Some(&42));
    }

    #[test]
    fn test_atomic_f64_edge_cases() {
        let atomic = AtomicF64::new(0.0);
        atomic.fetch_add(-5.0, Ordering::Relaxed);
        assert_eq!(atomic.load(Ordering::Relaxed), -5.0);
        atomic.store(1e308, Ordering::Relaxed);
        assert_eq!(atomic.load(Ordering::Relaxed), 1e308);
        let prev = atomic.swap(100.0, Ordering::Relaxed);
        assert_eq!(prev, 1e308);
        assert_eq!(atomic.load(Ordering::Relaxed), 100.0);
    }

    #[tokio::test]
    async fn test_atomic_compare_exchange_concurrent() {
        let atomic = Arc::new(AtomicF64::new(0.0));
        let mut tasks = Vec::new();
        let success_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        for _ in 0..10 {
            let atomic_clone = atomic.clone();
            let success_clone = success_count.clone();
            tasks.push(tokio::spawn(async move {
                match atomic_clone.compare_exchange(0.0, 1.0, Ordering::Relaxed, Ordering::Relaxed)
                {
                    Ok(_) => {
                        success_clone.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(_) => {}
                }
            }));
        }
        for task in tasks {
            task.await.unwrap();
        }
        assert_eq!(success_count.load(Ordering::Relaxed), 1);
        assert_eq!(atomic.load(Ordering::Relaxed), 1.0);
    }
}
