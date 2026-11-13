//! A cap on the memory of a test run.
//!
//! A propagation that fails to truncate or to merge its terms grows without end, and by the
//! time a test would notice, the machine has no memory left. The tests need a few hundred
//! megabytes, so an allocation that takes a test run beyond the cap is refused, which ends
//! the run with an allocation error.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

/// The most memory that a test run may hold in large blocks at one time, in bytes.
const CAP: usize = 2 << 30;

/// Smaller blocks than this are not counted. A sum that grows without end is in a few
/// blocks that are far larger, and counting every small block would slow all the threads
/// of a test down.
const COUNTED_FROM: usize = 1 << 16;

static IN_USE: AtomicUsize = AtomicUsize::new(0);

struct Capped;

#[global_allocator]
static ALLOCATOR: Capped = Capped;

/// What a block of `size` bytes counts for.
fn counted(size: usize) -> usize {
    if size < COUNTED_FROM {
        0
    } else {
        size
    }
}

/// Counts `bytes` more as in use, unless that is beyond the cap.
fn take(bytes: usize) -> bool {
    if bytes == 0 {
        return true;
    }
    if IN_USE.fetch_add(bytes, Relaxed) + bytes > CAP {
        IN_USE.fetch_sub(bytes, Relaxed);
        return false;
    }
    true
}

fn give_back(bytes: usize) {
    if bytes != 0 {
        IN_USE.fetch_sub(bytes, Relaxed);
    }
}

// SAFETY: every request is passed on to the system allocator as it is, or refused by
// returning null, which is what an allocator returns when it has no memory to give.
unsafe impl GlobalAlloc for Capped {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if !take(counted(layout.size())) {
            return std::ptr::null_mut();
        }
        System.alloc(layout)
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if !take(counted(layout.size())) {
            return std::ptr::null_mut();
        }
        System.alloc_zeroed(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        give_back(counted(layout.size()));
        System.dealloc(ptr, layout);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let (before, after) = (counted(layout.size()), counted(new_size));
        if !take(after.saturating_sub(before)) {
            return std::ptr::null_mut();
        }
        let moved = System.realloc(ptr, layout, new_size);
        if moved.is_null() {
            give_back(after.saturating_sub(before));
        } else {
            give_back(before.saturating_sub(after));
        }
        moved
    }
}
