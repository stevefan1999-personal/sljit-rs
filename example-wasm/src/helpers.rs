#[cfg(not(target_pointer_width = "64"))]
pub mod i64 {

    pub extern "C" fn i64_add(a_ptr: *const u64, b_ptr: *const u64, out_ptr: *mut u64) {
        unsafe {
            *out_ptr = (*a_ptr).wrapping_add(*b_ptr);
        }
    }

    pub extern "C" fn i64_sub(a_ptr: *const u64, b_ptr: *const u64, out_ptr: *mut u64) {
        unsafe {
            *out_ptr = (*a_ptr).wrapping_sub(*b_ptr);
        }
    }

    pub extern "C" fn i64_mul(a_ptr: *const u64, b_ptr: *const u64, out_ptr: *mut u64) {
        unsafe {
            *out_ptr = (*a_ptr).wrapping_mul(*b_ptr);
        }
    }

    pub extern "C" fn i64_and(a_ptr: *const u64, b_ptr: *const u64, out_ptr: *mut u64) {
        unsafe {
            *out_ptr = (*a_ptr) & (*b_ptr);
        }
    }

    pub extern "C" fn i64_or(a_ptr: *const u64, b_ptr: *const u64, out_ptr: *mut u64) {
        unsafe {
            *out_ptr = (*a_ptr) | (*b_ptr);
        }
    }

    pub extern "C" fn i64_xor(a_ptr: *const u64, b_ptr: *const u64, out_ptr: *mut u64) {
        unsafe {
            *out_ptr = (*a_ptr) ^ (*b_ptr);
        }
    }

    pub extern "C" fn i64_shl(a_ptr: *const u64, b_ptr: *const u64, out_ptr: *mut u64) {
        unsafe {
            *out_ptr = (*a_ptr).wrapping_shl(((*b_ptr) & 63) as u32);
        }
    }

    pub extern "C" fn i64_shr_u(a_ptr: *const u64, b_ptr: *const u64, out_ptr: *mut u64) {
        unsafe {
            *out_ptr = (*a_ptr).wrapping_shr(((*b_ptr) & 63) as u32);
        }
    }

    pub extern "C" fn i64_shr_s(a_ptr: *const u64, b_ptr: *const u64, out_ptr: *mut u64) {
        unsafe {
            *out_ptr = ((*a_ptr) as i64).wrapping_shr(((*b_ptr) & 63) as u32) as u64;
        }
    }

    pub extern "C" fn i64_rotl(a_ptr: *const u64, b_ptr: *const u64, out_ptr: *mut u64) {
        unsafe {
            *out_ptr = (*a_ptr).rotate_left(((*b_ptr) & 63) as u32);
        }
    }

    pub extern "C" fn i64_rotr(a_ptr: *const u64, b_ptr: *const u64, out_ptr: *mut u64) {
        unsafe {
            *out_ptr = (*a_ptr).rotate_right(((*b_ptr) & 63) as u32);
        }
    }

    pub extern "C" fn i64_popcnt(a_ptr: *const u64, out_ptr: *mut u64) {
        unsafe {
            *out_ptr = (*a_ptr).count_ones() as u64;
        }
    }

    pub extern "C" fn i64_clz(a_ptr: *const u64, out_ptr: *mut u64) {
        unsafe {
            *out_ptr = (*a_ptr).leading_zeros() as u64;
        }
    }

    pub extern "C" fn i64_ctz(a_ptr: *const u64, out_ptr: *mut u64) {
        unsafe {
            *out_ptr = (*a_ptr).trailing_zeros() as u64;
        }
    }

    pub extern "C" fn i64_div_s(a_ptr: *const u64, b_ptr: *const u64, out_ptr: *mut u64) {
        unsafe {
            let a = (*a_ptr) as i64;
            let b = (*b_ptr) as i64;

            *out_ptr = if b == 0 || (a == i64::MIN && b == -1) {
                a // Overflow case
            } else {
                a / b
            } as u64;
        }
    }

    pub extern "C" fn i64_div_u(a_ptr: *const u64, b_ptr: *const u64, out_ptr: *mut u64) {
        unsafe {
            let a = *a_ptr;
            let b = *b_ptr;
            *out_ptr = if b == 0 { 0 } else { a / b };
        }
    }

    pub extern "C" fn i64_rem_s(a_ptr: *const u64, b_ptr: *const u64, out_ptr: *mut u64) {
        unsafe {
            let a = (*a_ptr) as i64;
            let b = (*b_ptr) as i64;

            *out_ptr = if b == 0 || (a == i64::MIN && b == -1) {
                0 // Remainder is 0 in overflow case
            } else {
                a % b
            } as u64;
        }
    }

    pub extern "C" fn i64_rem_u(a_ptr: *const u64, b_ptr: *const u64, out_ptr: *mut u64) {
        unsafe {
            let a = *a_ptr;
            let b = *b_ptr;
            *out_ptr = if b == 0 { 0 } else { a % b };
        }
    }
    pub extern "C" fn i64_eq(a_ptr: *const u64, b_ptr: *const u64) -> u32 {
        unsafe { ((*a_ptr) == (*b_ptr)) as u32 }
    }

    pub extern "C" fn i64_neq(a_ptr: *const u64, b_ptr: *const u64) -> u32 {
        unsafe { ((*a_ptr) != (*b_ptr)) as u32 }
    }

    pub extern "C" fn i64_lt(a_ptr: *const u64, b_ptr: *const u64) -> u32 {
        unsafe { ((*a_ptr) < (*b_ptr)) as u32 }
    }

    pub extern "C" fn i64_lt_s(a_ptr: *const u64, b_ptr: *const u64) -> u32 {
        unsafe { (((*a_ptr) as i64) < ((*b_ptr) as i64)) as u32 }
    }

    pub extern "C" fn i64_gt(a_ptr: *const u64, b_ptr: *const u64) -> u32 {
        unsafe { ((*a_ptr) > (*b_ptr)) as u32 }
    }

    pub extern "C" fn i64_gt_s(a_ptr: *const u64, b_ptr: *const u64) -> u32 {
        unsafe { ((*a_ptr) as i64 > (*b_ptr) as i64) as u32 }
    }

    pub extern "C" fn i64_le(a_ptr: *const u64, b_ptr: *const u64) -> u32 {
        unsafe { ((*a_ptr) <= (*b_ptr)) as u32 }
    }

    pub extern "C" fn i64_le_s(a_ptr: *const u64, b_ptr: *const u64) -> u32 {
        unsafe { ((*a_ptr) as i64 <= (*b_ptr) as i64) as u32 }
    }

    pub extern "C" fn i64_ge(a_ptr: *const u64, b_ptr: *const u64) -> u32 {
        unsafe { ((*a_ptr) >= (*b_ptr)) as u32 }
    }

    pub extern "C" fn i64_ge_s(a_ptr: *const u64, b_ptr: *const u64) -> u32 {
        unsafe { ((*a_ptr) as i64 >= (*b_ptr) as i64) as u32 }
    }

    // Helper functions for 64-bit to float conversion
    pub extern "C" fn i64_to_f32_signed(low: u32, high: i32) -> f32 {
        let val = (low as u64) | ((high as i64 as u64) << 32);
        val as i64 as f32
    }

    pub extern "C" fn i64_to_f64_signed(low: u32, high: i32) -> f64 {
        let val = (low as u64) | ((high as i64 as u64) << 32);
        val as i64 as f64
    }

    pub extern "C" fn i64_to_f32_unsigned(low: u32, high: u32) -> f32 {
        let val = (low as u64) | ((high as u64) << 32);
        val as f32
    }

    pub extern "C" fn i64_to_f64_unsigned(low: u32, high: u32) -> f64 {
        let val = (low as u64) | ((high as u64) << 32);
        val as f64
    }
}

// Extend operations (from i32) - write to output pointer

pub extern "C" fn __i64_extend_i32_s(val: i32, out_ptr: *mut u64) {
    unsafe {
        *out_ptr = (val as i64) as u64;
    }
}

pub extern "C" fn __i64_extend_i32_u(val: u32, out_ptr: *mut u64) {
    unsafe {
        *out_ptr = val as u64;
    }
}

pub extern "C" fn popcnt32(x: u32) -> u32 {
    x.count_ones()
}

#[cfg(target_pointer_width = "64")]
pub extern "C" fn popcnt64(x: u64) -> u32 {
    x.count_ones()
}

pub extern "C" fn fmin32(x: f32, y: f32) -> f32 {
    x.min(y)
}
pub extern "C" fn fmin64(x: f64, y: f64) -> f64 {
    x.min(y)
}
pub extern "C" fn fmax32(x: f32, y: f32) -> f32 {
    x.max(y)
}
pub extern "C" fn fmax64(x: f64, y: f64) -> f64 {
    x.max(y)
}
pub extern "C" fn copysign32(x: f32, y: f32) -> f32 {
    x.copysign(y)
}
pub extern "C" fn copysign64(x: f64, y: f64) -> f64 {
    x.copysign(y)
}
pub extern "C" fn sqrtf32(x: f32) -> f32 {
    x.sqrt()
}
pub extern "C" fn sqrtf64(x: f64) -> f64 {
    x.sqrt()
}
pub extern "C" fn ceilf32(x: f32) -> f32 {
    x.ceil()
}
pub extern "C" fn ceilf64(x: f64) -> f64 {
    x.ceil()
}
pub extern "C" fn floorf32(x: f32) -> f32 {
    x.floor()
}
pub extern "C" fn floorf64(x: f64) -> f64 {
    x.floor()
}

pub extern "C" fn truncf32(x: f32) -> f32 {
    x.trunc()
}
pub extern "C" fn truncf64(x: f64) -> f64 {
    x.trunc()
}
pub extern "C" fn nintf32(x: f32) -> f32 {
    x.round_ties_even()
}

pub extern "C" fn nintf64(x: f64) -> f64 {
    x.round_ties_even()
}

#[cfg(not(target_pointer_width = "64"))]
pub use i64::*;
