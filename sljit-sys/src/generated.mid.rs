impl Compiler {
    #[inline(always)]
    pub fn alloc_memory(&mut self, size: sljit_s32) -> *mut ::core::ffi::c_void {
        unsafe { sljit_alloc_memory(self.0, size) }
    }
    #[inline(always)]
    pub fn compiler_get_allocator_data(&mut self) -> *mut ::core::ffi::c_void {
        unsafe { sljit_compiler_get_allocator_data(self.0) }
    }
    #[inline(always)]
    pub fn compiler_get_user_data(&mut self) -> *mut ::core::ffi::c_void {
        unsafe { sljit_compiler_get_user_data(self.0) }
    }
    #[inline(always)]
    pub fn compiler_set_user_data(&mut self, user_data: *mut ::core::ffi::c_void) {
        unsafe { sljit_compiler_set_user_data(self.0, user_data) }
    }
    #[inline(always)]
    pub fn emit_aligned_label(
        &mut self,
        alignment: sljit_s32,
        buffers: *mut sljit_read_only_buffer,
    ) -> Label {
        unsafe { sljit_emit_aligned_label(self.0, alignment, buffers) }.into()
    }
    #[inline(always)]
    pub fn emit_atomic_load(
        &mut self,
        op: sljit_s32,
        dst_reg: sljit_s32,
        mem_reg: sljit_s32,
    ) -> Result<&mut Self, ErrorCode> {
        ErrorCode::i32_as_result(unsafe { sljit_emit_atomic_load(self.0, op, dst_reg, mem_reg) })
            .and(Ok(self))
    }
    #[inline(always)]
    pub fn emit_atomic_store(
        &mut self,
        op: sljit_s32,
        src_reg: sljit_s32,
        mem_reg: sljit_s32,
        temp_reg: sljit_s32,
    ) -> Result<&mut Self, ErrorCode> {
        ErrorCode::i32_as_result(unsafe {
            sljit_emit_atomic_store(self.0, op, src_reg, mem_reg, temp_reg)
        })
        .and(Ok(self))
    }
    #[inline(always)]
    pub fn emit_call(&mut self, type_: sljit_s32, arg_types: sljit_s32) -> Jump {
        unsafe { sljit_emit_call(self.0, type_, arg_types) }.into()
    }
    #[inline(always)]
    pub fn emit_cmp(
        &mut self,
        type_: sljit_s32,
        src1: sljit_s32,
        src1w: sljit_sw,
        src2: sljit_s32,
        src2w: sljit_sw,
    ) -> Jump {
        unsafe { sljit_emit_cmp(self.0, type_, src1, src1w, src2, src2w) }.into()
    }
    #[inline(always)]
    pub fn emit_const(
        &mut self,
        op: sljit_s32,
        dst: sljit_s32,
        dstw: sljit_sw,
        init_value: sljit_sw,
    ) -> Constant {
        unsafe { sljit_emit_const(self.0, op, dst, dstw, init_value) }.into()
    }
    #[inline(always)]
    pub fn emit_enter(
        &mut self,
        options: sljit_s32,
        arg_types: sljit_s32,
        scratches: sljit_s32,
        saveds: sljit_s32,
        local_size: sljit_s32,
    ) -> Result<&mut Self, ErrorCode> {
        ErrorCode::i32_as_result(unsafe {
            sljit_emit_enter(self.0, options, arg_types, scratches, saveds, local_size)
        })
        .and(Ok(self))
    }
    #[inline(always)]
    pub fn emit_fcmp(
        &mut self,
        type_: sljit_s32,
        src1: sljit_s32,
        src1w: sljit_sw,
        src2: sljit_s32,
        src2w: sljit_sw,
    ) -> Jump {
        unsafe { sljit_emit_fcmp(self.0, type_, src1, src1w, src2, src2w) }.into()
    }
    #[inline(always)]
    pub fn emit_fcopy(
        &mut self,
        op: sljit_s32,
        freg: sljit_s32,
        reg: sljit_s32,
    ) -> Result<&mut Self, ErrorCode> {
        ErrorCode::i32_as_result(unsafe { sljit_emit_fcopy(self.0, op, freg, reg) }).and(Ok(self))
    }
    #[inline(always)]
    pub fn emit_fmem(
        &mut self,
        type_: sljit_s32,
        freg: sljit_s32,
        mem: sljit_s32,
        memw: sljit_sw,
    ) -> Result<&mut Self, ErrorCode> {
        ErrorCode::i32_as_result(unsafe { sljit_emit_fmem(self.0, type_, freg, mem, memw) })
            .and(Ok(self))
    }
    #[inline(always)]
    pub fn emit_fmem_update(
        &mut self,
        type_: sljit_s32,
        freg: sljit_s32,
        mem: sljit_s32,
        memw: sljit_sw,
    ) -> Result<&mut Self, ErrorCode> {
        ErrorCode::i32_as_result(unsafe { sljit_emit_fmem_update(self.0, type_, freg, mem, memw) })
            .and(Ok(self))
    }
    #[inline(always)]
    pub fn emit_fop1(
        &mut self,
        op: sljit_s32,
        dst: sljit_s32,
        dstw: sljit_sw,
        src: sljit_s32,
        srcw: sljit_sw,
    ) -> Result<&mut Self, ErrorCode> {
        ErrorCode::i32_as_result(unsafe { sljit_emit_fop1(self.0, op, dst, dstw, src, srcw) })
            .and(Ok(self))
    }
    #[inline(always)]
    pub fn emit_fop2(
        &mut self,
        op: sljit_s32,
        dst: sljit_s32,
        dstw: sljit_sw,
        src1: sljit_s32,
        src1w: sljit_sw,
        src2: sljit_s32,
        src2w: sljit_sw,
    ) -> Result<&mut Self, ErrorCode> {
        ErrorCode::i32_as_result(unsafe {
            sljit_emit_fop2(self.0, op, dst, dstw, src1, src1w, src2, src2w)
        })
        .and(Ok(self))
    }
    #[inline(always)]
    pub fn emit_fop2r(
        &mut self,
        op: sljit_s32,
        dst_freg: sljit_s32,
        src1: sljit_s32,
        src1w: sljit_sw,
        src2: sljit_s32,
        src2w: sljit_sw,
    ) -> Result<&mut Self, ErrorCode> {
        ErrorCode::i32_as_result(unsafe {
            sljit_emit_fop2r(self.0, op, dst_freg, src1, src1w, src2, src2w)
        })
        .and(Ok(self))
    }
    #[inline(always)]
    pub fn emit_fselect(
        &mut self,
        type_: sljit_s32,
        dst_freg: sljit_s32,
        src1: sljit_s32,
        src1w: sljit_sw,
        src2_freg: sljit_s32,
    ) -> Result<&mut Self, ErrorCode> {
        ErrorCode::i32_as_result(unsafe {
            sljit_emit_fselect(self.0, type_, dst_freg, src1, src1w, src2_freg)
        })
        .and(Ok(self))
    }
    #[inline(always)]
    pub fn emit_fset32(
        &mut self,
        freg: sljit_s32,
        value: sljit_f32,
    ) -> Result<&mut Self, ErrorCode> {
        ErrorCode::i32_as_result(unsafe { sljit_emit_fset32(self.0, freg, value) }).and(Ok(self))
    }
    #[inline(always)]
    pub fn emit_fset64(
        &mut self,
        freg: sljit_s32,
        value: sljit_f64,
    ) -> Result<&mut Self, ErrorCode> {
        ErrorCode::i32_as_result(unsafe { sljit_emit_fset64(self.0, freg, value) }).and(Ok(self))
    }
    #[inline(always)]
    pub fn emit_icall(
        &mut self,
        type_: sljit_s32,
        arg_types: sljit_s32,
        src: sljit_s32,
        srcw: sljit_sw,
    ) -> Result<&mut Self, ErrorCode> {
        ErrorCode::i32_as_result(unsafe { sljit_emit_icall(self.0, type_, arg_types, src, srcw) })
            .and(Ok(self))
    }
    #[inline(always)]
    pub fn emit_ijump(
        &mut self,
        type_: sljit_s32,
        src: sljit_s32,
        srcw: sljit_sw,
    ) -> Result<&mut Self, ErrorCode> {
        ErrorCode::i32_as_result(unsafe { sljit_emit_ijump(self.0, type_, src, srcw) })
            .and(Ok(self))
    }
    #[inline(always)]
    pub fn emit_jump(&mut self, type_: sljit_s32) -> Jump {
        unsafe { sljit_emit_jump(self.0, type_) }.into()
    }
    #[inline(always)]
    pub fn emit_label(&mut self) -> Label {
        unsafe { sljit_emit_label(self.0) }.into()
    }
    #[inline(always)]
    pub fn emit_mem(
        &mut self,
        type_: sljit_s32,
        reg: sljit_s32,
        mem: sljit_s32,
        memw: sljit_sw,
    ) -> Result<&mut Self, ErrorCode> {
        ErrorCode::i32_as_result(unsafe { sljit_emit_mem(self.0, type_, reg, mem, memw) })
            .and(Ok(self))
    }
    #[inline(always)]
    pub fn emit_mem_update(
        &mut self,
        type_: sljit_s32,
        reg: sljit_s32,
        mem: sljit_s32,
        memw: sljit_sw,
    ) -> Result<&mut Self, ErrorCode> {
        ErrorCode::i32_as_result(unsafe { sljit_emit_mem_update(self.0, type_, reg, mem, memw) })
            .and(Ok(self))
    }
    #[inline(always)]
    pub fn emit_op0(&mut self, op: sljit_s32) -> Result<&mut Self, ErrorCode> {
        ErrorCode::i32_as_result(unsafe { sljit_emit_op0(self.0, op) }).and(Ok(self))
    }
    #[inline(always)]
    pub fn emit_op1(
        &mut self,
        op: sljit_s32,
        dst: sljit_s32,
        dstw: sljit_sw,
        src: sljit_s32,
        srcw: sljit_sw,
    ) -> Result<&mut Self, ErrorCode> {
        ErrorCode::i32_as_result(unsafe { sljit_emit_op1(self.0, op, dst, dstw, src, srcw) })
            .and(Ok(self))
    }
    #[inline(always)]
    pub fn emit_op2(
        &mut self,
        op: sljit_s32,
        dst: sljit_s32,
        dstw: sljit_sw,
        src1: sljit_s32,
        src1w: sljit_sw,
        src2: sljit_s32,
        src2w: sljit_sw,
    ) -> Result<&mut Self, ErrorCode> {
        ErrorCode::i32_as_result(unsafe {
            sljit_emit_op2(self.0, op, dst, dstw, src1, src1w, src2, src2w)
        })
        .and(Ok(self))
    }
    #[inline(always)]
    pub fn emit_op2_shift(
        &mut self,
        op: sljit_s32,
        dst: sljit_s32,
        dstw: sljit_sw,
        src1: sljit_s32,
        src1w: sljit_sw,
        src2: sljit_s32,
        src2w: sljit_sw,
        shift_arg: sljit_sw,
    ) -> Result<&mut Self, ErrorCode> {
        ErrorCode::i32_as_result(unsafe {
            sljit_emit_op2_shift(self.0, op, dst, dstw, src1, src1w, src2, src2w, shift_arg)
        })
        .and(Ok(self))
    }
    #[inline(always)]
    pub fn emit_op2cmpz(
        &mut self,
        op: sljit_s32,
        dst: sljit_s32,
        dstw: sljit_sw,
        src1: sljit_s32,
        src1w: sljit_sw,
        src2: sljit_s32,
        src2w: sljit_sw,
    ) -> Jump {
        unsafe { sljit_emit_op2cmpz(self.0, op, dst, dstw, src1, src1w, src2, src2w) }.into()
    }
    #[inline(always)]
    pub fn emit_op2r(
        &mut self,
        op: sljit_s32,
        dst_reg: sljit_s32,
        src1: sljit_s32,
        src1w: sljit_sw,
        src2: sljit_s32,
        src2w: sljit_sw,
    ) -> Result<&mut Self, ErrorCode> {
        ErrorCode::i32_as_result(unsafe {
            sljit_emit_op2r(self.0, op, dst_reg, src1, src1w, src2, src2w)
        })
        .and(Ok(self))
    }
    #[inline(always)]
    pub fn emit_op2u(
        &mut self,
        op: sljit_s32,
        src1: sljit_s32,
        src1w: sljit_sw,
        src2: sljit_s32,
        src2w: sljit_sw,
    ) -> Result<&mut Self, ErrorCode> {
        ErrorCode::i32_as_result(unsafe { sljit_emit_op2u(self.0, op, src1, src1w, src2, src2w) })
            .and(Ok(self))
    }
    #[inline(always)]
    pub fn emit_op_addr(&mut self, op: sljit_s32, dst: sljit_s32, dstw: sljit_sw) -> Jump {
        unsafe { sljit_emit_op_addr(self.0, op, dst, dstw) }.into()
    }
    #[inline(always)]
    pub fn emit_op_custom(
        &mut self,
        instruction: *mut ::core::ffi::c_void,
        size: sljit_u32,
    ) -> Result<&mut Self, ErrorCode> {
        ErrorCode::i32_as_result(unsafe { sljit_emit_op_custom(self.0, instruction, size) })
            .and(Ok(self))
    }
    #[inline(always)]
    pub fn emit_op_dst(
        &mut self,
        op: sljit_s32,
        dst: sljit_s32,
        dstw: sljit_sw,
    ) -> Result<&mut Self, ErrorCode> {
        ErrorCode::i32_as_result(unsafe { sljit_emit_op_dst(self.0, op, dst, dstw) }).and(Ok(self))
    }
    #[inline(always)]
    pub fn emit_op_flags(
        &mut self,
        op: sljit_s32,
        dst: sljit_s32,
        dstw: sljit_sw,
        type_: sljit_s32,
    ) -> Result<&mut Self, ErrorCode> {
        ErrorCode::i32_as_result(unsafe { sljit_emit_op_flags(self.0, op, dst, dstw, type_) })
            .and(Ok(self))
    }
    #[inline(always)]
    pub fn emit_op_src(
        &mut self,
        op: sljit_s32,
        src: sljit_s32,
        srcw: sljit_sw,
    ) -> Result<&mut Self, ErrorCode> {
        ErrorCode::i32_as_result(unsafe { sljit_emit_op_src(self.0, op, src, srcw) }).and(Ok(self))
    }
    #[inline(always)]
    pub fn emit_return(
        &mut self,
        op: sljit_s32,
        src: sljit_s32,
        srcw: sljit_sw,
    ) -> Result<&mut Self, ErrorCode> {
        ErrorCode::i32_as_result(unsafe { sljit_emit_return(self.0, op, src, srcw) }).and(Ok(self))
    }
    #[inline(always)]
    pub fn emit_return_to(
        &mut self,
        src: sljit_s32,
        srcw: sljit_sw,
    ) -> Result<&mut Self, ErrorCode> {
        ErrorCode::i32_as_result(unsafe { sljit_emit_return_to(self.0, src, srcw) }).and(Ok(self))
    }
    #[inline(always)]
    pub fn emit_return_void(&mut self) -> Result<&mut Self, ErrorCode> {
        ErrorCode::i32_as_result(unsafe { sljit_emit_return_void(self.0) }).and(Ok(self))
    }
    #[inline(always)]
    pub fn emit_select(
        &mut self,
        type_: sljit_s32,
        dst_reg: sljit_s32,
        src1: sljit_s32,
        src1w: sljit_sw,
        src2_reg: sljit_s32,
    ) -> Result<&mut Self, ErrorCode> {
        ErrorCode::i32_as_result(unsafe {
            sljit_emit_select(self.0, type_, dst_reg, src1, src1w, src2_reg)
        })
        .and(Ok(self))
    }
    #[inline(always)]
    pub fn emit_shift_into(
        &mut self,
        op: sljit_s32,
        dst_reg: sljit_s32,
        src1_reg: sljit_s32,
        src2_reg: sljit_s32,
        src3: sljit_s32,
        src3w: sljit_sw,
    ) -> Result<&mut Self, ErrorCode> {
        ErrorCode::i32_as_result(unsafe {
            sljit_emit_shift_into(self.0, op, dst_reg, src1_reg, src2_reg, src3, src3w)
        })
        .and(Ok(self))
    }
    #[inline(always)]
    pub fn emit_simd_extend(
        &mut self,
        type_: sljit_s32,
        vreg: sljit_s32,
        src: sljit_s32,
        srcw: sljit_sw,
    ) -> Result<&mut Self, ErrorCode> {
        ErrorCode::i32_as_result(unsafe { sljit_emit_simd_extend(self.0, type_, vreg, src, srcw) })
            .and(Ok(self))
    }
    #[inline(always)]
    pub fn emit_simd_lane_mov(
        &mut self,
        type_: sljit_s32,
        vreg: sljit_s32,
        lane_index: sljit_s32,
        srcdst: sljit_s32,
        srcdstw: sljit_sw,
    ) -> Result<&mut Self, ErrorCode> {
        ErrorCode::i32_as_result(unsafe {
            sljit_emit_simd_lane_mov(self.0, type_, vreg, lane_index, srcdst, srcdstw)
        })
        .and(Ok(self))
    }
    #[inline(always)]
    pub fn emit_simd_lane_replicate(
        &mut self,
        type_: sljit_s32,
        vreg: sljit_s32,
        src: sljit_s32,
        src_lane_index: sljit_s32,
    ) -> Result<&mut Self, ErrorCode> {
        ErrorCode::i32_as_result(unsafe {
            sljit_emit_simd_lane_replicate(self.0, type_, vreg, src, src_lane_index)
        })
        .and(Ok(self))
    }
    #[inline(always)]
    pub fn emit_simd_mov(
        &mut self,
        type_: sljit_s32,
        vreg: sljit_s32,
        srcdst: sljit_s32,
        srcdstw: sljit_sw,
    ) -> Result<&mut Self, ErrorCode> {
        ErrorCode::i32_as_result(unsafe {
            sljit_emit_simd_mov(self.0, type_, vreg, srcdst, srcdstw)
        })
        .and(Ok(self))
    }
    #[inline(always)]
    pub fn emit_simd_op2(
        &mut self,
        type_: sljit_s32,
        dst_vreg: sljit_s32,
        src1_vreg: sljit_s32,
        src2: sljit_s32,
        src2w: sljit_sw,
    ) -> Result<&mut Self, ErrorCode> {
        ErrorCode::i32_as_result(unsafe {
            sljit_emit_simd_op2(self.0, type_, dst_vreg, src1_vreg, src2, src2w)
        })
        .and(Ok(self))
    }
    #[inline(always)]
    pub fn emit_simd_replicate(
        &mut self,
        type_: sljit_s32,
        vreg: sljit_s32,
        src: sljit_s32,
        srcw: sljit_sw,
    ) -> Result<&mut Self, ErrorCode> {
        ErrorCode::i32_as_result(unsafe {
            sljit_emit_simd_replicate(self.0, type_, vreg, src, srcw)
        })
        .and(Ok(self))
    }
    #[inline(always)]
    pub fn emit_simd_sign(
        &mut self,
        type_: sljit_s32,
        vreg: sljit_s32,
        dst: sljit_s32,
        dstw: sljit_sw,
    ) -> Result<&mut Self, ErrorCode> {
        ErrorCode::i32_as_result(unsafe { sljit_emit_simd_sign(self.0, type_, vreg, dst, dstw) })
            .and(Ok(self))
    }
    #[inline(always)]
    pub fn free_compiler(&mut self) {
        unsafe { sljit_free_compiler(self.0) }
    }
    #[inline(always)]
    pub fn get_compiler_error(&mut self) -> sljit_s32 {
        unsafe { sljit_get_compiler_error(self.0) }
    }
    #[inline(always)]
    pub fn get_executable_offset(&mut self) -> sljit_sw {
        unsafe { sljit_get_executable_offset(self.0) }
    }
    #[inline(always)]
    pub fn get_first_const(&mut self) -> Constant {
        unsafe { sljit_get_first_const(self.0) }.into()
    }
    #[inline(always)]
    pub fn get_first_jump(&mut self) -> Jump {
        unsafe { sljit_get_first_jump(self.0) }.into()
    }
    #[inline(always)]
    pub fn get_first_label(&mut self) -> Label {
        unsafe { sljit_get_first_label(self.0) }.into()
    }
    #[inline(always)]
    pub fn get_generated_code_size(&mut self) -> sljit_uw {
        unsafe { sljit_get_generated_code_size(self.0) }
    }
    #[inline(always)]
    pub fn get_local_base(
        &mut self,
        dst: sljit_s32,
        dstw: sljit_sw,
        offset: sljit_sw,
    ) -> sljit_s32 {
        unsafe { sljit_get_local_base(self.0, dst, dstw, offset) }
    }
    #[inline(always)]
    pub fn serialize_compiler(&mut self, options: sljit_s32, size: *mut sljit_uw) -> *mut sljit_uw {
        unsafe { sljit_serialize_compiler(self.0, options, size) }
    }
    #[inline(always)]
    pub fn set_compiler_memory_error(&mut self) {
        unsafe { sljit_set_compiler_memory_error(self.0) }
    }
    #[inline(always)]
    pub fn set_context(
        &mut self,
        options: sljit_s32,
        arg_types: sljit_s32,
        scratches: sljit_s32,
        saveds: sljit_s32,
        local_size: sljit_s32,
    ) -> sljit_s32 {
        unsafe { sljit_set_context(self.0, options, arg_types, scratches, saveds, local_size) }
    }
    #[inline(always)]
    pub fn set_current_flags(&mut self, current_flags: sljit_s32) {
        unsafe { sljit_set_current_flags(self.0, current_flags) }
    }
}
