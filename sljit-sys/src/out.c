#include "sljit/sljit_src/sljitLir.h"

// Static wrappers

sljit_s32 sljit_get_compiler_error__extern(struct sljit_compiler *compiler) { return sljit_get_compiler_error(compiler); }
void * sljit_get_allocator_data__extern(struct sljit_compiler *compiler) { return sljit_get_allocator_data(compiler); }
void * sljit_get_exec_allocator_data__extern(struct sljit_compiler *compiler) { return sljit_get_exec_allocator_data(compiler); }
sljit_sw sljit_get_executable_offset__extern(struct sljit_compiler *compiler) { return sljit_get_executable_offset(compiler); }
sljit_uw sljit_get_generated_code_size__extern(struct sljit_compiler *compiler) { return sljit_get_generated_code_size(compiler); }
sljit_uw sljit_get_label_addr__extern(struct sljit_label *label) { return sljit_get_label_addr(label); }
sljit_uw sljit_get_jump_addr__extern(struct sljit_jump *jump) { return sljit_get_jump_addr(jump); }
sljit_uw sljit_get_const_addr__extern(struct sljit_const *const_) { return sljit_get_const_addr(const_); }
