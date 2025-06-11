#include "sljit/sljit_src/sljitLir.h"

// Static wrappers

sljit_s32 sljit_get_compiler_error__extern(struct sljit_compiler *compiler) { return sljit_get_compiler_error(compiler); }
void * sljit_compiler_get_allocator_data__extern(struct sljit_compiler *compiler) { return sljit_compiler_get_allocator_data(compiler); }
void sljit_compiler_set_user_data__extern(struct sljit_compiler *compiler, void *user_data) { sljit_compiler_set_user_data(compiler, user_data); }
void * sljit_compiler_get_user_data__extern(struct sljit_compiler *compiler) { return sljit_compiler_get_user_data(compiler); }
sljit_sw sljit_get_executable_offset__extern(struct sljit_compiler *compiler) { return sljit_get_executable_offset(compiler); }
sljit_uw sljit_get_generated_code_size__extern(struct sljit_compiler *compiler) { return sljit_get_generated_code_size(compiler); }
sljit_uw sljit_get_label_addr__extern(struct sljit_label *label) { return sljit_get_label_addr(label); }
sljit_uw sljit_get_label_abs_addr__extern(struct sljit_label *label) { return sljit_get_label_abs_addr(label); }
sljit_uw sljit_get_jump_addr__extern(struct sljit_jump *jump) { return sljit_get_jump_addr(jump); }
sljit_uw sljit_get_const_addr__extern(struct sljit_const *const_) { return sljit_get_const_addr(const_); }
struct sljit_label * sljit_get_first_label__extern(struct sljit_compiler *compiler) { return sljit_get_first_label(compiler); }
struct sljit_jump * sljit_get_first_jump__extern(struct sljit_compiler *compiler) { return sljit_get_first_jump(compiler); }
struct sljit_const * sljit_get_first_const__extern(struct sljit_compiler *compiler) { return sljit_get_first_const(compiler); }
struct sljit_label * sljit_get_next_label__extern(struct sljit_label *label) { return sljit_get_next_label(label); }
struct sljit_jump * sljit_get_next_jump__extern(struct sljit_jump *jump) { return sljit_get_next_jump(jump); }
struct sljit_const * sljit_get_next_const__extern(struct sljit_const *const_) { return sljit_get_next_const(const_); }
struct sljit_label * sljit_jump_get_label__extern(struct sljit_jump *jump) { return sljit_jump_get_label(jump); }
sljit_uw sljit_jump_get_target__extern(struct sljit_jump *jump) { return sljit_jump_get_target(jump); }
