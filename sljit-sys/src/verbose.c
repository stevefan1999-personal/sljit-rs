#include "sljit/sljit_src/sljitLir.h"

#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
SLJIT_API_FUNC_ATTRIBUTE void sljit_compiler_verbose_helper(struct sljit_compiler *compiler) {
    sljit_compiler_verbose(compiler, stdout);
}
#endif