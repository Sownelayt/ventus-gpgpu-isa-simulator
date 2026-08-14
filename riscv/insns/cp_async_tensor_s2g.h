// CP_ASYNC_TENSOR_S2G: descriptor-addressed tensor copy, shared to global.
{
  npc = sext_xlen(pc + 4);
  const bool tma_s2g = true;
  const uint32_t tma_reduce_mode = (insn.bits() >> 29) & 7;
  const reg_t tma_shared_base = READ_REG(insn.rd());
  const reg_t tma_desc_ptr = RS1;
#include "ventus_tma_v2_tensor.inc"
}
