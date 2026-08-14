// CP_ASYNC_TENSOR: descriptor-addressed tensor copy, global to shared.
{
  npc = sext_xlen(pc + 4);
  const bool tma_s2g = false;
  const uint32_t tma_reduce_mode = VENTUS_TMA_V2_REDUCE_COPY;
  const reg_t tma_shared_base = READ_REG(insn.rd());
  const reg_t tma_desc_ptr = RS1;
#include "ventus_tma_v2_tensor.inc"
}
