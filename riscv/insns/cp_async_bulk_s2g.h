// CP_ASYNC_BULK_S2G: copy or atomically reduce RS2 bytes from RS1 shared
// memory into rd global memory.  inst[31:29] is the redOp and inst[28:27]
// is the 32-bit element type for reduce operations.
{
  npc = sext_xlen(pc + 4);
  const reg_t src = RS1;
  const reg_t size = RS2;
  const reg_t dst = READ_REG(insn.rd());
  const uint32_t reduce_mode = (insn.bits() >> 29) & 7;
  const uint32_t reduce_type = (insn.bits() >> 27) & 3;
  const bool copy = reduce_mode == VENTUS_TMA_V2_REDUCE_COPY;
  const bool arithmetic = reduce_mode == VENTUS_TMA_V2_REDUCE_ADD ||
                          reduce_mode == VENTUS_TMA_V2_REDUCE_MIN ||
                          reduce_mode == VENTUS_TMA_V2_REDUCE_MAX;
  const bool bitwise = reduce_mode == VENTUS_TMA_V2_REDUCE_AND ||
                       reduce_mode == VENTUS_TMA_V2_REDUCE_OR ||
                       reduce_mode == VENTUS_TMA_V2_REDUCE_XOR;
  const bool arithmetic_type =
      reduce_type == VENTUS_TMA_V2_BULK_REDUCE_TYPE_U32 ||
      reduce_type == VENTUS_TMA_V2_BULK_REDUCE_TYPE_S32;
  const bool encoding_ok =
      (copy && reduce_type == VENTUS_TMA_V2_BULK_REDUCE_TYPE_U32) ||
      (arithmetic && arithmetic_type) ||
      (bitwise && reduce_type == VENTUS_TMA_V2_BULK_REDUCE_TYPE_B32);
  auto set_status = [&](reg_t code, reg_t detail) {
    if (p->get_csr(CSR_DMA_STATUS, insn, false, true) == 0)
      p->put_csr(CSR_DMA_STATUS, (detail << 8) | code);
  };

  if ((src & 15) || (dst & 15) || size == 0 || (size & 15) ||
      !encoding_ok) {
    set_status(VENTUS_TMA_STATUS_UNSUPPORTED_FEATURE,
               VENTUS_TMA_V2_FUNCT_BULK_S2G);
  } else if (copy) {
    for (reg_t i = 0; i < size; ++i)
      MMU.store_uint8(dst + i, MMU.load_uint8(src + i));
  } else {
    for (reg_t i = 0; i < size; i += sizeof(uint32_t)) {
      const uint32_t operand = MMU.load_uint32(src + i);
      const uint32_t old = MMU.load_uint32(dst + i);
      uint32_t result = old;
      switch (reduce_mode) {
        case VENTUS_TMA_V2_REDUCE_ADD:
          result = old + operand;
          break;
        case VENTUS_TMA_V2_REDUCE_MIN:
          result = reduce_type == VENTUS_TMA_V2_BULK_REDUCE_TYPE_S32
                       ? (static_cast<int32_t>(old) <
                                  static_cast<int32_t>(operand)
                              ? old
                              : operand)
                       : (old < operand ? old : operand);
          break;
        case VENTUS_TMA_V2_REDUCE_MAX:
          result = reduce_type == VENTUS_TMA_V2_BULK_REDUCE_TYPE_S32
                       ? (static_cast<int32_t>(old) >
                                  static_cast<int32_t>(operand)
                              ? old
                              : operand)
                       : (old > operand ? old : operand);
          break;
        case VENTUS_TMA_V2_REDUCE_AND:
          result = old & operand;
          break;
        case VENTUS_TMA_V2_REDUCE_OR:
          result = old | operand;
          break;
        case VENTUS_TMA_V2_REDUCE_XOR:
          result = old ^ operand;
          break;
      }
      MMU.store_uint32(dst + i, result);
    }
  }
}
