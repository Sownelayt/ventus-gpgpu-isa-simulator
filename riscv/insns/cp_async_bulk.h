// CP_ASYNC_BULK: copy RS2 bytes from RS1 global memory to rd shared memory.
{
  npc = sext_xlen(pc + 4);
  const reg_t src = RS1;
  const reg_t size = RS2;
  const reg_t dst = READ_REG(insn.rd());
  auto& tma_binding =
      P.gpgpu_unit.tma_binding(P.get_csr(CSR_WID));
  auto set_status = [&](reg_t code, reg_t detail) {
    if (p->get_csr(CSR_DMA_STATUS, insn, false, true) == 0)
      p->put_csr(CSR_DMA_STATUS, (detail << 8) | code);
  };
  bool binding_ok = true;
  if (tma_binding.armed) {
    const reg_t state = MMU.load_uint32(tma_binding.address + 4);
    binding_ok = size <= tma_binding.bytes &&
                 ((state >> 24) & 0xff) ==
                     tma_binding.generation;
    if (!binding_ok)
      set_status(VENTUS_TMA_STATUS_MBARRIER_PROTOCOL, 4);
  }
  if ((src & 15) || (dst & 15) || size == 0 || (size & 15) || !binding_ok) {
    set_status(VENTUS_TMA_STATUS_UNSUPPORTED_FEATURE,
               VENTUS_TMA_V2_FUNCT_BULK_G2S);
  } else {
    for (reg_t i = 0; i < size; ++i)
      MMU.store_uint8(dst + i, MMU.load_uint8(src + i));
  }

  if (binding_ok && tma_binding.armed) {
    const reg_t address = tma_binding.address;
    const reg_t bytes = size;
    const reg_t pending = MMU.load_uint32(address);
    reg_t state = MMU.load_uint32(address + 4);
    if (bytes == 0 || pending < bytes) {
      set_status(VENTUS_TMA_STATUS_MBARRIER_PROTOCOL, 3);
    } else {
      const reg_t next = pending - bytes;
      tma_binding.bytes -= bytes;
      if (tma_binding.bytes == 0)
        tma_binding.armed = false;
      MMU.store_uint32(address, next);
      if (next == 0 && ((state >> 16) & 0xff) == 0) {
        const reg_t expected = (state >> 8) & 0xff;
        const reg_t generation = (((state >> 24) + 1) & 0xff) << 24;
        state = ((state ^ 1) & ~reg_t(0xffff0002)) | generation;
        state |= expected << 16;
        MMU.store_uint32(address + 4, state);
      }
    }
  }
}
