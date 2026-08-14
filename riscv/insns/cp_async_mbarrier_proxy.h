// Shared-memory mbarrier and async-proxy operations. bits[26:25] select:
// init, arrive.expect_tx, wait, or fence.proxy.async.shared.
{
  npc = sext_xlen(pc + 4);
  const reg_t op = (insn.bits() >> 25) & 3;
  const reg_t address = RS1;
  const reg_t value = RS2;
  const reg_t wid = P.get_csr(CSR_WID);
  const reg_t owner = P.get_csr(CSR_WGID);
  auto& binding = P.gpgpu_unit.tma_binding(wid);
  auto fail = [&](reg_t detail) {
    if (p->get_csr(CSR_DMA_STATUS, insn, false, true) == 0)
      p->put_csr(CSR_DMA_STATUS,
                 (detail << 8) | VENTUS_TMA_STATUS_MBARRIER_PROTOCOL);
  };

  if (op == VENTUS_TMA_V2_MBARRIER_INIT) {
    const bool exists = P.gpgpu_unit.tma_barrier_exists(owner, address);
    const reg_t pending = exists ? MMU.load_uint32(address) : 0;
    const bool allocated = (address & 7) == 0 && value != 0 && value <= 255 &&
        !binding.armed && pending == 0 &&
        P.gpgpu_unit.tma_barrier_allocate(owner, address);
    if (!allocated) {
      fail(0);
    } else {
      MMU.store_uint32(address, 0);
      // The shared representation is architecturally opaque.  bit1 records
      // that phase zero may accept arrivals; bits31:24 hold generation.
      MMU.store_uint32(address + 4, (value << 8) | (value << 16) | 2);
    }
  } else if (op == VENTUS_TMA_V2_MBARRIER_ARRIVE_EXPECT_TX) {
    const reg_t pending = MMU.load_uint32(address);
    const reg_t state = MMU.load_uint32(address + 4);
    const reg_t pending_arrivals = (state >> 16) & 0xff;
    if ((address & 7) ||
        !P.gpgpu_unit.tma_barrier_exists(owner, address) ||
        value == 0 || pending_arrivals == 0 ||
        pending + value > 0xffffffffULL ||
        binding.armed || !(state & 2)) {
      fail(1);
    } else {
      MMU.store_uint32(address, pending + value);
      MMU.store_uint32(address + 4,
                       (state & ~reg_t(0x00ff0000)) |
                           ((pending_arrivals - 1) << 16));
      binding.armed = true;
      binding.address = address;
      binding.bytes = value;
      binding.generation = (state >> 24) & 0xff;
    }
  } else if (op == VENTUS_TMA_V2_MBARRIER_WAIT) {
    const reg_t pending = MMU.load_uint32(address);
    const reg_t state = MMU.load_uint32(address + 4);
    const reg_t requested_phase = value & 1;
    if ((address & 7) ||
        !P.gpgpu_unit.tma_barrier_exists(owner, address)) {
      fail(2);
    } else if (pending != 0 || (state & 1) == requested_phase) {
      // A real mbarrier wait deschedules this warp. Re-executing the same PC
      // lets the synchronous reference model run another warp meanwhile.
      set_pc(pc);
    } else {
      MMU.store_uint32(address + 4, state | 2);
    }
  } else {
    // Spike has one synchronous memory proxy; the fence is architecturally
    // visible as an ordering point but needs no additional state transition.
  }
}
