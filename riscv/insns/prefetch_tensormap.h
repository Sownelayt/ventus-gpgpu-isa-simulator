// PREFETCH_TENSORMAP: best-effort cache hint for a 128B tensor map descriptor.
// Spike has no architectural cache state here, so the instruction is a NOP.
{
  npc = sext_xlen(pc + 4);
  reg_t desc_ptr = RS1;
  (void)desc_ptr;
}
