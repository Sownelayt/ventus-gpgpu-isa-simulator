// INVALIDATE_TENSORMAP: publish that a global TensorMap may have changed.
// Spike reads the descriptor directly for every tensor instruction and has
// no descriptor cache, so invalidation is architecturally complete as a NOP.
{
  npc = sext_xlen(pc + 4);
  reg_t desc_ptr = RS1;
  (void)desc_ptr;
}
