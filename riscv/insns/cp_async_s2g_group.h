// S2G commit_group / wait_group. Spike executes DMA synchronously, so valid
// operations have no remaining work; invalid zimm values still update the
// architected sticky DMA status CSR.
{
  npc = sext_xlen(pc + 4);
  const reg_t zimm = insn.rs1();
  const bool valid = zimm == VENTUS_TMA_V2_S2G_GROUP_COMMIT ||
      (zimm >= VENTUS_TMA_V2_S2G_GROUP_WAIT_BASE &&
       zimm <= VENTUS_TMA_V2_S2G_GROUP_WAIT_BASE +
                   VENTUS_TMA_V2_S2G_GROUP_WAIT_KEEP_MAX);
  if (!valid && p->get_csr(CSR_DMA_STATUS, insn, false, true) == 0)
    p->put_csr(CSR_DMA_STATUS,
               (zimm << 8) | VENTUS_TMA_STATUS_INVALID_GROUP_OPERATION);
}
