// CP_ASYNC_BULK_S2G: copy RS2 bytes from RS1 (shared mem) to reg[rd] (global mem)
{
  npc = sext_xlen(pc + 4);
  reg_t src  = RS1;
  reg_t size = RS2;
  reg_t dst  = READ_REG(insn.rd());
  for (reg_t i = 0; i < size; i++)
    MMU.store_uint8(dst + i, MMU.load_uint8(src + i));
}
