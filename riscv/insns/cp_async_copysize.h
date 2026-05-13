// CP_ASYNC_COPYSIZE: copy (4 << inst[26:25]) bytes from RS1 to reg[rd]
{
  npc = sext_xlen(pc + 4);
  reg_t src  = RS1;
  reg_t copysize = (insn.bits() >> 25) & 0x3;
  reg_t size = ((reg_t)4) << copysize;
  reg_t dst  = READ_REG(insn.rd());
  for (reg_t i = 0; i < size; i++)
    MMU.store_uint8(dst + i, MMU.load_uint8(src + i));
}
