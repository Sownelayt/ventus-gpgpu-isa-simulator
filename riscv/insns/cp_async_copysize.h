// CP_ASYNC_COPYSIZE: copy (4 << RS2) bytes from RS1 to reg[rd]
{
  reg_t src  = RS1;
  reg_t size = 4 << RS2;
  reg_t dst  = READ_REG(insn.rd());
  for (reg_t i = 0; i < size; i++)
    MMU.store_uint8(dst + i, MMU.load_uint8(src + i));
}
