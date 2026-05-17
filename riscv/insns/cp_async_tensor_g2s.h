// CP_ASYNC_TENSOR_G2S: descriptor-addressed tensor copy from global to shared.
//
// Operands:
//   rd  = shared-memory destination pointer
//   rs1 = global-memory tensor map descriptor pointer
//   rs2 = VGPR dynamic parameter block
//
// Descriptor v0 is 128B / 32 u32 words:
//   word 0      magic/version/flags, currently informational
//   word 1      bits [3:0] dataType, [7:4] rank, [9:8] interleave,
//               [11:10] swizzle, [13:12] L2promotion, [14] oobfill
//   word 2      globalAddress
//   word 3      descriptor size/reserved
//   word 4..8   globalDim[0..4]
//   word 9..13  byteStride[0..4], including dim0 byte stride
//   word 14..18 boxDim[0..4]
//   word 19..23 elementStrides[0..4]
//   word 24..31 reserved
//
// VGPR dynamic block v0:
//   word 0..4   tensorCoords[0..4]
{
  npc = sext_xlen(pc + 4);
  reg_t dstAddr = READ_REG(insn.rd());
  reg_t descPtr = RS1;

  reg_t desc[32];
  for (int i = 0; i < 32; i++)
    desc[i] = (reg_t)MMU.load_uint32(descPtr + 4 * i);

  reg_t coords[5] = {0, 0, 0, 0, 0};
  for (int i = 0; i < 5; i++)
    coords[i] = (reg_t)P.VU.elt<uint32_t>(2, insn.rs2(), i);

  static const bool debug = std::getenv("VENTUS_TMA_G2S_DEBUG") != nullptr;

  reg_t control = desc[1];
  reg_t dataType = control & 0xf;
  reg_t rank = (control >> 4) & 0xf;
  reg_t interleaveMode = (control >> 8) & 0x3;
  reg_t swizzleMode = (control >> 10) & 0x3;
  reg_t oobfill = (control >> 14) & 0x1;

  reg_t globalAddress = desc[2];
  reg_t globalDim[5];
  reg_t byteStride[5];
  reg_t boxDim[5];
  reg_t eStride[5];
  for (int i = 0; i < 5; i++) {
    globalDim[i] = desc[4 + i];
    byteStride[i] = desc[9 + i];
    boxDim[i] = desc[14 + i];
    eStride[i] = desc[19 + i] ? desc[19 + i] : 1;
  }

  reg_t elemSize;
  switch (dataType) {
    case 0:  elemSize = 1; break; // UINT8
    case 1:  elemSize = 2; break; // UINT16
    case 2:  elemSize = 4; break; // UINT32
    case 3:  elemSize = 1; break; // INT8
    case 4:  elemSize = 2; break; // INT16
    case 5:  elemSize = 4; break; // INT32
    case 6:  elemSize = 4; break; // FP32
    case 7:  elemSize = 2; break; // FP16
    case 8:  elemSize = 2; break; // BF16
    case 9:  elemSize = 8; break; // UINT64
    case 10: elemSize = 8; break; // INT64
    case 11: elemSize = 8; break; // FP64
    default:
      fprintf(stderr,
              "cp.async.tensor.g2s: unknown dataType=%u, defaulting elemSize=4\n",
              (unsigned)dataType);
      elemSize = 4;
      break;
  }

  if (rank == 0 || rank > 5) {
    fprintf(stderr,
            "cp.async.tensor.g2s: unsupported rank=%u (must be 1..5); skipping\n",
            (unsigned)rank);
    return npc;
  }
  if (interleaveMode != 0) {
    fprintf(stderr,
            "cp.async.tensor.g2s: unsupported interleaveMode=%u; skipping\n",
            (unsigned)interleaveMode);
    return npc;
  }

  // Descriptor v0 carries dim0 byte stride explicitly. Keep old dense tests
  // robust by accepting zero there as "elemSize".
  if (byteStride[0] == 0)
    byteStride[0] = elemSize;

  reg_t boxAddr = globalAddress;
  for (reg_t d = 0; d < rank; d++)
    boxAddr += coords[d] * byteStride[d];

  if (debug) {
    fprintf(stderr,
            "cp.async.tensor.g2s: dst=0x%08llx"
            " desc=0x%08llx dyn_vreg=v%u"
            " ctrl=0x%08llx rank=%u dtype=%u global=0x%08llx"
            " coords=[%u,%u,%u,%u,%u] boxAddr=0x%08llx"
            " gdim=[%u,%u,%u,%u,%u] stride=[%u,%u,%u,%u,%u]"
            " box=[%u,%u,%u,%u,%u]\n",
            (unsigned long long)dstAddr, (unsigned long long)descPtr,
            (unsigned)insn.rs2(), (unsigned long long)control,
            (unsigned)rank, (unsigned)dataType,
            (unsigned long long)globalAddress,
            (unsigned)coords[0], (unsigned)coords[1], (unsigned)coords[2],
            (unsigned)coords[3], (unsigned)coords[4],
            (unsigned long long)boxAddr,
            (unsigned)globalDim[0], (unsigned)globalDim[1],
            (unsigned)globalDim[2], (unsigned)globalDim[3],
            (unsigned)globalDim[4],
            (unsigned)byteStride[0], (unsigned)byteStride[1],
            (unsigned)byteStride[2], (unsigned)byteStride[3],
            (unsigned)byteStride[4],
            (unsigned)boxDim[0], (unsigned)boxDim[1],
            (unsigned)boxDim[2], (unsigned)boxDim[3],
            (unsigned)boxDim[4]);
  }

  reg_t outDim[5] = {1, 1, 1, 1, 1};
  for (reg_t d = 0; d < rank; d++) {
    if (d == 0 || eStride[d] <= 1)
      outDim[d] = boxDim[d];
    else
      outDim[d] = (boxDim[d] + eStride[d] - 1) / eStride[d];
  }

  auto is_integer_dtype = [&](reg_t dt) -> bool {
    return dt <= 5 || dt == 9 || dt == 10;
  };

  auto fill_oob = [&](reg_t dst) {
    uint8_t fill = (!is_integer_dtype(dataType) && oobfill) ? 0xff : 0x00;
    for (reg_t b = 0; b < elemSize; b++)
      MMU.store_uint8(dst + b, fill);
  };

  auto swizzle_dst_offset = [&](reg_t logical_off, reg_t row) -> reg_t {
    if (swizzleMode == 0) return logical_off;
    reg_t chunk_bits = swizzleMode;         // 1/2/3 for 32B/64B/128B
    reg_t span = 16 << chunk_bits;
    reg_t chunk_mask = (1 << chunk_bits) - 1;
    reg_t low = logical_off & 0xf;
    reg_t chunk = (logical_off >> 4) & chunk_mask;
    reg_t row_low = row & chunk_mask;
    return (logical_off & ~(span - 1)) | ((chunk ^ row_low) << 4) | low;
  };

  reg_t total = 1;
  for (reg_t d = 0; d < rank; d++)
    total *= outDim[d];

  for (reg_t lin = 0; lin < total; lin++) {
    reg_t idx[5] = {0, 0, 0, 0, 0};
    reg_t rem = lin;
    for (reg_t d = 0; d < rank; d++) {
      idx[d] = rem % outDim[d];
      rem /= outDim[d];
    }

    reg_t src_off = 0;
    bool oob = false;
    for (reg_t d = 0; d < rank; d++) {
      reg_t coord = coords[d] + idx[d] * eStride[d];
      if (coord >= globalDim[d])
        oob = true;
      src_off += idx[d] * eStride[d] * byteStride[d];
    }

    reg_t dst_off = 0;
    reg_t dst_mul = elemSize;
    for (reg_t d = 0; d < rank; d++) {
      dst_off += idx[d] * dst_mul;
      dst_mul *= outDim[d];
    }
    reg_t row = 0;
    reg_t row_mul = 1;
    for (reg_t d = 1; d < rank; d++) {
      row += idx[d] * row_mul;
      row_mul *= outDim[d];
    }
    dst_off = swizzle_dst_offset(dst_off, row);

    reg_t src = boxAddr + src_off;
    reg_t dst = dstAddr + dst_off;
    if (oob) {
      fill_oob(dst);
    } else {
      switch (elemSize) {
        case 8:
          MMU.store_uint64(dst, MMU.load_uint64(src));
          break;
        case 4:
          MMU.store_uint32(dst, MMU.load_uint32(src));
          break;
        case 2:
          MMU.store_uint16(dst, MMU.load_uint16(src));
          break;
        case 1:
          MMU.store_uint8(dst, MMU.load_uint8(src));
          break;
        default:
          for (reg_t b = 0; b < elemSize; b++)
            MMU.store_uint8(dst + b, MMU.load_uint8(src + b));
          break;
      }
    }
  }
}
