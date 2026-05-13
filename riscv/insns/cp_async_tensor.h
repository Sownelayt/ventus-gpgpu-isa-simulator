// CP_ASYNC_TENSOR: copy a multi-dim tensor "box" from global memory to shared
// memory, using a descriptor read directly from VGPR operands.
//
// Descriptor layout (uint32 words), read from VGPRs v{rs1}, v{rs2}, v{rd}:
//   VRS1 (param_buf[0..31])  — tensor descriptor
//     [0]       dataType      (see table below)
//     [1]       tensorRank    (1..5)
//     [2]       globalAddress (unused here; BoxAddress below is used for the
//                              actual copy source)
//     [3..7]    globalDim[0..4]
//     [8..12]   globalStrides[0..4]   (bytes between successive "rows/slices"
//                                      along each dim; see offset formula)
//   VRS2 (param_buf[32..63]) — box descriptor
//     [0]       BoxAddress            (source base address in global mem)
//     [1..5]    boxDim[0..4]          (extents of the sub-box to copy)
//     [6..10]   elementStrides[0..4]  (step in elements, not bytes; 1 = dense)
//     [11]      interleaveMode        (unsupported here except 0)
//     [12]      swizzleMode           (0/1/2/3 = none/32B/64B/128B)
//   VRS3 (param_buf[64..95]) — destination
//     [0]       dst shared-memory address (tightly packed, row-major in
//                                          boxDim shape, then optional swizzle)
//
// dataType encoding (kept in sync with host/kernel tests):
//   0  UINT8     1 byte
//   1  UINT16    2 bytes
//   2  UINT32    4 bytes
//   3  INT8      1 byte
//   4  INT16     2 bytes
//   5  INT32     4 bytes
//   6  FP32      4 bytes
//   7  FP16      2 bytes
//   8  BF16      2 bytes
//   9  UINT64    8 bytes
//   10 INT64     8 bytes
//   11 FP64      8 bytes
//
// DMA in spike is synchronous, so the copy is done eagerly and the companion
// CP_ASYNC_FENCE is a NOP. Because all lanes issue the same opcode with the
// same descriptor values in VGPRs, the copy is idempotent: executing it N
// times yields the same result as once.
//
// Offset formulae:
//   Let dim_byte_stride(d) = elemSize           if d == 0
//                          = globalStrides[d-1] if d >= 1
//   Source offset for multi-index idx[]:
//     src_off = Σ_d idx[d] * elementStrides[d] * dim_byte_stride(d)
//   Destination offset (box is tightly packed, innermost dim first):
//     dst_off = Σ_d idx[d] * prod_{k<d} boxDim[k] * elemSize
{
  npc = sext_xlen(pc + 4);
  auto u32 = [&](reg_t word_idx) -> reg_t {
    if (word_idx < 32) {
      return (reg_t)P.VU.elt<uint32_t>(1, insn.rs1(), word_idx);
    }
    if (word_idx < 64) {
      return (reg_t)P.VU.elt<uint32_t>(2, insn.rs2(), word_idx - 32);
    }
    return (reg_t)P.VU.elt<uint32_t>(3, insn.rd(), word_idx - 64);
  };

  // VRS1
  reg_t dataType = u32(0);
  reg_t rank     = u32(1);
  reg_t gStride[5];
  for (int i = 0; i < 5; i++) gStride[i] = u32(8 + i);

  // VRS2
  reg_t boxAddr = u32(32);
  reg_t boxDim[5] = {1, 1, 1, 1, 1};
  for (reg_t i = 0; i < rank && i < 5; i++) boxDim[i] = u32(33 + i);
  reg_t eStride[5] = {1, 1, 1, 1, 1};
  for (reg_t i = 0; i < rank && i < 5; i++) eStride[i] = u32(38 + i);
  reg_t interleaveMode = u32(43);
  reg_t swizzleMode = u32(44);

  // VRS3
  reg_t dstAddr = u32(64);

  // dataType → element byte size. Unknown codes fall back to 4 (FP32-ish)
  // with a stderr warning so testing surfaces the issue instead of silently
  // corrupting memory.
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
              "cp.async.tensor: unknown dataType=%u, defaulting elemSize=4\n",
              (unsigned)dataType);
      elemSize = 4;
      break;
  }

  if (rank == 0 || rank > 5) {
    fprintf(stderr,
            "cp.async.tensor: unsupported rank=%u (must be 1..5); skipping\n",
            (unsigned)rank);
    return npc;
  }
  if (interleaveMode != 0) {
    fprintf(stderr,
            "cp.async.tensor: unsupported interleaveMode=%u; skipping\n",
            (unsigned)interleaveMode);
    return npc;
  }
  if (swizzleMode > 3) {
    fprintf(stderr,
            "cp.async.tensor: unsupported swizzleMode=%u; skipping\n",
            (unsigned)swizzleMode);
    return npc;
  }

  // Per-dim byte stride in source. Zero dim_byte_stride(0) never happens (it's
  // elemSize). dim_byte_stride(d>=1) is read from globalStrides[d-1].
  auto dim_byte_stride = [&](reg_t d) -> reg_t {
    return (d == 0) ? elemSize : gStride[d - 1];
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

  // Enumerate linear index `lin` in [0, ∏ boxDim[d]) and decompose it into
  // per-dim indices (inner-most dim = 0). Compose source and destination
  // offsets on the fly. This is O(total * rank) which is fine for all
  // reasonable test sizes (up to a few KB).
  reg_t total = 1;
  for (reg_t d = 0; d < rank; d++) total *= boxDim[d];

  for (reg_t lin = 0; lin < total; lin++) {
    reg_t idx[5] = {0, 0, 0, 0, 0};
    reg_t rem = lin;
    for (reg_t d = 0; d < rank; d++) {
      idx[d] = rem % boxDim[d];
      rem   /= boxDim[d];
    }

    reg_t src_off = 0;
    for (reg_t d = 0; d < rank; d++) {
      src_off += idx[d] * eStride[d] * dim_byte_stride(d);
    }

    reg_t dst_off = 0;
    reg_t dst_mul = elemSize;
    for (reg_t d = 0; d < rank; d++) {
      dst_off += idx[d] * dst_mul;
      dst_mul *= boxDim[d];
    }
    reg_t row = 0;
    reg_t row_mul = 1;
    for (reg_t d = 1; d < rank; d++) {
      row += idx[d] * row_mul;
      row_mul *= boxDim[d];
    }
    dst_off = swizzle_dst_offset(dst_off, row);

    // Copy using the widest granule that divides elemSize, to keep logs
    // readable and to avoid spurious sub-word load/store churn. Falls back
    // to bytes for sizes MMU does not have a primitive for.
    reg_t src = boxAddr + src_off;
    reg_t dst = dstAddr + dst_off;
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
        for (reg_t b = 0; b < elemSize; b++) {
          MMU.store_uint8(dst + b, MMU.load_uint8(src + b));
        }
        break;
    }
  }
}
