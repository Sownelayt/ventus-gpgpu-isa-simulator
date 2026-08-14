// Shared implementation for CP_ASYNC_TENSOR G2S/S2G.
// The including handler defines tma_s2g, tma_shared_base, and tma_desc_ptr.
auto tma_set_status = [&](reg_t code, reg_t detail) {
  if (p->get_csr(CSR_DMA_STATUS, insn, false, true) == 0)
    p->put_csr(CSR_DMA_STATUS, (detail << 8) | code);
};
auto& tma_binding =
    P.gpgpu_unit.tma_binding(P.get_csr(CSR_WID));

auto tma_complete_g2s = [&](reg_t actual_bytes) {
  if (tma_s2g || !tma_binding.armed) return;
  const reg_t address = tma_binding.address;
  const reg_t bytes = actual_bytes;
  const reg_t pending = MMU.load_uint32(address);
  reg_t state = MMU.load_uint32(address + 4);
  if (bytes == 0 || pending < bytes ||
      tma_binding.bytes < bytes ||
      ((state >> 24) & 0xff) != tma_binding.generation) {
    tma_set_status(VENTUS_TMA_STATUS_MBARRIER_PROTOCOL, 3);
    return;
  }
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
};

bool tma_legal = true;
reg_t tma_error_code = VENTUS_TMA_STATUS_INVALID_DESCRIPTOR;
reg_t tma_error_detail = 0;
auto tma_fail = [&](reg_t code, reg_t detail) {
  if (tma_legal) {
    tma_legal = false;
    tma_error_code = code;
    tma_error_detail = detail;
  }
};

uint32_t tma_desc[32] = {};
if ((tma_desc_ptr & (VENTUS_TMA_V2_DESCRIPTOR_ALIGNMENT - 1)) != 0) {
  tma_fail(VENTUS_TMA_STATUS_INVALID_DESCRIPTOR, 7);
} else {
  for (unsigned i = 0; i < 32; ++i)
    tma_desc[i] = MMU.load_uint32(tma_desc_ptr + 4 * i);
}

if (tma_legal && tma_desc[VENTUS_TMA_V2_WORD_MAGIC] != VENTUS_TMA_V2_MAGIC)
  tma_fail(VENTUS_TMA_STATUS_INVALID_DESCRIPTOR, 1);
if (tma_legal) {
  for (unsigned word = 27; word < 32; ++word)
    if (tma_desc[word] != 0)
      tma_fail(VENTUS_TMA_STATUS_INVALID_DESCRIPTOR, 3);
}

const uint32_t tma_control = tma_desc[VENTUS_TMA_V2_WORD_CONTROL];
const uint32_t tma_dtype = tma_control & 0x1f;
const uint32_t tma_rank = (tma_control >> 5) & 7;
const uint32_t tma_interleave = (tma_control >> 8) & 3;
const uint32_t tma_swizzle = (tma_control >> 10) & 7;
const uint32_t tma_swizzle_atomicity = (tma_control >> 13) & 3;
const uint32_t tma_l2_promotion = (tma_control >> 16) & 3;
const uint32_t tma_oob_fill = (tma_control >> 18) & 1;
const uint32_t tma_access_mode = (tma_control >> 19) & 3;
uint32_t tma_dtype_bits = 0;
bool tma_float = false;
switch (tma_dtype) {
  case VENTUS_TMA_DTYPE_U8: tma_dtype_bits = 8; break;
  case VENTUS_TMA_DTYPE_U16: tma_dtype_bits = 16; break;
  case VENTUS_TMA_DTYPE_U32:
  case VENTUS_TMA_DTYPE_S32: tma_dtype_bits = 32; break;
  case VENTUS_TMA_DTYPE_U64:
  case VENTUS_TMA_DTYPE_S64: tma_dtype_bits = 64; break;
  case VENTUS_TMA_DTYPE_FP16:
  case VENTUS_TMA_DTYPE_BF16: tma_dtype_bits = 16; tma_float = true; break;
  case VENTUS_TMA_DTYPE_FP32:
  case VENTUS_TMA_DTYPE_FP32_FTZ:
  case VENTUS_TMA_DTYPE_TF32:
  case VENTUS_TMA_DTYPE_TF32_FTZ:
    tma_dtype_bits = 32; tma_float = true; break;
  case VENTUS_TMA_DTYPE_FP64: tma_dtype_bits = 64; tma_float = true; break;
  case VENTUS_TMA_DTYPE_B4X16:
  case VENTUS_TMA_DTYPE_B4X16_P64: tma_dtype_bits = 4; break;
  case VENTUS_TMA_DTYPE_B6: tma_dtype_bits = 6; break;
  default:
    tma_fail(VENTUS_TMA_STATUS_UNSUPPORTED_FEATURE, 4);
    break;
}
const bool tma_sub_byte = tma_dtype >= VENTUS_TMA_DTYPE_B4X16;
const bool tma_padded_sub_byte =
    tma_dtype == VENTUS_TMA_DTYPE_B4X16_P64 ||
    tma_dtype == VENTUS_TMA_DTYPE_B6;

const bool tma_reduce = tma_reduce_mode != VENTUS_TMA_V2_REDUCE_COPY;
static_assert(VENTUS_TMA_V2_REDUCE_ATOMIC == 1,
              "Ventus TMA reduce must use atomic element updates");
if (tma_reduce && (!tma_s2g || tma_reduce_mode == VENTUS_TMA_V2_REDUCE_RESERVED ||
                   (tma_dtype != VENTUS_TMA_DTYPE_U32 &&
                    tma_dtype != VENTUS_TMA_DTYPE_S32)))
  tma_fail(VENTUS_TMA_STATUS_UNSUPPORTED_FEATURE, 12);

if ((tma_control >> 21) != 0 || (tma_control & (1u << 15)) != 0)
  tma_fail(VENTUS_TMA_STATUS_INVALID_DESCRIPTOR, 3);
if (tma_rank < 1 || tma_rank > VENTUS_TMA_V2_RANK_MAX)
  tma_fail(VENTUS_TMA_STATUS_INVALID_DESCRIPTOR, 5);
const unsigned tma_active_rank =
    tma_rank >= 1 && tma_rank <= VENTUS_TMA_V2_RANK_MAX ? tma_rank : 0;
if (tma_l2_promotion != 0 || tma_access_mode != 0 ||
    tma_swizzle_atomicity != VENTUS_TMA_SWIZZLE_ATOM_16B ||
    tma_swizzle > VENTUS_TMA_SWIZZLE_128B)
  tma_fail(VENTUS_TMA_STATUS_UNSUPPORTED_FEATURE, 12);
if (tma_oob_fill > VENTUS_TMA_OOB_NAN ||
    (tma_oob_fill == VENTUS_TMA_OOB_NAN && (!tma_float || tma_s2g)))
  tma_fail(VENTUS_TMA_STATUS_UNSUPPORTED_FEATURE, 12);
if (tma_interleave > VENTUS_TMA_INTERLEAVE_32B)
  tma_fail(VENTUS_TMA_STATUS_INVALID_DESCRIPTOR, 6);
if (tma_interleave != VENTUS_TMA_INTERLEAVE_NONE && tma_rank < 3)
  tma_fail(VENTUS_TMA_STATUS_INVALID_DESCRIPTOR, 6);
if (tma_interleave == VENTUS_TMA_INTERLEAVE_32B &&
    tma_swizzle != VENTUS_TMA_SWIZZLE_32B)
  tma_fail(VENTUS_TMA_STATUS_INVALID_DESCRIPTOR, 6);
const uint64_t tma_global_base64 =
    static_cast<uint64_t>(tma_desc[VENTUS_TMA_V2_WORD_GLOBAL_BASE]) |
    (static_cast<uint64_t>(
         tma_desc[VENTUS_TMA_V2_WORD_GLOBAL_BASE + 1]) << 32);
const uint32_t tma_global_alignment =
    tma_padded_sub_byte || tma_interleave == VENTUS_TMA_INTERLEAVE_32B
        ? 32 : 16;
if ((tma_global_base64 & (tma_global_alignment - 1)) != 0)
  tma_fail(VENTUS_TMA_STATUS_INVALID_DESCRIPTOR, 7);
if ((tma_shared_base & 127) != 0)
  tma_fail(VENTUS_TMA_STATUS_INVALID_DESCRIPTOR, 7);
if (tma_global_base64 > 0xffffffffULL)
  tma_fail(VENTUS_TMA_STATUS_ADDRESS_OVERFLOW, 11);
const uint32_t tma_global_base = static_cast<uint32_t>(tma_global_base64);

uint64_t tma_dims[5] = {};
uint64_t tma_strides[4] = {};
uint32_t tma_box[5] = {};
uint32_t tma_element_strides[5] = {};
uint32_t tma_counts[5] = {};
int64_t tma_coords[5] = {};
for (unsigned d = 0; d < 5; ++d) {
  const uint32_t raw_dim = tma_desc[VENTUS_TMA_V2_WORD_GLOBAL_DIMS + d];
  tma_dims[d] = raw_dim == 0 && d < tma_rank
                    ? (uint64_t{1} << 32) : raw_dim;
  tma_box[d] = tma_desc[VENTUS_TMA_V2_WORD_BOX_DIMS + d];
  tma_element_strides[d] =
      tma_desc[VENTUS_TMA_V2_WORD_ELEMENT_STRIDES + d];
  if (d < 4) {
    const uint32_t word = VENTUS_TMA_V2_WORD_GLOBAL_STRIDES + d * 2;
    tma_strides[d] = static_cast<uint64_t>(tma_desc[word]) |
                     (static_cast<uint64_t>(tma_desc[word + 1]) << 32);
  }
  tma_coords[d] = static_cast<int32_t>(P.VU.elt<uint32_t>(2, insn.rs2(), d));
  if (d < tma_rank) {
    if (tma_box[d] == 0 || tma_box[d] > 256)
      tma_fail(VENTUS_TMA_STATUS_INVALID_DESCRIPTOR, 8);
    tma_counts[d] = tma_box[d];
  } else if (tma_dims[d] != 0 || tma_box[d] != 0 ||
             tma_element_strides[d] != 0 ||
             (d < 4 && tma_strides[d] != 0)) {
    tma_fail(VENTUS_TMA_STATUS_INVALID_DESCRIPTOR, 3);
  }
}
for (unsigned d = 0; d < tma_active_rank; ++d)
  if (tma_element_strides[d] != 1)
    tma_fail(VENTUS_TMA_STATUS_UNSUPPORTED_FEATURE, 12);
if (tma_active_rank != 0)
  for (unsigned s = tma_active_rank - 1; s < 4; ++s)
    if (tma_strides[s] != 0)
      tma_fail(VENTUS_TMA_STATUS_INVALID_DESCRIPTOR, 3);
if (tma_s2g)
  for (unsigned d = 0; d < tma_active_rank; ++d)
    if (tma_coords[d] < 0)
      tma_fail(VENTUS_TMA_STATUS_INVALID_DESCRIPTOR, 10);

uint64_t tma_required_span =
    (static_cast<uint64_t>(tma_dims[0]) * tma_dtype_bits + 7) / 8;
for (unsigned d = 1; d < tma_active_rank; ++d) {
  const uint32_t alignment =
      tma_padded_sub_byte || tma_interleave == VENTUS_TMA_INTERLEAVE_32B
          ? 32 : 16;
  if (tma_strides[d - 1] == 0 || tma_strides[d - 1] % alignment != 0 ||
      tma_strides[d - 1] < tma_required_span)
    tma_fail(VENTUS_TMA_STATUS_INVALID_DESCRIPTOR, 9);
  tma_required_span =
      static_cast<uint64_t>(tma_strides[d - 1]) * tma_dims[d];
}
const uint32_t tma_row_bytes =
    (static_cast<uint64_t>(tma_box[0]) * tma_dtype_bits + 7) / 8;
if (tma_row_bytes == 0 || (tma_row_bytes & 15) != 0)
  tma_fail(VENTUS_TMA_STATUS_INVALID_DESCRIPTOR, 8);
if (tma_dtype == VENTUS_TMA_DTYPE_B4X16 && (tma_dims[0] & 1) != 0)
  tma_fail(VENTUS_TMA_STATUS_INVALID_DESCRIPTOR, 8);
if (tma_swizzle != VENTUS_TMA_SWIZZLE_NONE &&
    tma_row_bytes > (16u << tma_swizzle))
  tma_fail(VENTUS_TMA_STATUS_INVALID_DESCRIPTOR, 6);
if (tma_sub_byte && (tma_oob_fill != VENTUS_TMA_OOB_ZERO || tma_reduce))
  tma_fail(VENTUS_TMA_STATUS_UNSUPPORTED_FEATURE, 12);
if (tma_sub_byte && tma_interleave != VENTUS_TMA_INTERLEAVE_NONE)
  tma_fail(VENTUS_TMA_STATUS_UNSUPPORTED_FEATURE, 12);
if (tma_dtype == VENTUS_TMA_DTYPE_B4X16_P64 && tma_s2g)
  tma_fail(VENTUS_TMA_STATUS_UNSUPPORTED_FEATURE, 12);
if (tma_padded_sub_byte &&
    (tma_box[0] != 128 || (tma_dims[0] & 127) != 0 ||
     (tma_coords[0] & 127) != 0))
  tma_fail(VENTUS_TMA_STATUS_INVALID_DESCRIPTOR, 8);
if (tma_dtype == VENTUS_TMA_DTYPE_B6 &&
    !(tma_interleave == VENTUS_TMA_INTERLEAVE_NONE &&
      (tma_swizzle == VENTUS_TMA_SWIZZLE_NONE ||
       tma_swizzle == VENTUS_TMA_SWIZZLE_128B)))
  tma_fail(VENTUS_TMA_STATUS_UNSUPPORTED_FEATURE, 12);
if (tma_dtype == VENTUS_TMA_DTYPE_B4X16 && (tma_coords[0] & 31) != 0)
  tma_fail(VENTUS_TMA_STATUS_INVALID_DESCRIPTOR, 7);

// CUDA constrains the dynamic bounding-box origin to a 16B address. Keep
// signed coordinate arithmetic here so negative G2S OOB coordinates cannot
// wrap through an unsigned offset and issue an unrelated memory access.
if (tma_legal) {
  __int128 tma_bounding_origin = tma_global_base64;
  if (tma_interleave == VENTUS_TMA_INTERLEAVE_NONE || tma_sub_byte) {
    if (tma_dtype_bits == 4)
      tma_bounding_origin += static_cast<__int128>(tma_coords[0]) / 2;
    else if (tma_dtype_bits == 6)
      tma_bounding_origin += static_cast<__int128>(tma_coords[0]) * 3 / 4;
    else
      tma_bounding_origin += static_cast<__int128>(tma_coords[0]) *
                             (tma_dtype_bits / 8);
  } else {
    const uint32_t tma_slice_bytes =
        tma_interleave == VENTUS_TMA_INTERLEAVE_16B ? 16 : 32;
    const uint32_t tma_channels = tma_slice_bytes / (tma_dtype_bits / 8);
    const int64_t tma_coordinate0 = tma_coords[0];
    const int64_t tma_slice = tma_coordinate0 >= 0
        ? tma_coordinate0 / tma_channels
        : -static_cast<int64_t>(
              (static_cast<uint64_t>(-tma_coordinate0) + tma_channels - 1) /
              tma_channels);
    const uint32_t tma_in_slice =
        static_cast<uint32_t>(tma_coordinate0) & (tma_channels - 1);
    const __int128 tma_slice_stride =
        static_cast<__int128>(tma_strides[tma_rank - 3]) *
        tma_dims[tma_rank - 2];
    tma_bounding_origin += static_cast<__int128>(tma_slice) *
                           tma_slice_stride +
                           tma_in_slice * (tma_dtype_bits / 8);
  }
  for (unsigned d = 1; d < tma_active_rank; ++d)
    tma_bounding_origin += static_cast<__int128>(tma_coords[d]) *
                           tma_strides[d - 1];
  if (tma_bounding_origin < 0 || tma_bounding_origin > 0xffffffffULL)
    tma_fail(VENTUS_TMA_STATUS_ADDRESS_OVERFLOW, 11);
  else if ((static_cast<uint64_t>(tma_bounding_origin) & 15) != 0)
    tma_fail(VENTUS_TMA_STATUS_INVALID_DESCRIPTOR, 7);
}

uint64_t tma_logical_elements = tma_active_rank == 0 ? 0 : 1;
for (unsigned d = 0; d < tma_active_rank; ++d)
  tma_logical_elements *= tma_counts[d];
const uint64_t tma_logical_bytes =
    (tma_logical_elements * tma_dtype_bits + 7) / 8;
if (!tma_s2g && tma_legal && tma_binding.armed) {
  const reg_t state = MMU.load_uint32(tma_binding.address + 4);
  if (tma_logical_bytes == 0 ||
      tma_logical_bytes > tma_binding.bytes ||
      ((state >> 24) & 0xff) != tma_binding.generation)
    tma_fail(VENTUS_TMA_STATUS_MBARRIER_PROTOCOL, 4);
}

auto tma_swizzle_offset = [&](uint32_t logical, uint32_t row) -> uint32_t {
  if (tma_swizzle == VENTUS_TMA_SWIZZLE_NONE) return logical;
  const uint32_t span = 16u << tma_swizzle;
  const uint32_t mask = (1u << tma_swizzle) - 1;
  const uint32_t phase = (tma_shared_base >> 7) & mask;
  return (logical & ~(span - 1)) |
         ((((logical >> 4) & mask) ^ ((row + phase) & mask)) << 4) |
         (logical & 15);
};
auto tma_global_offset = [&](const int64_t coord[5]) -> uint64_t {
  const uint32_t elem_bytes = tma_dtype_bits / 8;
  if (tma_interleave == VENTUS_TMA_INTERLEAVE_NONE || tma_sub_byte) {
    uint64_t offset = tma_dtype_bits == 4
                          ? static_cast<uint64_t>(coord[0]) / 2
                          : tma_dtype_bits == 6
                                ? static_cast<uint64_t>(coord[0]) * 6 / 8
                                : static_cast<uint64_t>(coord[0]) * elem_bytes;
    for (unsigned d = 1; d < tma_rank; ++d)
      offset += static_cast<uint64_t>(coord[d]) * tma_strides[d - 1];
    return offset;
  }
  const uint32_t slice_bytes =
      tma_interleave == VENTUS_TMA_INTERLEAVE_16B ? 16 : 32;
  const uint32_t channels = slice_bytes / elem_bytes;
  const uint64_t slice = static_cast<uint64_t>(coord[0]) / channels;
  const uint64_t in_slice = static_cast<uint64_t>(coord[0]) % channels;
  const uint64_t slice_stride =
      static_cast<uint64_t>(tma_strides[tma_rank - 3]) *
      tma_dims[tma_rank - 2];
  uint64_t offset = in_slice * elem_bytes + slice * slice_stride;
  for (unsigned d = 1; d < tma_rank; ++d)
    offset += static_cast<uint64_t>(coord[d]) * tma_strides[d - 1];
  return offset;
};

if (tma_legal) {
  uint32_t index[5] = {};
  bool done = false;
  while (!done) {
    int64_t coord[5] = {};
    uint64_t linear_element = 0;
    uint64_t multiplier = 1;
    uint32_t row = 0;
    uint32_t row_multiplier = 1;
    bool in_bounds = true;
    for (unsigned d = 0; d < tma_rank; ++d) {
      coord[d] = tma_coords[d] + static_cast<int64_t>(index[d]);
      in_bounds = in_bounds && coord[d] >= 0 &&
                  static_cast<uint64_t>(coord[d]) < tma_dims[d];
      linear_element += static_cast<uint64_t>(index[d]) * multiplier;
      multiplier *= tma_counts[d];
      if (d > 0) {
        row += index[d] * row_multiplier;
        row_multiplier *= tma_counts[d];
      }
    }
    if (tma_padded_sub_byte && index[0] == 0) {
      const uint32_t packed_bytes =
          tma_dtype == VENTUS_TMA_DTYPE_B4X16_P64 ? 64 : 96;
      const uint64_t global64 =
          tma_global_base64 + tma_global_offset(coord);
      if (global64 + packed_bytes - 1 > 0xffffffffULL) {
        tma_fail(VENTUS_TMA_STATUS_ADDRESS_OVERFLOW, 11);
      } else if (in_bounds) {
        if (!tma_s2g) {
          const uint32_t bytes_per_lane =
              tma_dtype == VENTUS_TMA_DTYPE_B4X16_P64 ? 8 : 12;
          for (unsigned lane = 0; lane < 8; ++lane) {
            for (unsigned byte = 0; byte < bytes_per_lane; ++byte) {
              const uint32_t logical = row * 128 + lane * 16 + byte;
              const reg_t shared = tma_shared_base +
                  tma_swizzle_offset(logical, row);
              MMU.store_uint8(
                  shared,
                  MMU.load_uint8(static_cast<reg_t>(
                      global64 + lane * bytes_per_lane + byte)));
            }
          }
        } else {
          // .b6p2x16: pack the low six bits of each 16-byte shared lane.
          for (unsigned lane = 0; lane < 8; ++lane) {
            uint32_t accumulator = 0;
            unsigned accumulator_bits = 0;
            unsigned output_byte = 0;
            for (unsigned element = 0; element < 16; ++element) {
              const uint32_t logical = row * 128 + lane * 16 + element;
              const reg_t shared = tma_shared_base +
                  tma_swizzle_offset(logical, row);
              accumulator |=
                  (static_cast<uint32_t>(MMU.load_uint8(shared)) & 0x3f)
                  << accumulator_bits;
              accumulator_bits += 6;
              while (accumulator_bits >= 8) {
                MMU.store_uint8(
                    static_cast<reg_t>(global64 + lane * 12 + output_byte),
                    accumulator & 0xff);
                accumulator >>= 8;
                accumulator_bits -= 8;
                ++output_byte;
              }
            }
          }
        }
      } else if (!tma_s2g) {
        const uint32_t valid_bytes =
            tma_dtype == VENTUS_TMA_DTYPE_B4X16_P64 ? 8 : 12;
        for (unsigned lane = 0; lane < 8; ++lane)
          for (unsigned byte = 0; byte < valid_bytes; ++byte) {
            const uint32_t logical = row * 128 + lane * 16 + byte;
            MMU.store_uint8(
                tma_shared_base + tma_swizzle_offset(logical, row), 0);
          }
      }
    } else {
      const bool packed_skip = tma_dtype_bits == 4 && (index[0] & 1);
      if (!packed_skip && !tma_padded_sub_byte) {
      const uint32_t elem_bytes = tma_dtype_bits == 4 ? 1 : tma_dtype_bits / 8;
      const uint32_t byte_in_row = tma_dtype_bits == 4
                                       ? index[0] / 2
                                       : index[0] * elem_bytes;
      // CUDA gives every swizzled logical row a physical pitch equal to
      // the selected span. Short rows leave shared-memory padding rather
      // than being tightly packed before the XOR permutation.
      const uint32_t logical_byte =
          tma_swizzle == VENTUS_TMA_SWIZZLE_NONE
              ? (tma_dtype_bits == 4
                     ? linear_element / 2
                     : linear_element * elem_bytes)
              : row * (16u << tma_swizzle) + byte_in_row;
      const reg_t shared = tma_shared_base +
                           tma_swizzle_offset(logical_byte, row);
      if (in_bounds) {
        const reg_t global = tma_global_base + tma_global_offset(coord);
        if (static_cast<uint64_t>(global) + elem_bytes - 1 >
            0xffffffffULL) {
          tma_fail(VENTUS_TMA_STATUS_ADDRESS_OVERFLOW, 11);
        } else if (tma_reduce) {
          // Spike is sequential, so this direct update is the architectural
          // model of one element-wise atomic reduction.
          const uint32_t operand = MMU.load_uint32(shared);
          const uint32_t old = MMU.load_uint32(global);
          uint32_t result = old;
          switch (tma_reduce_mode) {
            case VENTUS_TMA_V2_REDUCE_ADD: result = old + operand; break;
            case VENTUS_TMA_V2_REDUCE_MIN:
              result = tma_dtype == VENTUS_TMA_DTYPE_S32
                           ? (static_cast<int32_t>(old) < static_cast<int32_t>(operand)
                                  ? old : operand)
                           : (old < operand ? old : operand);
              break;
            case VENTUS_TMA_V2_REDUCE_MAX:
              result = tma_dtype == VENTUS_TMA_DTYPE_S32
                           ? (static_cast<int32_t>(old) > static_cast<int32_t>(operand)
                                  ? old : operand)
                           : (old > operand ? old : operand);
              break;
            case VENTUS_TMA_V2_REDUCE_AND: result = old & operand; break;
            case VENTUS_TMA_V2_REDUCE_OR: result = old | operand; break;
            case VENTUS_TMA_V2_REDUCE_XOR: result = old ^ operand; break;
          }
          MMU.store_uint32(global, result);
        } else {
          for (unsigned byte = 0; byte < elem_bytes; ++byte) {
            if (tma_s2g)
              MMU.store_uint8(global + byte, MMU.load_uint8(shared + byte));
            else
              MMU.store_uint8(shared + byte, MMU.load_uint8(global + byte));
          }
        }
      } else if (!tma_s2g) {
        const uint8_t fp16_nan[2] = {0x00, 0x7e};
        const uint8_t bf16_nan[2] = {0xc0, 0x7f};
        const uint8_t fp32_nan[4] = {0x00, 0x00, 0xc0, 0x7f};
        const uint8_t fp64_nan[8] = {0, 0, 0, 0, 0, 0, 0xf8, 0x7f};
        for (unsigned byte = 0; byte < elem_bytes; ++byte) {
          uint8_t fill = 0;
          if (tma_oob_fill == VENTUS_TMA_OOB_NAN) {
            if (tma_dtype == VENTUS_TMA_DTYPE_FP16) fill = fp16_nan[byte];
            else if (tma_dtype == VENTUS_TMA_DTYPE_BF16) fill = bf16_nan[byte];
            else if (tma_dtype == VENTUS_TMA_DTYPE_FP64) fill = fp64_nan[byte];
            else fill = fp32_nan[byte];
          }
          MMU.store_uint8(shared + byte, fill);
        }
      }
      }
    }
    unsigned d = 0;
    while (d < tma_rank) {
      ++index[d];
      if (index[d] < tma_counts[d]) break;
      index[d] = 0;
      ++d;
    }
    done = d == tma_rank;
  }
} else {
  tma_set_status(tma_error_code, tma_error_detail);
}
if (tma_legal)
  tma_complete_g2s(tma_logical_bytes);
