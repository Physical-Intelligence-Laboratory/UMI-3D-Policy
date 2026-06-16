#pragma once

#include <cstdint>

static constexpr uint32_t HIK_SHM_MAGIC = 0x48494B31;   // "HIK1"
static constexpr size_t   HIK_MAX_IMAGE_BYTES = 5 * 1024 * 1024;  // 5 MB

#pragma pack(push, 1)
struct HikShmHeader {
    uint32_t magic;        // HIK_SHM_MAGIC
    uint32_t width;
    uint32_t height;
    uint32_t channels;     // 3
    uint32_t data_bytes;   // width * height * channels
    uint32_t status;       // 0 invalid, 1 valid
    uint64_t seq;          // even = stable, odd = writer in progress
    double   timestamp;    // seconds
};
#pragma pack(pop)

static constexpr size_t HIK_SHM_TOTAL_BYTES =
    sizeof(HikShmHeader) + HIK_MAX_IMAGE_BYTES;