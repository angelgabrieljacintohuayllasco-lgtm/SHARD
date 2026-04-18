"""
SHARD Binary Record Format — v1

Every record in a .bin shard file has the following layout:

  Offset   Size   Type          Description
  ------   ----   ----          -----------
  0        4      uint32 BE     record_length: total bytes in this record
                                (from key_length field to CRC, inclusive)
  4        2      uint16 BE     key_length: byte length of the key
  6        N      bytes         key: UTF-8 encoded key string
  6+N      4      uint32 BE     value_length: byte length of the value
  10+N     M      bytes         value: UTF-8 encoded value (typically JSON)
  10+N+M   4      uint32 BE     crc32: checksum of all bytes from key_length
                                to end of value (excludes this field itself)

Each shard file begins with an 8-byte header:
  Bytes 0-3: MAGIC = b"SHRD"
  Bytes 4-5: FORMAT_VERSION = 1 (uint16 BE)
  Bytes 6-7: reserved = 0 (uint16 BE)

The record_length prefix enables O(1) skip-ahead during scans: reading 4 bytes
tells you exactly how many bytes to advance to reach the next record.
"""

import struct
import zlib
from typing import Tuple

# ── File header ────────────────────────────────────────────────────────────────
MAGIC = b"SHRD"
FORMAT_VERSION = 1
_HEADER_FMT = ">4sHH"  # magic(4) + version(2) + reserved(2)
HEADER_SIZE = struct.calcsize(_HEADER_FMT)

# ── Record field formats ───────────────────────────────────────────────────────
_REC_LEN_FMT = ">I"   # 4 bytes: total record body length
_KEY_LEN_FMT = ">H"   # 2 bytes: key byte length
_VAL_LEN_FMT = ">I"   # 4 bytes: value byte length
_CRC_FMT = ">I"       # 4 bytes: CRC32 checksum

_REC_LEN_SIZE = struct.calcsize(_REC_LEN_FMT)
_KEY_LEN_SIZE = struct.calcsize(_KEY_LEN_FMT)
_VAL_LEN_SIZE = struct.calcsize(_VAL_LEN_FMT)
_CRC_SIZE = struct.calcsize(_CRC_FMT)


def encode_record(key: str, value: str) -> bytes:
    """
    Encode a key-value pair into SHARD binary format.

    Args:
        key:   The record key (e.g. a word or ID).
        value: The record value (e.g. a JSON string).

    Returns:
        Bytes ready to be appended to a shard file.
    """
    key_bytes = key.encode("utf-8")
    val_bytes = value.encode("utf-8")

    body = (
        struct.pack(_KEY_LEN_FMT, len(key_bytes))
        + key_bytes
        + struct.pack(_VAL_LEN_FMT, len(val_bytes))
        + val_bytes
    )

    crc = zlib.crc32(body) & 0xFFFFFFFF
    body += struct.pack(_CRC_FMT, crc)

    return struct.pack(_REC_LEN_FMT, len(body)) + body


def decode_record(data: bytes, offset: int = 0) -> Tuple[str, str, int]:
    """
    Decode a single record from a byte buffer starting at *offset*.

    Args:
        data:   Byte buffer (or mmap object) containing one or more records.
        offset: Byte offset of the start of this record.

    Returns:
        Tuple of (key, value, bytes_consumed).

    Raises:
        ValueError: If the CRC32 checksum does not match (data corruption).
        struct.error: If the buffer is too short to contain a complete record.
    """
    rec_len = struct.unpack_from(_REC_LEN_FMT, data, offset)[0]
    offset += _REC_LEN_SIZE

    body = data[offset: offset + rec_len]

    # Validate CRC32
    stored_crc = struct.unpack_from(_CRC_FMT, body, len(body) - _CRC_SIZE)[0]
    computed_crc = zlib.crc32(body[: -_CRC_SIZE]) & 0xFFFFFFFF
    if stored_crc != computed_crc:
        raise ValueError(
            f"CRC mismatch at offset {offset}: "
            f"stored={stored_crc:#010x}, computed={computed_crc:#010x}"
        )

    pos = 0
    key_len = struct.unpack_from(_KEY_LEN_FMT, body, pos)[0]
    pos += _KEY_LEN_SIZE
    key = body[pos: pos + key_len].decode("utf-8")
    pos += key_len

    val_len = struct.unpack_from(_VAL_LEN_FMT, body, pos)[0]
    pos += _VAL_LEN_SIZE
    value = body[pos: pos + val_len].decode("utf-8")

    bytes_consumed = _REC_LEN_SIZE + rec_len
    return key, value, bytes_consumed


def shard_file_header() -> bytes:
    """Return the 8-byte magic header for a new shard file."""
    return struct.pack(_HEADER_FMT, MAGIC, FORMAT_VERSION, 0)


def validate_header(data: bytes) -> bool:
    """Return True if the bytes start with a valid SHARD file header."""
    if len(data) < HEADER_SIZE:
        return False
    magic, version, _ = struct.unpack_from(_HEADER_FMT, data, 0)
    return magic == MAGIC and version == FORMAT_VERSION
