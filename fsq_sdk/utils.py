"""
Utility functions for FSQ file format handling.

This module provides helper functions for packing and unpacking binary data
according to the FSQ format specification.
"""

import struct
import numpy as np
from typing import Tuple

# Struct formats as per FSQ specification
FILE_HEADER_FMT = "<4sHHHHIHHQ36s"  # 64 bytes total
FRAME_HEADER_FMT = "<IIQ"  # 16 bytes total
BLOCK_HEADER_FMT = "<HHHHQ"  # 16 bytes total

# Calculate struct sizes
FILE_HEADER_SIZE = struct.calcsize(FILE_HEADER_FMT)
FRAME_HEADER_SIZE = struct.calcsize(FRAME_HEADER_FMT)
BLOCK_HEADER_SIZE = struct.calcsize(BLOCK_HEADER_FMT)


def pack_file_header(
    magic: str,
    version: int,
    header_size: int,
    max_width: int,
    max_height: int,
    total_frames: int,
    dtype: int,
    reserved1: int,
    index_offset: int,
    reserved2: bytes
) -> bytes:
    """
    Pack file header into bytes.
    
    Args:
        magic: File magic identifier (4 bytes, e.g., 'FSQ1')
        version: Format version
        header_size: Size of header
        max_width: Maximum frame width
        max_height: Maximum frame height
        total_frames: Total number of frames
        dtype: Data type identifier
        reserved1: Reserved field
        index_offset: Offset to frame index
        reserved2: Reserved bytes (36 bytes)
    
    Returns:
        Packed bytes of file header
    """
    magic_bytes = magic.encode('ascii')
    if len(magic_bytes) != 4:
        raise ValueError(f"Magic must be exactly 4 bytes, got {len(magic_bytes)}")
    
    return struct.pack(
        FILE_HEADER_FMT,
        magic_bytes,
        version,
        header_size,
        max_width,
        max_height,
        total_frames,
        dtype,
        reserved1,
        index_offset,
        reserved2
    )


def unpack_file_header(data: bytes) -> Tuple:
    """
    Unpack file header from bytes.
    
    Args:
        data: Bytes containing file header (at least 64 bytes)
    
    Returns:
        Tuple of header fields
    """
    if len(data) < FILE_HEADER_SIZE:
        raise ValueError(
            f"Insufficient data for file header: expected {FILE_HEADER_SIZE} bytes, "
            f"got {len(data)}"
        )
    
    unpacked = struct.unpack(FILE_HEADER_FMT, data[:FILE_HEADER_SIZE])
    # Convert magic from bytes to string
    magic = unpacked[0].decode('ascii')
    return (magic,) + unpacked[1:]


def pack_frame_header(frame_id: int, num_blocks: int, frame_size_bytes: int) -> bytes:
    """
    Pack frame header into bytes.
    
    Args:
        frame_id: Frame identifier
        num_blocks: Number of blocks in frame
        frame_size_bytes: Total size of frame data
    
    Returns:
        Packed bytes of frame header
    """
    return struct.pack(FRAME_HEADER_FMT, frame_id, num_blocks, frame_size_bytes)


def unpack_frame_header(data: bytes) -> Tuple[int, int, int]:
    """
    Unpack frame header from bytes.
    
    Args:
        data: Bytes containing frame header (at least 16 bytes)
    
    Returns:
        Tuple of (frame_id, num_blocks, frame_size_bytes)
    """
    if len(data) < FRAME_HEADER_SIZE:
        raise ValueError(
            f"Insufficient data for frame header: expected {FRAME_HEADER_SIZE} bytes, "
            f"got {len(data)}"
        )
    
    return struct.unpack(FRAME_HEADER_FMT, data[:FRAME_HEADER_SIZE])


def pack_block_header(
    x: int, y: int, width: int, height: int, data_bytes: int
) -> bytes:
    """
    Pack block header into bytes.
    
    Args:
        x: X-coordinate (column position)
        y: Y-coordinate (row position)
        width: Number of columns (horizontal extent)
        height: Number of rows (vertical extent)
        data_bytes: Size of data in bytes
    
    Returns:
        Packed bytes of block header
    """
    return struct.pack(BLOCK_HEADER_FMT, x, y, width, height, data_bytes)


def unpack_block_header(data: bytes) -> Tuple[int, int, int, int, int]:
    """
    Unpack block header from bytes.
    
    Args:
        data: Bytes containing block header (at least 16 bytes)
    
    Returns:
        Tuple of (x, y, width, height, data_bytes)
    """
    if len(data) < BLOCK_HEADER_SIZE:
        raise ValueError(
            f"Insufficient data for block header: expected {BLOCK_HEADER_SIZE} bytes, "
            f"got {len(data)}"
        )
    
    return struct.unpack(BLOCK_HEADER_FMT, data[:BLOCK_HEADER_SIZE])


def pack_block_data(data: np.ndarray) -> bytes:
    """
    Pack block data (2D float32 array) into bytes.
    
    Args:
        data: 2D numpy array of float32 values
    
    Returns:
        Packed bytes of block data (little-endian)
    """
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    
    # Ensure little-endian byte order
    if data.dtype.byteorder == '>':
        data = data.astype('<f4')
    
    return data.tobytes()


def unpack_block_data(data: bytes, width: int, height: int) -> np.ndarray:
    """
    Unpack block data from bytes into 2D float32 array.
    
    Args:
        data: Bytes containing block data
        width: Number of columns (horizontal extent)
        height: Number of rows (vertical extent)
    
    Returns:
        2D numpy array of shape (width, height), dtype float32
    """
    expected_bytes = width * height * 4
    if len(data) < expected_bytes:
        raise ValueError(
            f"Insufficient data for block: expected {expected_bytes} bytes, "
            f"got {len(data)}"
        )
    
    # Create array from bytes (little-endian float32)
    arr = np.frombuffer(data[:expected_bytes], dtype='<f4')
    
    # Reshape to 2D (width, height)
    return arr.reshape(width, height)


def pack_index_entry(frame_id: int, offset: int) -> bytes:
    """
    Pack a single index entry.
    
    Args:
        frame_id: Frame identifier
        offset: Byte offset to frame
    
    Returns:
        Packed bytes (12 bytes: I + Q)
    """
    return struct.pack("<IQ", frame_id, offset)


def unpack_index_entry(data: bytes) -> Tuple[int, int]:
    """
    Unpack a single index entry.
    
    Args:
        data: Bytes containing index entry (at least 12 bytes)
    
    Returns:
        Tuple of (frame_id, offset)
    """
    return struct.unpack("<IQ", data[:12])
