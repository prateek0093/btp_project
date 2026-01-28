"""
Decoder module for FSQ file format.

This module provides functions to read FSQ files from disk and parse them
into Python objects for easy access to frames and blocks.
"""

import os
from typing import Optional, Dict, Tuple
from .models import FSQFile, FSQFrame, FSQBlock
from .utils import (
    unpack_file_header,
    unpack_frame_header,
    unpack_block_header,
    unpack_block_data,
    unpack_index_entry,
    FILE_HEADER_SIZE,
    FRAME_HEADER_SIZE,
    BLOCK_HEADER_SIZE
)


def read_fsq(filename: str, use_index: bool = True) -> FSQFile:
    """
    Read and parse an FSQ file from disk.
    
    Args:
        filename: Path to .fsq file
        use_index: Whether to use frame index for faster access (if available)
    
    Returns:
        Parsed FSQFile object with all frames and blocks
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
        IOError: If file cannot be read
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    
    with open(filename, 'rb') as f:
        # Read and parse file header
        header_data = f.read(FILE_HEADER_SIZE)
        if len(header_data) < FILE_HEADER_SIZE:
            raise ValueError(
                f"File too small: expected at least {FILE_HEADER_SIZE} bytes, "
                f"got {len(header_data)}"
            )
        
        (magic, version, header_size, max_width, max_height, total_frames,
         dtype, reserved1, index_offset, reserved2) = unpack_file_header(header_data)
        
        # Create FSQFile object
        fsq_file = FSQFile(
            magic=magic,
            version=version,
            header_size=header_size,
            max_width=max_width,
            max_height=max_height,
            total_frames=total_frames,
            dtype=dtype,
            reserved1=reserved1,
            index_offset=index_offset,
            reserved2=reserved2
        )
        
        # Read index if available and requested
        index_dict = None
        if use_index and index_offset > 0:
            index_dict = _read_index(f, index_offset, total_frames)
            fsq_file.index = index_dict
        
        # Read frames
        if index_dict and use_index:
            # Use index for fast seeking
            fsq_file.frames = _read_frames_with_index(f, index_dict, total_frames)
        else:
            # Sequential read
            fsq_file.frames = _read_frames_sequential(f, total_frames)
        
        return fsq_file


def _read_index(f, index_offset: int, total_frames: int) -> Dict[int, Tuple[int, int]]:
    """
    Read frame index from file.
    
    Args:
        f: File object
        index_offset: Byte offset to index
        total_frames: Expected number of index entries
    
    Returns:
        Dictionary mapping frame_id to (offset, size)
    """
    f.seek(index_offset)
    index_dict = {}
    
    for _ in range(total_frames):
        entry_data = f.read(12)  # 4 bytes frame_id + 8 bytes offset
        if len(entry_data) < 12:
            break
        
        frame_id, offset = unpack_index_entry(entry_data)
        index_dict[frame_id] = (offset, 0)  # Size will be calculated from frame header
    
    return index_dict


def _read_frames_sequential(f, total_frames: int) -> list:
    """
    Read frames sequentially from current file position.
    
    Args:
        f: File object positioned at first frame
        total_frames: Number of frames to read
    
    Returns:
        List of FSQFrame objects
    """
    frames = []
    
    for _ in range(total_frames):
        frame = _read_frame(f)
        if frame is None:
            break
        frames.append(frame)
    
    return frames


def _read_frames_with_index(f, index_dict: Dict[int, Tuple[int, int]], total_frames: int) -> list:
    """
    Read frames using index for fast seeking.
    
    Args:
        f: File object
        index_dict: Index mapping frame_id to offset
        total_frames: Number of frames to read
    
    Returns:
        List of FSQFrame objects in order
    """
    frames = []
    
    for frame_id in range(total_frames):
        if frame_id not in index_dict:
            raise ValueError(f"Frame {frame_id} not found in index")
        
        offset, _ = index_dict[frame_id]
        f.seek(offset)
        frame = _read_frame(f)
        if frame is None:
            raise ValueError(f"Failed to read frame {frame_id} at offset {offset}")
        frames.append(frame)
    
    return frames


def _read_frame(f) -> Optional[FSQFrame]:
    """
    Read a single frame from current file position.
    
    Args:
        f: File object positioned at frame header
    
    Returns:
        FSQFrame object or None if end of file
    """
    # Read frame header
    frame_header_data = f.read(FRAME_HEADER_SIZE)
    if len(frame_header_data) < FRAME_HEADER_SIZE:
        return None
    
    frame_id, num_blocks, frame_size_bytes = unpack_frame_header(frame_header_data)
    
    frame = FSQFrame(
        frame_id=frame_id,
        num_blocks=num_blocks,
        frame_size_bytes=frame_size_bytes
    )
    
    # Read blocks
    for _ in range(num_blocks):
        block = _read_block(f)
        if block is None:
            raise ValueError(
                f"Failed to read block {len(frame.blocks)} in frame {frame_id}"
            )
        frame.blocks.append(block)
    
    return frame


def _read_block(f) -> Optional[FSQBlock]:
    """
    Read a single block from current file position.
    
    Args:
        f: File object positioned at block header
    
    Returns:
        FSQBlock object or None if end of file
    """
    # Read block header
    block_header_data = f.read(BLOCK_HEADER_SIZE)
    if len(block_header_data) < BLOCK_HEADER_SIZE:
        return None
    
    x, y, size_top, size_bottom, data_bytes = unpack_block_header(block_header_data)
    
    # Read block data
    data_raw = f.read(data_bytes)
    if len(data_raw) < data_bytes:
        raise ValueError(
            f"Insufficient block data: expected {data_bytes} bytes, got {len(data_raw)}"
        )
    
    data = unpack_block_data(data_raw, size_top, size_bottom)
    
    block = FSQBlock(
        x=x,
        y=y,
        size_top=size_top,
        size_bottom=size_bottom,
        data_bytes=data_bytes,
        data=data
    )
    
    return block


def get_frame(file: FSQFile, frame_id: int) -> FSQFrame:
    """
    Get a specific frame by ID.
    
    Args:
        file: FSQFile object
        frame_id: Frame identifier
    
    Returns:
        FSQFrame object
    
    Raises:
        ValueError: If frame_id is invalid
    """
    if frame_id < 0 or frame_id >= len(file.frames):
        raise ValueError(
            f"Invalid frame_id: {frame_id} (file has {len(file.frames)} frames)"
        )
    
    return file.frames[frame_id]


def get_block(frame: FSQFrame, block_index: int) -> FSQBlock:
    """
    Get a specific block from a frame by index.
    
    Args:
        frame: FSQFrame object
        block_index: Block index (0-based)
    
    Returns:
        FSQBlock object
    
    Raises:
        ValueError: If block_index is invalid
    """
    if block_index < 0 or block_index >= len(frame.blocks):
        raise ValueError(
            f"Invalid block_index: {block_index} (frame has {len(frame.blocks)} blocks)"
        )
    
    return frame.blocks[block_index]


def get_frame_by_id_fast(filename: str, frame_id: int) -> FSQFrame:
    """
    Fast retrieval of a single frame using the index (if available).
    
    This function opens the file, reads only the requested frame using the index,
    and closes the file. More efficient than loading the entire file when you
    only need one frame.
    
    Args:
        filename: Path to .fsq file
        frame_id: Frame identifier to retrieve
    
    Returns:
        FSQFrame object
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If frame_id is invalid or index not available
    """
    with open(filename, 'rb') as f:
        # Read file header to get index offset
        header_data = f.read(FILE_HEADER_SIZE)
        (magic, version, header_size, max_width, max_height, total_frames,
         dtype, reserved1, index_offset, reserved2) = unpack_file_header(header_data)
        
        if frame_id < 0 or frame_id >= total_frames:
            raise ValueError(
                f"Invalid frame_id: {frame_id} (file has {total_frames} frames)"
            )
        
        if index_offset == 0:
            raise ValueError("File does not have an index; use read_fsq instead")
        
        # Read index entry for this frame
        index_entry_offset = index_offset + (frame_id * 12)
        f.seek(index_entry_offset)
        entry_data = f.read(12)
        frame_id_found, offset = unpack_index_entry(entry_data)
        
        if frame_id_found != frame_id:
            raise ValueError(
                f"Index mismatch: expected frame_id {frame_id}, got {frame_id_found}"
            )
        
        # Seek to frame and read it
        f.seek(offset)
        frame = _read_frame(f)
        
        if frame is None:
            raise ValueError(f"Failed to read frame {frame_id} at offset {offset}")
        
        return frame
