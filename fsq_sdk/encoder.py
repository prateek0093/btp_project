"""Encoder module for FSQ file format.

This module provides functions to create FSQ files, frames, and blocks,
and write them to disk in the FSQ binary format.
"""

import numpy as np
from typing import Union, List
from .models import FSQFile, FSQFrame, FSQBlock
from .utils import (
    pack_file_header,
    pack_frame_header,
    pack_block_header,
    pack_block_data,
    pack_index_entry,
    FRAME_HEADER_SIZE,
    BLOCK_HEADER_SIZE
)
from .logging_config import get_logger

# Module logger
_logger = get_logger('encoder')


def create_fsq_file(max_width: int, max_height: int, dtype: int = 1) -> FSQFile:
    """
    Create a new FSQ file object.
    
    Args:
        max_width: Maximum frame width (must be <= 1024)
        max_height: Maximum frame height (must be <= 1024)
        dtype: Data type identifier (1 for float32)
    
    Returns:
        Initialized FSQFile object
    
    Raises:
        ValueError: If dimensions exceed maximum or dtype is invalid
    """
    if max_width > 1024 or max_width <= 0:
        raise ValueError(f"max_width must be between 1 and 1024, got {max_width}")
    if max_height > 1024 or max_height <= 0:
        raise ValueError(f"max_height must be between 1 and 1024, got {max_height}")
    if dtype != 1:
        raise ValueError(f"Only dtype=1 (float32) is supported, got {dtype}")
    
    _logger.debug(f"Creating FSQ file with max_width={max_width}, max_height={max_height}")
    
    return FSQFile(
        max_width=max_width,
        max_height=max_height,
        dtype=dtype
    )


def add_frame(file: FSQFile, frame_id: int) -> FSQFrame:
    """
    Create and add a new frame to the FSQ file.
    
    Args:
        file: FSQFile object to add frame to
        frame_id: Sequential frame identifier
    
    Returns:
        Created FSQFrame object
    
    Raises:
        ValueError: If frame_id is not sequential
    """
    # Validate sequential frame IDs
    expected_id = len(file.frames)
    if frame_id != expected_id:
        raise ValueError(
            f"Frame IDs must be sequential. Expected {expected_id}, got {frame_id}"
        )
    
    frame = FSQFrame(
        frame_id=frame_id,
        num_blocks=0,
        frame_size_bytes=FRAME_HEADER_SIZE  # Start with just header size
    )
    
    file.add_frame(frame)
    _logger.debug(f"Added frame {frame_id} to file (total frames: {len(file.frames)})")
    return frame


def add_block(
    frame: FSQFrame,
    file: FSQFile,
    x: int,
    y: int,
    width: int,
    height: int,
    data: Union[List[List[float]], np.ndarray]
) -> FSQBlock:
    """
    Create and add a new block to a frame.

    Args:
        frame: FSQFrame object to add block to
        file: FSQFile object (for max_width/max_height validation)
        x: X-coordinate (column) position of block top-left corner
        y: Y-coordinate (row) position of block top-left corner
        width: Number of columns in the block (horizontal extent)
        height: Number of rows in the block (vertical extent)
        data: Two-dimensional array of float values, shape (width, height)

    Returns:
        Created FSQBlock object

    Raises:
        ValueError: If constraints are violated
    """
    # Validate position and size constraints
    if x < 0 or y < 0:
        raise ValueError(f"Block position must be non-negative: x={x}, y={y}")
    if x + width > file.max_width:
        raise ValueError(
            f"Block exceeds max_width: x={x} + width={width} = {x + width} "
            f"> {file.max_width}"
        )
    if y + height > file.max_height:
        raise ValueError(
            f"Block exceeds max_height: y={y} + height={height} "
            f"= {y + height} > {file.max_height}"
        )
    if width <= 0 or height <= 0:
        raise ValueError(
            f"Block dimensions must be positive: width={width}, "
            f"height={height}"
        )
    
    # Convert data to numpy array if needed
    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=np.float32)
    else:
        if data.dtype != np.float32:
            data = data.astype(np.float32)
    
    # Validate data shape
    if data.shape != (width, height):
        raise ValueError(
            f"Data shape mismatch: expected ({width}, {height}), "
            f"got {data.shape}"
        )
    
    data_bytes = width * height * 4
    
    block = FSQBlock(
        x=x,
        y=y,
        width=width,
        height=height,
        data_bytes=data_bytes,
        data=data
    )
    
    frame.add_block(block)
    
    # Update frame size
    frame.frame_size_bytes += BLOCK_HEADER_SIZE + data_bytes
    
    _logger.debug(
        f"Added block to frame {frame.frame_id}: pos=({x}, {y}), "
        f"size={width}x{height}, data_bytes={data_bytes}"
    )
    
    return block


def write_fsq(file: FSQFile, filename: str, include_index: bool = True):
    """
    Write FSQ file object to disk in binary format.
    
    Args:
        file: FSQFile object to write
        filename: Output filename (should end with .fsq)
        include_index: Whether to include frame index at end of file
    
    Raises:
        IOError: If file cannot be written
    """
    _logger.info(f"Writing FSQ file to '{filename}' (frames={file.total_frames}, include_index={include_index})")

    with open(filename, 'wb') as f:
        # Calculate index offset if including index
        index_offset = 0
        if include_index:
            # Header + all frames
            index_offset = 64  # File header
            for frame in file.frames:
                index_offset += frame.frame_size_bytes
        
        # Update file object
        file.index_offset = index_offset if include_index else 0
        
        # Write file header
        header_bytes = pack_file_header(
            magic=file.magic,
            version=file.version,
            header_size=file.header_size,
            max_width=file.max_width,
            max_height=file.max_height,
            total_frames=file.total_frames,
            dtype=file.dtype,
            reserved1=file.reserved1,
            index_offset=file.index_offset,
            reserved2=file.reserved2
        )
        f.write(header_bytes)
        
        # Track frame offsets for index
        frame_offsets = []
        current_offset = 64
        
        # Write frames
        for frame in file.frames:
            frame_offsets.append((frame.frame_id, current_offset))
            
            # Write frame header
            frame_header = pack_frame_header(
                frame_id=frame.frame_id,
                num_blocks=frame.num_blocks,
                frame_size_bytes=frame.frame_size_bytes
            )
            f.write(frame_header)
            
            # Write blocks
            for block in frame.blocks:
                # Write block header
                block_header = pack_block_header(
                    x=block.x,
                    y=block.y,
                    width=block.width,
                    height=block.height,
                    data_bytes=block.data_bytes
                )
                f.write(block_header)
                
                # Write block data
                block_data = pack_block_data(block.data)
                f.write(block_data)
            
            current_offset += frame.frame_size_bytes
        
        # Write index if requested
        if include_index:
            for frame_id, offset in frame_offsets:
                index_entry = pack_index_entry(frame_id, offset)
                f.write(index_entry)
            _logger.debug(f"Wrote index with {len(frame_offsets)} entries")
    
    _logger.info(f"Successfully wrote FSQ file: {filename}")
