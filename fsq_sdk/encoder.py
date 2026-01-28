"""
Encoder module for FSQ file format.

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
    return frame


def add_block(
    frame: FSQFrame,
    file: FSQFile,
    x: int,
    y: int,
    size_top: int,
    size_bottom: int,
    data: Union[List[List[float]], np.ndarray]
) -> FSQBlock:
    """
    Create and add a new block to a frame.
    
    Args:
        frame: FSQFrame object to add block to
        file: FSQFile object (for max_width/max_height validation)
        x: X-coordinate position in frame
        y: Y-coordinate position in frame
        size_top: Width of block (S dimension)
        size_bottom: Height of block (T dimension)
        data: 2D array of float values, shape (size_top, size_bottom)
    
    Returns:
        Created FSQBlock object
    
    Raises:
        ValueError: If constraints are violated
    """
    # Validate position and size constraints
    if x < 0 or y < 0:
        raise ValueError(f"Block position must be non-negative: x={x}, y={y}")
    if x + size_top > file.max_width:
        raise ValueError(
            f"Block exceeds max_width: x={x} + size_top={size_top} = {x + size_top} "
            f"> {file.max_width}"
        )
    if y + size_bottom > file.max_height:
        raise ValueError(
            f"Block exceeds max_height: y={y} + size_bottom={size_bottom} "
            f"= {y + size_bottom} > {file.max_height}"
        )
    if size_top <= 0 or size_bottom <= 0:
        raise ValueError(
            f"Block dimensions must be positive: size_top={size_top}, "
            f"size_bottom={size_bottom}"
        )
    
    # Convert data to numpy array if needed
    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=np.float32)
    else:
        if data.dtype != np.float32:
            data = data.astype(np.float32)
    
    # Validate data shape
    if data.shape != (size_top, size_bottom):
        raise ValueError(
            f"Data shape mismatch: expected ({size_top}, {size_bottom}), "
            f"got {data.shape}"
        )
    
    data_bytes = size_top * size_bottom * 4
    
    block = FSQBlock(
        x=x,
        y=y,
        size_top=size_top,
        size_bottom=size_bottom,
        data_bytes=data_bytes,
        data=data
    )
    
    frame.add_block(block)
    
    # Update frame size
    frame.frame_size_bytes += BLOCK_HEADER_SIZE + data_bytes
    
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
                    size_top=block.size_top,
                    size_bottom=block.size_bottom,
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
                index_entry = pack_index_entry(frame_id, offset, 0)
                f.write(index_entry)
