"""
Data models for FSQ file format.

This module defines the core data structures used to represent FSQ files,
frames, and blocks in memory.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np


@dataclass
class FSQBlock:
    """
    Represents a single block within a frame.
    
    Attributes:
        x: X-coordinate position in the frame
        y: Y-coordinate position in the frame
        size_top: Width of the block (S dimension)
        size_bottom: Height of the block (T dimension)
        data_bytes: Size of data in bytes (should be S * T * 4)
        data: 2D numpy array of float32 values, shape (size_top, size_bottom)
    """
    x: int
    y: int
    size_top: int
    size_bottom: int
    data_bytes: int
    data: np.ndarray  # shape (size_top, size_bottom), dtype float32
    
    def __post_init__(self):
        """Validate block data after initialization."""
        expected_bytes = self.size_top * self.size_bottom * 4
        if self.data_bytes != expected_bytes:
            raise ValueError(
                f"Data bytes mismatch: expected {expected_bytes}, got {self.data_bytes}"
            )
        if self.data.shape != (self.size_top, self.size_bottom):
            raise ValueError(
                f"Data shape mismatch: expected ({self.size_top}, {self.size_bottom}), "
                f"got {self.data.shape}"
            )
        if self.data.dtype != np.float32:
            raise ValueError(f"Data must be float32, got {self.data.dtype}")


@dataclass
class FSQFrame:
    """
    Represents a single frame containing multiple blocks.
    
    Attributes:
        frame_id: Sequential frame identifier
        num_blocks: Number of blocks in this frame
        frame_size_bytes: Total size of frame data in bytes (header + all blocks)
        blocks: List of FSQBlock objects
    """
    frame_id: int
    num_blocks: int
    frame_size_bytes: int
    blocks: List[FSQBlock] = field(default_factory=list)
    
    def add_block(self, block: FSQBlock):
        """Add a block to this frame."""
        self.blocks.append(block)
        self.num_blocks = len(self.blocks)


@dataclass
class FSQFile:
    """
    Represents a complete FSQ file with header, frames, and optional index.
    
    Attributes:
        magic: File magic identifier (should be 'FSQ1')
        version: Format version (should be 1)
        header_size: Size of file header in bytes (64)
        max_width: Maximum frame width
        max_height: Maximum frame height
        total_frames: Total number of frames in the file
        dtype: Data type identifier (1 for float32)
        reserved1: Reserved field
        index_offset: Offset to frame index (0 if no index)
        reserved2: Reserved bytes
        frames: List of FSQFrame objects
        index: Optional dictionary mapping frame_id to (offset, size)
    """
    magic: str = 'FSQ1'
    version: int = 1
    header_size: int = 64
    max_width: int = 1024
    max_height: int = 1024
    total_frames: int = 0
    dtype: int = 1
    reserved1: int = 0
    index_offset: int = 0
    reserved2: bytes = field(default_factory=lambda: b'\x00' * 36)
    frames: List[FSQFrame] = field(default_factory=list)
    index: Optional[Dict[int, Tuple[int, int]]] = None
    
    def __post_init__(self):
        """Validate file header after initialization."""
        if self.magic != 'FSQ1':
            raise ValueError(f"Invalid magic: expected 'FSQ1', got '{self.magic}'")
        if self.version != 1:
            raise ValueError(f"Invalid version: expected 1, got {self.version}")
        if self.header_size != 64:
            raise ValueError(f"Invalid header size: expected 64, got {self.header_size}")
        if self.max_width > 1024 or self.max_height > 1024:
            raise ValueError(
                f"Dimensions exceed maximum: max_width={self.max_width}, "
                f"max_height={self.max_height} (max 1024)"
            )
        if self.dtype != 1:
            raise ValueError(f"Invalid dtype: expected 1 (float32), got {self.dtype}")
        if len(self.reserved2) != 36:
            raise ValueError(f"Reserved2 must be 36 bytes, got {len(self.reserved2)}")
    
    def add_frame(self, frame: FSQFrame):
        """Add a frame to this file."""
        # Validate sequential frame IDs
        if self.frames and frame.frame_id != self.frames[-1].frame_id + 1:
            if frame.frame_id != 0 and frame.frame_id != len(self.frames):
                raise ValueError(
                    f"Frame IDs must be sequential. Expected {len(self.frames)}, "
                    f"got {frame.frame_id}"
                )
        self.frames.append(frame)
        self.total_frames = len(self.frames)
