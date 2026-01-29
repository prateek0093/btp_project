"""
FSQ SDK - Python SDK for FSQ file format (version 1)

This package provides tools for encoding and decoding data in the FSQ binary file format.
FSQ is designed for efficient storage of frame-based data with rectangular blocks of float32 values.

Main Components:
    - FSQFile, FSQFrame, FSQBlock: Data models representing the file structure
    - create_fsq_file, add_frame, add_block, write_fsq: Encoding functions
    - read_fsq, get_frame, get_block: Decoding functions

Example usage:
    # Creating an FSQ file
    >>> from fsq_sdk import create_fsq_file, add_frame, add_block, write_fsq
    >>> import numpy as np
    >>> 
    >>> # Create file with max dimensions
    >>> fsq = create_fsq_file(max_width=512, max_height=512)
    >>> 
    >>> # Add a frame
    >>> frame = add_frame(fsq, frame_id=0)
    >>> 
    >>> # Add a block with data
    >>> data = np.random.rand(100, 100).astype(np.float32)
    >>> block = add_block(frame, fsq, x=0, y=0, width=100, height=100, data=data)
    >>> 
    >>> # Write to disk
    >>> write_fsq(fsq, 'output.fsq', include_index=True)
    
    # Reading an FSQ file
    >>> from fsq_sdk import read_fsq, get_frame, get_block
    >>> 
    >>> # Read file
    >>> fsq = read_fsq('output.fsq')
    >>> 
    >>> # Access frame and block
    >>> frame = get_frame(fsq, frame_id=0)
    >>> block = get_block(frame, block_index=0)
    >>> print(block.data.shape)  # (100, 100)
"""

__version__ = '0.1.0'
__author__ = 'FSQ SDK Contributors'

# Import main models
from .models import FSQFile, FSQFrame, FSQBlock

# Import encoder functions
from .encoder import (
    create_fsq_file,
    add_frame,
    add_block,
    write_fsq
)

# Import decoder functions
from .decoder import (
    read_fsq,
    get_frame,
    get_block,
    get_frame_by_id_fast,
    FSQMMapReader
)

# Import logging utilities
from .logging_config import (
    get_logger,
    set_log_level,
    enable_console_logging,
    disable_logging
)

# Define public API
__all__ = [
    # Version
    '__version__',
    
    # Models
    'FSQFile',
    'FSQFrame',
    'FSQBlock',
    
    # Encoder functions
    'create_fsq_file',
    'add_frame',
    'add_block',
    'write_fsq',
    
    # Decoder functions
    'read_fsq',
    'get_frame',
    'get_block',
    'get_frame_by_id_fast',
    'FSQMMapReader',
    
    # Logging utilities
    'get_logger',
    'set_log_level',
    'enable_console_logging',
    'disable_logging',
]
