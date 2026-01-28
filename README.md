# FSQ SDK

Python SDK for encoding and decoding data in the FSQ (version 1) binary file format.

## Overview

FSQ is a custom binary file format designed for efficient storage of frame-based data with rectangular blocks of float32 values. This SDK provides a simple and efficient Python interface for creating, writing, reading, and manipulating FSQ files.

## Features

- **Efficient Binary Format**: Sequential writes and reads for optimal I/O performance
- **Frame-based Organization**: Store data as frames containing multiple rectangular blocks
- **Optional Frame Index**: Fast seeking to specific frames using built-in index
- **NumPy Integration**: Direct array access for GPU/NumPy workflows
- **Type Safety**: Full validation of constraints and data types
- **Simple API**: Easy-to-use functions for encoding and decoding

## Installation

```bash
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Quick Start

### Encoding (Creating FSQ Files)

```python
from fsq_sdk import create_fsq_file, add_frame, add_block, write_fsq
import numpy as np

# Create a new FSQ file with maximum dimensions
fsq = create_fsq_file(max_width=512, max_height=512)

# Add a frame (frame IDs must be sequential starting from 0)
frame = add_frame(fsq, frame_id=0)

# Create block data (100x100 rectangle of random float32 values)
data = np.random.rand(100, 100).astype(np.float32)

# Add block to frame at position (0, 0)
block = add_block(frame, fsq, x=0, y=0, size_top=100, size_bottom=100, data=data)

# Add another block to the same frame
data2 = np.ones((50, 75), dtype=np.float32)
block2 = add_block(frame, fsq, x=100, y=0, size_top=50, size_bottom=75, data=data2)

# Write to disk with frame index
write_fsq(fsq, 'output.fsq', include_index=True)
```

### Decoding (Reading FSQ Files)

```python
from fsq_sdk import read_fsq, get_frame, get_block

# Read entire file
fsq = read_fsq('output.fsq')

# Access file metadata
print(f"Max dimensions: {fsq.max_width} x {fsq.max_height}")
print(f"Total frames: {fsq.total_frames}")

# Get specific frame
frame = get_frame(fsq, frame_id=0)
print(f"Frame has {frame.num_blocks} blocks")

# Get specific block
block = get_block(frame, block_index=0)
print(f"Block position: ({block.x}, {block.y})")
print(f"Block size: {block.size_top} x {block.size_bottom}")
print(f"Block data shape: {block.data.shape}")

# Access block data as NumPy array
data_array = block.data  # shape (100, 100), dtype float32
```

### Fast Frame Access

For files with an index, you can quickly access individual frames without loading the entire file:

```python
from fsq_sdk import get_frame_by_id_fast

# Efficiently read just one frame
frame = get_frame_by_id_fast('output.fsq', frame_id=5)
```

## API Reference

### Data Models

- **FSQFile**: Represents a complete FSQ file with header and frames
- **FSQFrame**: Represents a single frame containing blocks
- **FSQBlock**: Represents a rectangular block of float32 data

### Encoding Functions

- `create_fsq_file(max_width, max_height, dtype=1)`: Create new FSQ file
- `add_frame(file, frame_id)`: Add frame to file
- `add_block(frame, file, x, y, size_top, size_bottom, data)`: Add block to frame
- `write_fsq(file, filename, include_index=True)`: Write file to disk

### Decoding Functions

- `read_fsq(filename, use_index=True)`: Read FSQ file from disk
- `get_frame(file, frame_id)`: Get frame by ID
- `get_block(frame, block_index)`: Get block by index
- `get_frame_by_id_fast(filename, frame_id)`: Fast single-frame retrieval

## File Format Specification

### Constraints

- Maximum frame dimensions: 1024 x 1024
- Data type: float32 (dtype=1)
- Frame IDs must be sequential (0, 1, 2, ...)
- Blocks must fit within frame dimensions: `x + size_top <= max_width`, `y + size_bottom <= max_height`
- Block data size: `size_top * size_bottom * 4` bytes

### File Structure

1. **File Header** (64 bytes)
   - Magic: "FSQ1" (4 bytes)
   - Version: 1 (uint16)
   - Header size: 64 (uint16)
   - Max width, max height (uint16 each)
   - Total frames (uint32)
   - Data type (uint16)
   - Index offset (uint64)
   - Reserved fields

2. **Frames** (variable size)
   - Frame Header (16 bytes): frame_id, num_blocks, frame_size_bytes
   - Blocks (variable size each):
     - Block Header (16 bytes): x, y, size_top, size_bottom, data_bytes
     - Block Data: float32 values in row-major order

3. **Frame Index** (optional, 12 bytes per frame)
   - frame_id (uint32) + offset (uint64)

## Testing

Run the test suite:

```bash
pytest tests/
```

Run with coverage:

```bash
pytest --cov=fsq_sdk tests/
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.
