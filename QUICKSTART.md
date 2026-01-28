# FSQ SDK - Quick Start Guide

## Installation

```bash
# Clone or download the package
cd BTP_i

# Install in development mode
pip install -e .

# Or install with dev dependencies for testing
pip install -e ".[dev]"
```

## Basic Usage

### 1. Create an FSQ File

```python
from fsq_sdk import create_fsq_file, add_frame, add_block, write_fsq
import numpy as np

# Create file
fsq = create_fsq_file(max_width=512, max_height=512)

# Add frame
frame = add_frame(fsq, frame_id=0)

# Add block with data
data = np.random.rand(100, 100).astype(np.float32)
block = add_block(frame, fsq, x=0, y=0, size_top=100, size_bottom=100, data=data)

# Write to disk
write_fsq(fsq, 'output.fsq', include_index=True)
```

### 2. Read an FSQ File

```python
from fsq_sdk import read_fsq, get_frame, get_block

# Read file
fsq = read_fsq('output.fsq')

# Access data
frame = get_frame(fsq, frame_id=0)
block = get_block(frame, block_index=0)

# Use the data
print(block.data.shape)  # (100, 100)
print(block.data.dtype)  # float32
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=fsq_sdk tests/

# Run with verbose output
pytest -v tests/
```

## Running Examples

```bash
# Run basic usage example
python examples/basic_usage.py
```

This will create an `example.fsq` file demonstrating:
- Creating frames with multiple blocks
- Writing with index
- Reading and accessing data
- Fast frame access

## File Format Specification

### Structure
```
[File Header: 64 bytes]
[Frame 0 Data]
  [Frame Header: 16 bytes]
  [Block 0]
    [Block Header: 16 bytes]
    [Block Data: S×T×4 bytes]
  [Block 1]
    ...
[Frame 1 Data]
  ...
[Frame Index: 12 bytes per frame] (optional)
```

### Constraints
- Max dimensions: 1024 × 1024
- Data type: float32 only (dtype=1)
- Frame IDs: Sequential (0, 1, 2, ...)
- Block bounds: x + size_top ≤ max_width, y + size_bottom ≤ max_height

## API Summary

### Encoding
- `create_fsq_file(max_width, max_height, dtype=1)` - Create new file
- `add_frame(file, frame_id)` - Add frame to file
- `add_block(frame, file, x, y, size_top, size_bottom, data)` - Add block to frame
- `write_fsq(file, filename, include_index=True)` - Write to disk

### Decoding
- `read_fsq(filename, use_index=True)` - Read from disk
- `get_frame(file, frame_id)` - Get frame by ID
- `get_block(frame, block_index)` - Get block by index
- `get_frame_by_id_fast(filename, frame_id)` - Fast single-frame read

## Troubleshooting

### Import Error
If you get `ModuleNotFoundError: No module named 'fsq_sdk'`:
```bash
pip install -e .
```

### NumPy Version
Requires NumPy >= 1.19.0:
```bash
pip install "numpy>=1.19.0"
```

### Test Failures
Make sure you're in the package root directory:
```bash
cd BTP_i
pytest tests/
```

## Next Steps

1. Read the full [README.md](README.md) for detailed documentation
2. Check out [examples/basic_usage.py](examples/basic_usage.py) for complete examples
3. Review [tests/test_fsq_sdk.py](tests/test_fsq_sdk.py) for usage patterns
4. Explore the source code in the `fsq_sdk/` directory

## Support

For issues or questions, please check:
- Source code documentation (docstrings)
- Test suite for usage examples
- README.md for detailed specifications
