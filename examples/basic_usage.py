"""
Example usage of FSQ SDK.

This script demonstrates how to create, write, and read FSQ files.
"""

import numpy as np
from fsq_sdk import (
    create_fsq_file,
    add_frame,
    add_block,
    write_fsq,
    read_fsq,
    get_frame,
    get_block,
    get_frame_by_id_fast
)


def example_encode():
    """Example: Create and encode an FSQ file."""
    print("Creating FSQ file...")
    
    # Create a new FSQ file with max dimensions
    fsq = create_fsq_file(max_width=512, max_height=512)
    
    # Add Frame 0 with 2 blocks
    print("Adding Frame 0...")
    frame0 = add_frame(fsq, frame_id=0)
    
    # Block 1: 100x100 gradient
    data1 = np.linspace(0, 100, 100*100).reshape(100, 100).astype(np.float32)
    block1 = add_block(frame0, fsq, x=0, y=0, size_top=100, size_bottom=100, data=data1)
    print(f"  Added block at ({block1.x}, {block1.y}), size {block1.size_top}x{block1.size_bottom}")
    
    # Block 2: 50x75 ones
    data2 = np.ones((50, 75), dtype=np.float32) * 42.0
    block2 = add_block(frame0, fsq, x=100, y=0, size_top=50, size_bottom=75, data=data2)
    print(f"  Added block at ({block2.x}, {block2.y}), size {block2.size_top}x{block2.size_bottom}")
    
    # Add Frame 1 with 1 block
    print("Adding Frame 1...")
    frame1 = add_frame(fsq, frame_id=1)
    
    # Block: 200x150 random values
    data3 = np.random.randn(200, 150).astype(np.float32)
    block3 = add_block(frame1, fsq, x=0, y=100, size_top=200, size_bottom=150, data=data3)
    print(f"  Added block at ({block3.x}, {block3.y}), size {block3.size_top}x{block3.size_bottom}")
    
    # Write to disk with index
    print("\nWriting to 'example.fsq'...")
    write_fsq(fsq, 'example.fsq', include_index=True)
    print(f"File written successfully! Total frames: {fsq.total_frames}")
    
    return fsq


def example_decode():
    """Example: Read and decode an FSQ file."""
    print("\n" + "="*50)
    print("Reading FSQ file...")
    
    # Read the file
    fsq = read_fsq('example.fsq')
    
    print(f"\nFile metadata:")
    print(f"  Magic: {fsq.magic}")
    print(f"  Version: {fsq.version}")
    print(f"  Max dimensions: {fsq.max_width} x {fsq.max_height}")
    print(f"  Total frames: {fsq.total_frames}")
    print(f"  Has index: {'Yes' if fsq.index_offset > 0 else 'No'}")
    
    # Access Frame 0
    print(f"\nFrame 0:")
    frame0 = get_frame(fsq, frame_id=0)
    print(f"  Number of blocks: {frame0.num_blocks}")
    print(f"  Frame size: {frame0.frame_size_bytes} bytes")
    
    for i in range(frame0.num_blocks):
        block = get_block(frame0, block_index=i)
        print(f"  Block {i}: pos=({block.x}, {block.y}), "
              f"size={block.size_top}x{block.size_bottom}, "
              f"data_shape={block.data.shape}, "
              f"mean={block.data.mean():.2f}")
    
    # Access Frame 1
    print(f"\nFrame 1:")
    frame1 = get_frame(fsq, frame_id=1)
    print(f"  Number of blocks: {frame1.num_blocks}")
    
    for i in range(frame1.num_blocks):
        block = get_block(frame1, block_index=i)
        print(f"  Block {i}: pos=({block.x}, {block.y}), "
              f"size={block.size_top}x{block.size_bottom}, "
              f"data_shape={block.data.shape}, "
              f"mean={block.data.mean():.2f}, std={block.data.std():.2f}")


def example_fast_access():
    """Example: Fast access to specific frame using index."""
    print("\n" + "="*50)
    print("Fast frame access example...")
    
    # Quickly read just Frame 1 without loading entire file
    frame = get_frame_by_id_fast('example.fsq', frame_id=1)
    
    print(f"\nFast-loaded Frame {frame.frame_id}:")
    print(f"  Number of blocks: {frame.num_blocks}")
    
    block = get_block(frame, block_index=0)
    print(f"  Block data shape: {block.data.shape}")
    print(f"  Block data range: [{block.data.min():.2f}, {block.data.max():.2f}]")


def main():
    """Run all examples."""
    print("FSQ SDK Example\n")
    
    # Encoding example
    original_fsq = example_encode()
    
    # Decoding example
    example_decode()
    
    # Fast access example
    example_fast_access()
    
    print("\n" + "="*50)
    print("Example completed successfully!")
    print("\nGenerated file: example.fsq")


if __name__ == '__main__':
    main()
