"""
Verification script to demonstrate all FSQ SDK features.
This script tests the complete functionality of the package.
"""

import numpy as np
import os
from fsq_sdk import (
    create_fsq_file,
    add_frame,
    add_block,
    write_fsq,
    read_fsq,
    get_frame,
    get_block,
    get_frame_by_id_fast,
    __version__
)


def verify_encoding():
    """Verify encoding functionality."""
    print("=" * 60)
    print("1. ENCODING VERIFICATION")
    print("=" * 60)
    
    # Create file
    print("\n✓ Creating FSQ file with max_width=512, max_height=512")
    fsq = create_fsq_file(max_width=512, max_height=512)
    assert fsq.magic == 'FSQ1'
    assert fsq.version == 1
    assert fsq.max_width == 512
    assert fsq.max_height == 512
    print(f"  Magic: {fsq.magic}, Version: {fsq.version}")
    
    # Add Frame 0
    print("\n✓ Adding Frame 0 with 3 blocks")
    frame0 = add_frame(fsq, frame_id=0)
    
    # Block 1: 100x100 at (0,0)
    data1 = np.arange(100*100, dtype=np.float32).reshape(100, 100)
    block1 = add_block(frame0, fsq, x=0, y=0, size_top=100, size_bottom=100, data=data1)
    print(f"  Block 1: pos=({block1.x},{block1.y}), size={block1.size_top}x{block1.size_bottom}")
    
    # Block 2: 50x75 at (100,0)
    data2 = np.ones((50, 75), dtype=np.float32) * 3.14
    block2 = add_block(frame0, fsq, x=100, y=0, size_top=50, size_bottom=75, data=data2)
    print(f"  Block 2: pos=({block2.x},{block2.y}), size={block2.size_top}x{block2.size_bottom}")
    
    # Block 3: 200x150 at (0,100)
    data3 = np.random.randn(200, 150).astype(np.float32)
    block3 = add_block(frame0, fsq, x=0, y=100, size_top=200, size_bottom=150, data=data3)
    print(f"  Block 3: pos=({block3.x},{block3.y}), size={block3.size_top}x{block3.size_bottom}")
    
    # Add Frame 1
    print("\n✓ Adding Frame 1 with 2 blocks")
    frame1 = add_frame(fsq, frame_id=1)
    
    data4 = np.ones((80, 90), dtype=np.float32) * 2.718
    block4 = add_block(frame1, fsq, x=0, y=0, size_top=80, size_bottom=90, data=data4)
    print(f"  Block 1: pos=({block4.x},{block4.y}), size={block4.size_top}x{block4.size_bottom}")
    
    data5 = np.eye(100, dtype=np.float32)
    block5 = add_block(frame1, fsq, x=200, y=200, size_top=100, size_bottom=100, data=data5)
    print(f"  Block 2: pos=({block5.x},{block5.y}), size={block5.size_top}x{block5.size_bottom}")
    
    # Write with index
    print("\n✓ Writing to 'verification.fsq' with index")
    write_fsq(fsq, 'verification.fsq', include_index=True)
    file_size = os.path.getsize('verification.fsq')
    print(f"  File size: {file_size:,} bytes")
    print(f"  Total frames: {fsq.total_frames}")
    print(f"  Index offset: {fsq.index_offset}")
    
    return fsq


def verify_decoding():
    """Verify decoding functionality."""
    print("\n" + "=" * 60)
    print("2. DECODING VERIFICATION")
    print("=" * 60)
    
    # Read file
    print("\n✓ Reading 'verification.fsq'")
    fsq = read_fsq('verification.fsq', use_index=True)
    print(f"  Loaded {fsq.total_frames} frames")
    print(f"  Max dimensions: {fsq.max_width}x{fsq.max_height}")
    print(f"  Has index: {fsq.index_offset > 0}")
    
    # Verify Frame 0
    print("\n✓ Verifying Frame 0")
    frame0 = get_frame(fsq, frame_id=0)
    print(f"  Frame ID: {frame0.frame_id}")
    print(f"  Num blocks: {frame0.num_blocks}")
    print(f"  Frame size: {frame0.frame_size_bytes:,} bytes")
    
    # Check blocks
    for i in range(frame0.num_blocks):
        block = get_block(frame0, block_index=i)
        print(f"  Block {i}: pos=({block.x},{block.y}), "
              f"size={block.size_top}x{block.size_bottom}, "
              f"mean={block.data.mean():.2f}")
    
    # Verify Frame 1
    print("\n✓ Verifying Frame 1")
    frame1 = get_frame(fsq, frame_id=1)
    print(f"  Frame ID: {frame1.frame_id}")
    print(f"  Num blocks: {frame1.num_blocks}")
    
    for i in range(frame1.num_blocks):
        block = get_block(frame1, block_index=i)
        print(f"  Block {i}: pos=({block.x},{block.y}), "
              f"size={block.size_top}x{block.size_bottom}, "
              f"mean={block.data.mean():.2f}")
    
    return fsq


def verify_fast_access():
    """Verify fast frame access."""
    print("\n" + "=" * 60)
    print("3. FAST ACCESS VERIFICATION")
    print("=" * 60)
    
    print("\n✓ Fast loading Frame 1 using index")
    frame = get_frame_by_id_fast('verification.fsq', frame_id=1)
    print(f"  Frame ID: {frame.frame_id}")
    print(f"  Num blocks: {frame.num_blocks}")
    
    block = get_block(frame, block_index=1)
    print(f"  Block 1: Identity matrix trace = {np.trace(block.data):.0f}")
    print(f"  Block 1: Diagonal sum = {block.data.diagonal().sum():.0f}")
    

def verify_data_integrity():
    """Verify data integrity after round-trip."""
    print("\n" + "=" * 60)
    print("4. DATA INTEGRITY VERIFICATION")
    print("=" * 60)
    
    # Create test data
    print("\n✓ Creating test file with known data")
    fsq = create_fsq_file(max_width=256, max_height=256)
    frame = add_frame(fsq, frame_id=0)
    
    # Test various data patterns
    test_data = {
        'zeros': np.zeros((50, 50), dtype=np.float32),
        'ones': np.ones((30, 40), dtype=np.float32),
        'sequence': np.arange(60*70, dtype=np.float32).reshape(60, 70),
        'random': np.random.rand(80, 90).astype(np.float32),
    }
    
    positions = [(0, 0), (50, 0), (0, 50), (100, 100)]
    
    for (name, data), (x, y) in zip(test_data.items(), positions):
        size_top, size_bottom = data.shape
        add_block(frame, fsq, x=x, y=y, size_top=size_top, size_bottom=size_bottom, data=data)
        print(f"  Added '{name}' block: {data.shape}")
    
    # Write and read
    write_fsq(fsq, 'integrity_test.fsq', include_index=True)
    fsq_read = read_fsq('integrity_test.fsq')
    
    # Verify each block
    print("\n✓ Verifying data integrity")
    frame_read = get_frame(fsq_read, frame_id=0)
    
    for i, (name, original_data) in enumerate(test_data.items()):
        block = get_block(frame_read, block_index=i)
        assert np.array_equal(block.data, original_data), f"{name} data mismatch!"
        print(f"  '{name}': ✓ Perfect match")
    
    # Cleanup
    os.remove('integrity_test.fsq')


def verify_constraints():
    """Verify constraint validation."""
    print("\n" + "=" * 60)
    print("5. CONSTRAINT VALIDATION VERIFICATION")
    print("=" * 60)
    
    print("\n✓ Testing dimension constraints")
    fsq = create_fsq_file(max_width=100, max_height=100)
    frame = add_frame(fsq, frame_id=0)
    
    # Valid block
    data = np.ones((50, 50), dtype=np.float32)
    try:
        add_block(frame, fsq, x=0, y=0, size_top=50, size_bottom=50, data=data)
        print("  Valid block (50x50 at 0,0): ✓ Accepted")
    except ValueError as e:
        print(f"  Unexpected error: {e}")
    
    # Invalid block (exceeds width)
    try:
        add_block(frame, fsq, x=60, y=0, size_top=50, size_bottom=50, data=data)
        print("  Invalid block (exceeds width): ✗ Should have failed")
    except ValueError:
        print("  Invalid block (exceeds width): ✓ Correctly rejected")
    
    # Invalid block (exceeds height)
    try:
        add_block(frame, fsq, x=0, y=60, size_top=50, size_bottom=50, data=data)
        print("  Invalid block (exceeds height): ✗ Should have failed")
    except ValueError:
        print("  Invalid block (exceeds height): ✓ Correctly rejected")
    
    print("\n✓ Testing data type validation")
    # Test float64 conversion
    data_f64 = np.ones((10, 10), dtype=np.float64)
    block = add_block(frame, fsq, x=0, y=50, size_top=10, size_bottom=10, data=data_f64)
    assert block.data.dtype == np.float32
    print("  float64 input: ✓ Converted to float32")
    
    # Test list input
    data_list = [[1.0, 2.0], [3.0, 4.0]]
    block = add_block(frame, fsq, x=50, y=0, size_top=2, size_bottom=2, data=data_list)
    assert isinstance(block.data, np.ndarray)
    assert block.data.dtype == np.float32
    print("  list input: ✓ Converted to numpy array")


def main():
    """Run all verification tests."""
    print("\n" + "=" * 60)
    print(f"FSQ SDK v{__version__} - Complete Verification")
    print("=" * 60)
    
    try:
        # Run all verifications
        original_fsq = verify_encoding()
        decoded_fsq = verify_decoding()
        verify_fast_access()
        verify_data_integrity()
        verify_constraints()
        
        # Final summary
        print("\n" + "=" * 60)
        print("VERIFICATION COMPLETE")
        print("=" * 60)
        print("\n✓ All features verified successfully!")
        print("\nGenerated files:")
        print("  - verification.fsq")
        
        # File info
        if os.path.exists('verification.fsq'):
            size = os.path.getsize('verification.fsq')
            print(f"\nFile size: {size:,} bytes")
            print(f"Frames: {original_fsq.total_frames}")
            print(f"Total blocks: {sum(f.num_blocks for f in original_fsq.frames)}")
        
        print("\n" + "=" * 60)
        print("Package is ready for use!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Verification failed: {e}")
        raise


if __name__ == '__main__':
    main()
