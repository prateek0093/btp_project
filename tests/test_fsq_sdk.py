"""
Test suite for FSQ SDK.

This module tests encoding and decoding functionality with various scenarios.
"""

import pytest
import numpy as np
import os
import tempfile
from fsq_sdk import (
    create_fsq_file,
    add_frame,
    add_block,
    write_fsq,
    read_fsq,
    get_frame,
    get_block,
    get_frame_by_id_fast,
    FSQFile,
    FSQFrame,
    FSQBlock
)


class TestEncoding:
    """Test encoding functionality."""
    
    def test_create_fsq_file(self):
        """Test creating a new FSQ file."""
        fsq = create_fsq_file(max_width=512, max_height=512)
        assert fsq.magic == 'FSQ1'
        assert fsq.version == 1
        assert fsq.max_width == 512
        assert fsq.max_height == 512
        assert fsq.dtype == 1
        assert len(fsq.frames) == 0
    
    def test_create_fsq_file_invalid_dimensions(self):
        """Test that invalid dimensions raise errors."""
        with pytest.raises(ValueError):
            create_fsq_file(max_width=2000, max_height=512)
        
        with pytest.raises(ValueError):
            create_fsq_file(max_width=512, max_height=2000)
        
        with pytest.raises(ValueError):
            create_fsq_file(max_width=0, max_height=512)
    
    def test_add_frame(self):
        """Test adding frames to a file."""
        fsq = create_fsq_file(max_width=512, max_height=512)
        
        frame0 = add_frame(fsq, frame_id=0)
        assert frame0.frame_id == 0
        assert len(fsq.frames) == 1
        
        frame1 = add_frame(fsq, frame_id=1)
        assert frame1.frame_id == 1
        assert len(fsq.frames) == 2
    
    def test_add_frame_non_sequential(self):
        """Test that non-sequential frame IDs raise errors."""
        fsq = create_fsq_file(max_width=512, max_height=512)
        add_frame(fsq, frame_id=0)
        
        with pytest.raises(ValueError):
            add_frame(fsq, frame_id=5)
    
    def test_add_block(self):
        """Test adding blocks to a frame."""
        fsq = create_fsq_file(max_width=512, max_height=512)
        frame = add_frame(fsq, frame_id=0)
        
        data = np.random.rand(100, 100).astype(np.float32)
        block = add_block(frame, fsq, x=0, y=0, size_top=100, size_bottom=100, data=data)
        
        assert block.x == 0
        assert block.y == 0
        assert block.size_top == 100
        assert block.size_bottom == 100
        assert block.data.shape == (100, 100)
        assert np.array_equal(block.data, data)
        assert len(frame.blocks) == 1
    
    def test_add_block_exceeds_bounds(self):
        """Test that blocks exceeding bounds raise errors."""
        fsq = create_fsq_file(max_width=512, max_height=512)
        frame = add_frame(fsq, frame_id=0)
        
        data = np.random.rand(200, 200).astype(np.float32)
        
        # Block would exceed width
        with pytest.raises(ValueError):
            add_block(frame, fsq, x=400, y=0, size_top=200, size_bottom=200, data=data)
        
        # Block would exceed height
        with pytest.raises(ValueError):
            add_block(frame, fsq, x=0, y=400, size_top=200, size_bottom=200, data=data)
    
    def test_add_block_wrong_data_shape(self):
        """Test that wrong data shape raises error."""
        fsq = create_fsq_file(max_width=512, max_height=512)
        frame = add_frame(fsq, frame_id=0)
        
        data = np.random.rand(50, 50).astype(np.float32)
        
        with pytest.raises(ValueError):
            add_block(frame, fsq, x=0, y=0, size_top=100, size_bottom=100, data=data)


class TestRoundTrip:
    """Test complete encoding and decoding cycle."""
    
    def test_simple_round_trip(self):
        """Test encoding and decoding a simple file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test.fsq')
            
            # Create file
            fsq = create_fsq_file(max_width=256, max_height=256)
            frame = add_frame(fsq, frame_id=0)
            
            data = np.arange(100 * 100, dtype=np.float32).reshape(100, 100)
            add_block(frame, fsq, x=0, y=0, size_top=100, size_bottom=100, data=data)
            
            # Write to disk
            write_fsq(fsq, filepath, include_index=True)
            
            # Read back
            fsq_read = read_fsq(filepath)
            
            # Verify
            assert fsq_read.max_width == 256
            assert fsq_read.max_height == 256
            assert fsq_read.total_frames == 1
            assert len(fsq_read.frames) == 1
            
            frame_read = get_frame(fsq_read, frame_id=0)
            assert frame_read.num_blocks == 1
            
            block_read = get_block(frame_read, block_index=0)
            assert block_read.x == 0
            assert block_read.y == 0
            assert block_read.size_top == 100
            assert block_read.size_bottom == 100
            assert np.array_equal(block_read.data, data)
    
    def test_multiple_frames_and_blocks(self):
        """Test file with 2 frames, each with 2 blocks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_multi.fsq')
            
            # Create file
            fsq = create_fsq_file(max_width=512, max_height=512)
            
            # Frame 0 with 2 blocks
            frame0 = add_frame(fsq, frame_id=0)
            data0_0 = np.ones((100, 100), dtype=np.float32) * 1.0
            add_block(frame0, fsq, x=0, y=0, size_top=100, size_bottom=100, data=data0_0)
            
            data0_1 = np.ones((50, 75), dtype=np.float32) * 2.0
            add_block(frame0, fsq, x=100, y=0, size_top=50, size_bottom=75, data=data0_1)
            
            # Frame 1 with 2 blocks
            frame1 = add_frame(fsq, frame_id=1)
            data1_0 = np.ones((80, 90), dtype=np.float32) * 3.0
            add_block(frame1, fsq, x=0, y=100, size_top=80, size_bottom=90, data=data1_0)
            
            data1_1 = np.ones((120, 130), dtype=np.float32) * 4.0
            add_block(frame1, fsq, x=200, y=200, size_top=120, size_bottom=130, data=data1_1)
            
            # Write to disk
            write_fsq(fsq, filepath, include_index=True)
            
            # Read back
            fsq_read = read_fsq(filepath)
            
            # Verify structure
            assert fsq_read.total_frames == 2
            assert len(fsq_read.frames) == 2
            
            # Verify frame 0
            frame0_read = get_frame(fsq_read, frame_id=0)
            assert frame0_read.num_blocks == 2
            
            block0_0 = get_block(frame0_read, block_index=0)
            assert np.array_equal(block0_0.data, data0_0)
            
            block0_1 = get_block(frame0_read, block_index=1)
            assert np.array_equal(block0_1.data, data0_1)
            
            # Verify frame 1
            frame1_read = get_frame(fsq_read, frame_id=1)
            assert frame1_read.num_blocks == 2
            
            block1_0 = get_block(frame1_read, block_index=0)
            assert np.array_equal(block1_0.data, data1_0)
            
            block1_1 = get_block(frame1_read, block_index=1)
            assert np.array_equal(block1_1.data, data1_1)
    
    def test_round_trip_without_index(self):
        """Test encoding and decoding without frame index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_no_index.fsq')
            
            # Create file
            fsq = create_fsq_file(max_width=256, max_height=256)
            frame = add_frame(fsq, frame_id=0)
            
            data = np.random.rand(50, 60).astype(np.float32)
            add_block(frame, fsq, x=0, y=0, size_top=50, size_bottom=60, data=data)
            
            # Write without index
            write_fsq(fsq, filepath, include_index=False)
            
            # Read back
            fsq_read = read_fsq(filepath, use_index=False)
            
            # Verify
            assert fsq_read.index_offset == 0
            assert fsq_read.total_frames == 1
            
            frame_read = get_frame(fsq_read, frame_id=0)
            block_read = get_block(frame_read, block_index=0)
            assert np.array_equal(block_read.data, data)
    
    def test_fast_frame_access(self):
        """Test fast frame access using index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_fast.fsq')
            
            # Create file with multiple frames
            fsq = create_fsq_file(max_width=256, max_height=256)
            
            for i in range(5):
                frame = add_frame(fsq, frame_id=i)
                data = np.ones((50, 50), dtype=np.float32) * float(i)
                add_block(frame, fsq, x=0, y=0, size_top=50, size_bottom=50, data=data)
            
            # Write with index
            write_fsq(fsq, filepath, include_index=True)
            
            # Fast access to specific frame
            frame = get_frame_by_id_fast(filepath, frame_id=3)
            assert frame.frame_id == 3
            assert frame.num_blocks == 1
            
            block = get_block(frame, block_index=0)
            assert np.all(block.data == 3.0)


class TestValidation:
    """Test validation and error handling."""
    
    def test_invalid_magic(self):
        """Test that invalid magic raises error."""
        with pytest.raises(ValueError, match="Invalid magic"):
            FSQFile(magic='ABCD', max_width=512, max_height=512)
    
    def test_invalid_version(self):
        """Test that invalid version raises error."""
        with pytest.raises(ValueError, match="Invalid version"):
            FSQFile(version=2, max_width=512, max_height=512)
    
    def test_block_data_validation(self):
        """Test block data validation."""
        # Wrong shape
        with pytest.raises(ValueError):
            FSQBlock(
                x=0, y=0, size_top=10, size_bottom=10,
                data_bytes=400,
                data=np.zeros((5, 5), dtype=np.float32)
            )
        
        # Wrong data type
        with pytest.raises(ValueError):
            FSQBlock(
                x=0, y=0, size_top=10, size_bottom=10,
                data_bytes=400,
                data=np.zeros((10, 10), dtype=np.float64)
            )
        
        # Wrong data_bytes
        with pytest.raises(ValueError):
            FSQBlock(
                x=0, y=0, size_top=10, size_bottom=10,
                data_bytes=100,  # Should be 400
                data=np.zeros((10, 10), dtype=np.float32)
            )
    
    def test_read_nonexistent_file(self):
        """Test reading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            read_fsq('nonexistent.fsq')
    
    def test_get_invalid_frame(self):
        """Test getting invalid frame raises error."""
        fsq = create_fsq_file(max_width=256, max_height=256)
        frame = add_frame(fsq, frame_id=0)
        
        with pytest.raises(ValueError):
            get_frame(fsq, frame_id=5)
    
    def test_get_invalid_block(self):
        """Test getting invalid block raises error."""
        fsq = create_fsq_file(max_width=256, max_height=256)
        frame = add_frame(fsq, frame_id=0)
        data = np.zeros((10, 10), dtype=np.float32)
        add_block(frame, fsq, x=0, y=0, size_top=10, size_bottom=10, data=data)
        
        with pytest.raises(ValueError):
            get_block(frame, block_index=5)


class TestDataTypes:
    """Test handling of different data types and conversions."""
    
    def test_list_input(self):
        """Test that list input is converted to numpy array."""
        fsq = create_fsq_file(max_width=256, max_height=256)
        frame = add_frame(fsq, frame_id=0)
        
        # Use list of lists
        data_list = [[1.0, 2.0], [3.0, 4.0]]
        block = add_block(frame, fsq, x=0, y=0, size_top=2, size_bottom=2, data=data_list)
        
        assert isinstance(block.data, np.ndarray)
        assert block.data.dtype == np.float32
        assert block.data.shape == (2, 2)
    
    def test_float64_conversion(self):
        """Test that float64 is converted to float32."""
        fsq = create_fsq_file(max_width=256, max_height=256)
        frame = add_frame(fsq, frame_id=0)
        
        data_float64 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        block = add_block(frame, fsq, x=0, y=0, size_top=2, size_bottom=2, data=data_float64)
        
        assert block.data.dtype == np.float32


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_element_block(self):
        """Test block with single element."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_single.fsq')
            
            fsq = create_fsq_file(max_width=256, max_height=256)
            frame = add_frame(fsq, frame_id=0)
            
            data = np.array([[42.0]], dtype=np.float32)
            add_block(frame, fsq, x=0, y=0, size_top=1, size_bottom=1, data=data)
            
            write_fsq(fsq, filepath)
            fsq_read = read_fsq(filepath)
            
            block = get_block(get_frame(fsq_read, 0), 0)
            assert block.data[0, 0] == 42.0
    
    def test_max_dimensions(self):
        """Test using maximum allowed dimensions."""
        fsq = create_fsq_file(max_width=1024, max_height=1024)
        frame = add_frame(fsq, frame_id=0)
        
        # Add block at maximum position
        data = np.ones((1, 1), dtype=np.float32)
        block = add_block(frame, fsq, x=1023, y=1023, size_top=1, size_bottom=1, data=data)
        
        assert block.x == 1023
        assert block.y == 1023
    
    def test_empty_file(self):
        """Test file with no frames."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_empty.fsq')
            
            fsq = create_fsq_file(max_width=256, max_height=256)
            write_fsq(fsq, filepath)
            
            fsq_read = read_fsq(filepath)
            assert fsq_read.total_frames == 0
            assert len(fsq_read.frames) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
