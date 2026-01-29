import sif_parser
import numpy as np
import cv2
import math
import os
import time
from scipy.ndimage import label, center_of_mass, sum as ndi_sum

# ==========================================
# 1. CONFIGURATION
# ==========================================
FILE_PATH = "your_camera_capture.sif"

# Detection Parameters
FIXED_THRESHOLD = 1600
FIXED_MIN_SIZE = 6
FIXED_PADDING = 6

# Optimization Settings
FIT_INTERVAL = 5   # Recalculate background every N frames
FIT_RESOLUTION = (512, 512) # Downscale size for the curve fitting math

# Tracking
DECAY_FRAMES = 1

# ==========================================
# 2. PROCESSOR CLASS
# ==========================================
class SIFProcessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
        self.height = 0
        self.width = 0
        self.num_frames = 0
        
        # Optimization Cache
        self.A_small = None   # Design matrix for 64x64 (Quartic)
        self.cached_bg = None # The upscaled, full-res background
        self.last_fit_idx = -9999
        
        self.load_data()
        self.precompute_small_design_matrix()

    def load_data(self):
        print("Loading SIF data...")
        if os.path.exists(self.filepath):
            self.data, info = sif_parser.np_open(self.filepath)
        else:
            print("Using DUMMY data for simulation.")
            self.data = np.zeros((300, 512, 512), dtype=np.float32)
            Y, X = np.mgrid[:512, :512]
            for i in range(300):
                # Slowly changing Quartic background
                # (x^4 term effect is small but present)
                factor = 0.000000001 * (1 + (i * 0.01))
                bg = factor * ((X-256)**4 + (Y-256)**4) 
                # Add some quadratic term too so it's not flat
                bg += 0.00005 * ((X-256)**2 + (Y-256)**2)

                self.data[i] += bg + np.random.normal(10, 2, (512, 512))
                
                # Object
                r, c = 200 + i, 200 + i
                if r < 480 and c < 480:
                    self.data[i, r:r+30, c:c+30] += 2500

        self.num_frames, self.height, self.width = self.data.shape

    def precompute_small_design_matrix(self):
        """
        Pre-calculates QUARTIC (Deg 4) design matrix for the SMALL resolution (64x64).
        """
        w_small, h_small = FIT_RESOLUTION
        print(f"Pre-computing QUARTIC model for {w_small}x{h_small}...")
        
        X, Y = np.meshgrid(np.arange(w_small), np.arange(h_small))
        x_flat = (X.flatten() - w_small/2) / (w_small/2)
        y_flat = (Y.flatten() - h_small/2) / (h_small/2)
        
        # Quartic Terms (Degree 4)
        ones = np.ones(x_flat.shape)
        x, y = x_flat, y_flat
        x2, y2, xy = x**2, y**2, x*y
        x3, y3 = x**3, y**3
        x2y, xy2 = (x**2)*y, x*(y**2)
        x4, y4 = x**4, y**4
        x3y, xy3 = (x**3)*y, x*(y**3)
        x2y2 = (x**2)*(y**2)
        
        self.A_small = np.c_[ones, x, y, x2, xy, y2, x3, x2y, xy2, y3, x4, x3y, x2y2, xy3, y4]

    def update_background_cache(self, frame_data):
        # 1. Resize to small (cv2 uses (width, height))
        small_frame = cv2.resize(frame_data, FIT_RESOLUTION, interpolation=cv2.INTER_AREA)
        
        # 2. Fit (Math is fast here because matrix is small)
        C, _, _, _ = np.linalg.lstsq(self.A_small, small_frame.flatten(), rcond=None)
        
        # 3. Generate Small Surface
        # numpy reshape uses (height, width)
        bg_small = np.dot(self.A_small, C).reshape(FIT_RESOLUTION[1], FIT_RESOLUTION[0])
        
        # 4. Upscale to Full Resolution
        self.cached_bg = cv2.resize(bg_small, (self.width, self.height), interpolation=cv2.INTER_CUBIC)

    def process_frame(self, frame_idx):
        raw_frame = self.data[frame_idx].astype(np.float32)
        
        # CHECK INTERVAL: Should we recalculate the curve?
        if self.cached_bg is None or (frame_idx - self.last_fit_idx >= FIT_INTERVAL):
            # print(f"[System] Recalculating Background Curve at Frame {frame_idx}")
            self.update_background_cache(raw_frame)
            self.last_fit_idx = frame_idx
            
        # Subtract cached background
        diff = raw_frame - self.cached_bg
        
        # Grouping Logic
        mask = diff > FIXED_THRESHOLD 
        structure = np.array([[0,1,0], [1,1,1], [0,1,0]]) 
        labeled_array, num_features = label(mask, structure=structure)
        
        raw_boxes = []
        if num_features > 0:
            indices = np.arange(1, num_features + 1)
            sizes = ndi_sum(mask, labeled_array, index=indices)
            centers = center_of_mass(mask, labeled_array, index=indices)
            if num_features == 1: sizes, centers = [sizes], [centers]

            for size, center in zip(sizes, centers):
                if size < FIXED_MIN_SIZE: continue
                side = math.sqrt(size) + FIXED_PADDING
                cy, cx = center
                raw_boxes.append([cy - side/2, cx - side/2, cy + side/2, cx + side/2])
        
        return raw_frame, diff, raw_boxes

# ==========================================
# 3. GLOBAL TRACKER
# ==========================================
class GlobalTracker:
    def __init__(self, decay_frames):
        self.decay_limit = decay_frames
        self.tracks = [] 
        self.next_id = 1

    def check_overlap(self, box1, box2):
        if box1[3] < box2[1] or box2[3] < box1[1]: return False 
        if box1[2] < box2[0] or box2[2] < box1[0]: return False 
        return True

    def union_boxes(self, box1, box2):
        return [min(box1[0], box2[0]), min(box1[1], box2[1]), max(box1[2], box2[2]), max(box1[3], box2[3])]

    def update(self, new_raw_boxes):
        for t in self.tracks: t['life'] -= 1

        for raw in new_raw_boxes:
            matched = False
            for t in self.tracks:
                if self.check_overlap(raw, t['coords']):
                    t['coords'] = self.union_boxes(raw, t['coords'])
                    t['life'] = self.decay_limit 
                    matched = True
                    break 
            
            if not matched:
                self.tracks.append({'coords': raw, 'life': self.decay_limit, 'id': self.next_id})
                self.next_id += 1

        self.tracks = [t for t in self.tracks if t['life'] > 0]
        self.consolidate_tracks()
        return self.tracks

    def consolidate_tracks(self):
        if not self.tracks: return
        i = 0
        while i < len(self.tracks):
            j = i + 1
            while j < len(self.tracks):
                t1, t2 = self.tracks[i], self.tracks[j]
                if self.check_overlap(t1['coords'], t2['coords']):
                    t1['coords'] = self.union_boxes(t1['coords'], t2['coords'])
                    t1['life'] = max(t1['life'], t2['life'])
                    self.tracks.pop(j)
                else:
                    j += 1
            i += 1

# ==========================================
# 4. MAIN VIEWER
# ==========================================
def main():
    proc = SIFProcessor(FILE_PATH)
    tracker = GlobalTracker(decay_frames=DECAY_FRAMES)
    
    frame_idx = 0
    paused = False
    
    print("\n--- CONTROLS ---")
    print(f"Algorithm: Quartic (Power 4) Fit")
    print(f"Interval: Re-fit every {FIT_INTERVAL} frames")
    print("SPACE : Pause / Play")
    print("Q     : Quit")
    print("----------------")

    while True:
        if not paused:
            # A. Process
            raw, diff, raw_boxes = proc.process_frame(frame_idx)
            
            # B. Track
            global_tracks = tracker.update(raw_boxes)
            
            # C. Render
            raw_disp = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            raw_color = cv2.cvtColor(raw_disp, cv2.COLOR_GRAY2BGR)
            
            diff_disp = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            diff_color = cv2.applyColorMap(diff_disp, cv2.COLORMAP_INFERNO)

            # Draw Tracks
            for t in global_tracks:
                y1, x1, y2, x2 = map(int, t['coords'])
                intensity = int(255 * (t['life'] / DECAY_FRAMES))
                color = (intensity, 255, 255 - intensity)
                
                cv2.rectangle(diff_color, (x1, y1), (x2, y2), color, 2)
                cv2.putText(diff_color, f"ID:{t['id']}", (x1, y1-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Info text
            status_text = "Recalculating..." if (frame_idx % FIT_INTERVAL == 0) else "Using Cache"
            cv2.putText(raw_color, f"Frame {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(raw_color, f"Fit: {status_text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

            combined = np.hstack((raw_color, diff_color))
            cv2.imshow("Quartic Tracker", combined)
            
            frame_idx += 1
            if frame_idx >= proc.num_frames: frame_idx = 0

        # Uncapped speed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord(' '): paused = not paused

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()