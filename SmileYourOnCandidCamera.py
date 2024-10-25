import cv2
import os
from datetime import timedelta
import numpy as np
from collections import deque

class SmileDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        if self.face_cascade.empty() or self.smile_cascade.empty():
            raise ValueError("Error loading cascade classifiers. Check OpenCV installation.")

    def preprocess_frame(self, frame):
        """Preprocess frame to improve detection with glasses"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        return gray

    def draw_debug_overlay(self, frame, face_rect, smiles, roi=None):
        """Draw debug information on frame"""
        debug_frame = frame.copy()
        
        if roi:
            x1, y1, x2, y2 = roi
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw face detection
        if face_rect is not None:
            x, y, w, h = face_rect
            cv2.rectangle(debug_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Draw only valid smiles within the lower half of face
            for (sx, sy, sw, sh) in smiles:
                # Convert smile coordinates to frame coordinates
                frame_sx = x + sx
                frame_sy = y + sy
                
                # Draw smile rectangle
                cv2.rectangle(debug_frame, 
                            (frame_sx, frame_sy), 
                            (frame_sx + sw, frame_sy + sh),
                            (0, 0, 255), 2)
        
        return debug_frame

    def process_frame(self, frame, roi=None, debug=False):
        """Process a single frame to detect smiles"""
        height, width = frame.shape[:2]
        target_width = 800
        scale = target_width / width
        dim = (target_width, int(height * scale))
        frame = cv2.resize(frame, dim)
        
        processed = self.preprocess_frame(frame)
        
        if roi:
            x1, y1, x2, y2 = roi
            processed_roi = processed[y1:y2, x1:x2]
        else:
            processed_roi = processed
            x1, y1 = 0, 0
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            processed_roi,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100),
            maxSize=(400, 400)
        )
        
        largest_face = None
        smiles = []
        has_smile = False
        
        # Only process the largest face detection
        if len(faces) > 0:
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = largest_face
            
            # Adjust coordinates to account for ROI offset
            largest_face = (x + x1, y + y1, w, h)
            
            # Extract lower half of face for smile detection
            face_roi = processed[y + y1:y + y1 + h, x + x1:x + x1 + w]
            lower_half_y = h // 2
            lower_face_roi = face_roi[lower_half_y:, :]
            
            # Detect smiles only in lower half of face
            smiles = self.smile_cascade.detectMultiScale(
                lower_face_roi,
                scaleFactor=1.2,
                minNeighbors=25,
                minSize=(int(w*0.4), int(h*0.1)),
                maxSize=(int(w*0.7), int(h*0.3))
            )
            
            # Adjust smile coordinates to account for lower face ROI
            smiles = [(sx, sy + lower_half_y, sw, sh) for (sx, sy, sw, sh) in smiles]
            
            # Consider it a valid smile if detected within the lower face region
            has_smile = len(smiles) > 0
        
        if debug:
            return has_smile, self.draw_debug_overlay(frame, largest_face, smiles, roi)
        return has_smile, frame

    def extract_smiles_from_video(self, video_path, output_folder, roi=None, 
                                skip_frames=3, min_smile_duration=0.3, 
                                debug=False, frame_buffer_size=2):
        """
        Extract frames containing smiles from a video file, including surrounding frames
        
        Args:
            frame_buffer_size (int): Number of frames to save before and after the smile
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")
        
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        min_smile_frames = int(min_smile_duration * fps)
        
        # Create debug folder if needed
        debug_folder = os.path.join(output_folder, 'debug') if debug else None
        if debug:
            os.makedirs(debug_folder, exist_ok=True)
        
        # Initialize frame buffer
        frame_buffer = deque(maxlen=frame_buffer_size * 2 + 1)
        processed_buffer = deque(maxlen=frame_buffer_size * 2 + 1)
        
        frame_count = 0
        smile_count = 0
        consecutive_smiles = 0
        last_smile_frame = None
        
        while True:
            ret, frame = video.read()
            if not ret:
                break
            
            # Store original frame in buffer
            frame_buffer.append(frame.copy())
            
            if frame_count % skip_frames == 0:
                is_smiling, processed_frame = self.process_frame(frame, roi, debug)
                processed_buffer.append((is_smiling, processed_frame))
                
                if is_smiling:
                    consecutive_smiles += 1
                    if consecutive_smiles >= min_smile_frames:
                        if (last_smile_frame is None or 
                            frame_count - last_smile_frame > fps):
                            
                            smile_count += 1
                            timestamp = timedelta(seconds=frame_count/fps)
                            
                            # Save sequence of frames
                            for i, buf_frame in enumerate(frame_buffer):
                                # Calculate relative position (-2, -1, 0, 1, 2)
                                relative_pos = i - frame_buffer_size
                                
                                # Save clean frame
                                _, clean_frame = self.process_frame(buf_frame, roi, debug=False)
                                filename = f'smile_{smile_count:03d}_frame_{relative_pos:+d}_time_{timestamp}.jpg'
                                output_path = os.path.join(output_folder, filename)
                                cv2.imwrite(output_path, clean_frame)
                                
                                # Save debug frame if debug mode is on
                                if debug and i < len(processed_buffer):
                                    _, debug_frame = processed_buffer[i]
                                    debug_filename = f'debug_smile_{smile_count:03d}_frame_{relative_pos:+d}_time_{timestamp}.jpg'
                                    debug_path = os.path.join(debug_folder, debug_filename)
                                    cv2.imwrite(debug_path, debug_frame)
                            
                            last_smile_frame = frame_count
                            print(f"Saved smile sequence {smile_count} at {timestamp}")
                else:
                    consecutive_smiles = 0
                
                if frame_count % (skip_frames * 100) == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
            
            frame_count += 1
        
        video.release()
        return smile_count

def process_videos():
    detector = SmileDetector()
    
    # Define input and output directories
    base_dir = '/Users/randyrektor/Desktop/Smile Your On Candid Camera'
    video_dir = os.path.join(base_dir, '1 VIDEO')
    output_dir = os.path.join(base_dir, '2 SMILES')
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Common video file extensions
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.m4v')
    
    # Get list of video files
    video_files = [f for f in os.listdir(video_dir) 
                  if os.path.isfile(os.path.join(video_dir, f)) 
                  and f.lower().endswith(video_extensions)]
    
    if not video_files:
        print(f"No video files found in {video_dir}")
        return
    
    print(f"Found {len(video_files)} video files to process:")
    for i, video_file in enumerate(video_files, 1):
        print(f"{i}. {video_file}")
    print("\nStarting processing...")
    
    # Process each video file
    total_smiles = 0
    for i, video_file in enumerate(video_files, 1):
        video_path = os.path.join(video_dir, video_file)
        
        # Create a subdirectory for this video's output
        video_name = os.path.splitext(video_file)[0]
        video_output_dir = os.path.join(output_dir, video_name)
        
        print(f"\n{'='*50}")
        print(f"Processing video {i}/{len(video_files)}: {video_file}")
        print(f"Output directory: {video_output_dir}")
        print(f"{'='*50}\n")
        
        try:
            # Adjusted ROI for podcast setup
            roi = (200, 50, 600, 450)
            
            num_smiles = detector.extract_smiles_from_video(
                video_path=video_path,
                output_folder=video_output_dir,
                roi=roi,
                skip_frames=3,
                min_smile_duration=0.3,
                debug=True,
                frame_buffer_size=2
            )
            
            total_smiles += num_smiles
            print(f"\nCompleted {video_file}: Found {num_smiles} smiles.")
            
        except Exception as e:
            print(f"Error processing {video_file}: {str(e)}")
            continue
    
    print(f"\n{'='*50}")
    print(f"Processing complete!")
    print(f"Processed {len(video_files)} videos")
    print(f"Found {total_smiles} total smiles")
    print(f"{'='*50}")

if __name__ == "__main__":
    process_videos()