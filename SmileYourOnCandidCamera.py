import cv2
import os
from datetime import timedelta
import numpy as np
from collections import deque
import logging
from pathlib import Path

class SmileDetector:
    def __init__(self, debug_mode=False):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        if (self.face_cascade.empty() or self.profile_cascade.empty() or 
            self.smile_cascade.empty()):
            raise ValueError("Error loading cascade classifiers. Check OpenCV installation.")
        
        self.debug_mode = debug_mode
        self._setup_logging()
    
    def _setup_logging(self):
        logging.basicConfig(
            level=logging.DEBUG if self.debug_mode else logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def preprocess_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        gray = cv2.convertScaleAbs(gray, alpha=1.3, beta=15)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        return gray

    def detect_faces(self, processed_frame, roi=None):
        #"Modified to only detect more frontal faces"
        if roi:
            x1, y1, x2, y2 = roi
            processed_roi = processed_frame[y1:y2, x1:x2]
        else:
            processed_roi = processed_frame
            x1, y1 = 0, 0

        # Only detect frontal faces with stricter parameters
        frontal_faces = self.face_cascade.detectMultiScale(
            processed_roi,
            scaleFactor=1.2,
            minNeighbors=5,        # Increased for more selective detection
            minSize=(50, 50),
            maxSize=(500, 500)
        )

        # Just process frontal faces
        all_faces = [(x + x1, y + y1, w, h) for (x, y, w, h) in frontal_faces]

        if not all_faces:
            return []

        # Only keep the largest face
        largest_face = max(all_faces, key=lambda rect: rect[2] * rect[3])
        return [largest_face]  # Return as list to maintain compatibility
    
    def detect_smile(self, processed_frame, face_rect):
        x, y, w, h = face_rect
        face_roi = processed_frame[y:y + h, x:x + w]
        
        # Focus even more on the mouth area
        lower_half_y = int(h * 0.58)  # Moved even lower
        lower_face_roi = face_roi[lower_half_y:, :]
        
        # Much larger size requirements for obvious smiles
        smile_min_size = (int(w*0.4), int(h*0.22))   # Significantly larger minimums
        smile_max_size = (int(w*0.8), int(h*0.38))   # Slightly larger max size
        
        smiles = self.smile_cascade.detectMultiScale(
            lower_face_roi,
            scaleFactor=1.1,         # Even more precise scaling
            minNeighbors=80,         # Much stricter detection (up from 65)
            minSize=smile_min_size,
            maxSize=smile_max_size
        )
        
        return [(sx, sy + lower_half_y, sw, sh) for (sx, sy, sw, sh) in smiles]

    def draw_debug_overlay(self, frame, faces, smiles_dict, roi=None):
        """Simplified debug overlay for single face"""
        debug_frame = frame.copy()
        
        if roi:
            x1, y1, x2, y2 = roi
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Should only be one face, but keeping the loop for code consistency
        for i, face_rect in enumerate(faces):
            x, y, w, h = face_rect
            cv2.rectangle(debug_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            if tuple(face_rect) in smiles_dict:
                for (sx, sy, sw, sh) in smiles_dict[tuple(face_rect)]:
                    smile_x = x + sx
                    smile_y = y + sy
                    cv2.rectangle(debug_frame, 
                                (smile_x, smile_y), 
                                (smile_x + sw, smile_y + sh),
                                (0, 0, 255), 2)
        
        # Update the face count display
        cv2.putText(debug_frame, f"Faces found: {len(faces)}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return debug_frame

    def process_frame(self, frame, roi=None, debug=False):
        """Modified to process only the largest face"""
        height, width = frame.shape[:2]
        target_width = 1280
        scale = target_width / width
        dim = (target_width, int(height * scale))
        detection_frame = cv2.resize(frame, dim)
        
        if roi is None:
            resized_height, resized_width = detection_frame.shape[:2]
            roi = (
                int(resized_width * 0.25), # Left boundary at 25%
                int(resized_height * 0.1), # Top boundary at 10%
                int(resized_width * 0.75), # Right boundary at 75%
                int(resized_height * 0.6) # Bottom boundary at 60%
            )
        
        processed = self.preprocess_frame(detection_frame)
        faces = self.detect_faces(processed, roi)  # Will now return at most one face
        
        smiles_dict = {}
        has_smile = False
        
        if faces:  # Will only be one face if any
            face_rect = faces[0]
            smiles = self.detect_smile(processed, face_rect)
            if smiles:
                has_smile = True
                smiles_dict[tuple(face_rect)] = smiles
        
        if debug:
            debug_frame = self.draw_debug_overlay(detection_frame, faces, smiles_dict, roi)
            debug_frame = cv2.resize(debug_frame, (1920, 1080), interpolation=cv2.INTER_LINEAR)
        else:
            debug_frame = None
        
        return has_smile, debug_frame

    def extract_smiles_from_video(self, video_path, output_folder, roi=None,
                                skip_frames=4, min_smile_duration=0.3,
                                debug=False, frame_buffer_size=2):
        video_path = Path(video_path)
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        if debug:
            debug_folder = output_folder / 'debug'
            debug_folder.mkdir(exist_ok=True)
        
        video = cv2.VideoCapture(str(video_path))
        if not video.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")
        
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        min_smile_frames = int(min_smile_duration * fps)
        
        frame_buffer = deque(maxlen=frame_buffer_size * 2 + 1)
        processed_buffer = deque(maxlen=frame_buffer_size * 2 + 1)
        
        frame_count = 0
        smile_count = 0
        consecutive_smiles = 0
        last_smile_frame = None
        debug_count = 0
        
        self.logger.info(f"Processing video: {video_path.name}")
        self.logger.info(f"Total frames: {total_frames}, FPS: {fps}")
        
        try:
            while True:
                ret, frame = video.read()
                if not ret:
                    break
                
                frame_buffer.append(frame.copy())
                
                if frame_count % skip_frames == 0:
                    is_smiling, processed_frame = self.process_frame(frame, roi, True)
                    processed_buffer.append((is_smiling, processed_frame))
                    
                    if debug and debug_count % 5000 == 0:
                        debug_path = debug_folder / f'debug_frame_{frame_count}_smile_{is_smiling}.png'
                        cv2.imwrite(str(debug_path), processed_frame, 
                                  [cv2.IMWRITE_PNG_COMPRESSION, 3])
                    
                    debug_count += 1
                    
                    if is_smiling:
                        consecutive_smiles += 1
                        if consecutive_smiles >= min_smile_frames:
                            if (last_smile_frame is None or 
                                frame_count - last_smile_frame > fps * 2):
                                
                                smile_count += 1
                                timestamp = timedelta(seconds=frame_count/fps)
                                
                                for i, buf_frame in enumerate(frame_buffer):
                                    relative_pos = i - frame_buffer_size
                                    
                                    filename = f'smile_{smile_count:03d}_frame_{relative_pos:+d}_time_{timestamp}.png'
                                    output_path = output_folder / filename
                                    cv2.imwrite(str(output_path), buf_frame, 
                                              [cv2.IMWRITE_PNG_COMPRESSION, 3])
                                    
                                    if debug and i < len(processed_buffer):
                                        _, debug_frame = processed_buffer[i]
                                        debug_filename = f'debug_smile_{smile_count:03d}_frame_{relative_pos:+d}_time_{timestamp}.png'
                                        debug_path = debug_folder / debug_filename
                                        cv2.imwrite(str(debug_path), debug_frame,
                                                  [cv2.IMWRITE_PNG_COMPRESSION, 3])
                                
                                last_smile_frame = frame_count
                                self.logger.info(f"Saved smile sequence {smile_count} at {timestamp}")
                    else:
                        consecutive_smiles = 0
                    
                    if frame_count % (skip_frames * 1000) == 0:
                        progress = (frame_count / total_frames) * 100
                        self.logger.info(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
                
                frame_count += 1
            
        except Exception as e:
            self.logger.error(f"Error processing frame {frame_count}: {str(e)}")
            raise
        finally:
            video.release()
        
        self.logger.info(f"Processing complete. Found {smile_count} smiles.")
        return smile_count

def process_videos(base_dir=None):
    if base_dir is None:
        base_dir = Path.home() / 'Desktop/Smile Your On Candid Camera'
    else:
        base_dir = Path(base_dir)
    
    video_dir = base_dir / '1 VIDEO'
    output_dir = base_dir / '2 SMILES'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.m4v')
    
    video_files = [f for f in video_dir.iterdir() 
                  if f.is_file() and f.suffix.lower() in video_extensions]
    
    if not video_files:
        print(f"No video files found in {video_dir}")
        return
    
    print(f"Found {len(video_files)} video files to process:")
    for i, video_file in enumerate(video_files, 1):
        print(f"{i}. {video_file.name}")
    print("\nStarting processing...")
    
    detector = SmileDetector(debug_mode=False)
    total_smiles = 0
    
    for i, video_file in enumerate(video_files, 1):
        video_output_dir = output_dir / video_file.stem
        
        print(f"\n{'='*50}")
        print(f"Processing video {i}/{len(video_files)}: {video_file.name}")
        print(f"Output directory: {video_output_dir}")
        print(f"{'='*50}\n")
        
        try:
            # Get video dimensions for ROI calculation
            temp_video = cv2.VideoCapture(str(video_file))
            if not temp_video.isOpened():
                raise ValueError(f"Error opening video file: {video_file}")
            
            _, first_frame = temp_video.read()
            if first_frame is None:
                raise ValueError(f"Could not read first frame from {video_file}")
                
            height, width = first_frame.shape[:2]
            temp_video.release()
            
            # Calculate ROI for target dimensions (1280p width)
            target_width = 1280
            scale = target_width / width
            target_height = int(height * scale)
            
            roi = (
                int(target_width * 0.25),    # Left boundary at 25% of width
                int(target_height * 0.1),   # Top boundary at 10% of height
                int(target_width * 0.75),     # Right boundary at 75% of width
                int(target_height * 0.6)    # Bottom boundary at 60% of height
            )
            
            try:
                num_smiles = detector.extract_smiles_from_video(
                    video_path=video_file,
                    output_folder=video_output_dir,
                    roi=roi,
                    skip_frames=4,            # Process every other frame
                    min_smile_duration=0.4,   # 400ms minimum smile duration
                    debug=False,
                    frame_buffer_size=2       # Save 2 frames before and after smile
                )
                
                total_smiles += num_smiles
                print(f"\nCompleted {video_file.name}: Found {num_smiles} smiles.")
                
            except Exception as e:
                print(f"Error processing video: {str(e)}")
                detector.logger.error(f"Error processing video: {str(e)}", exc_info=True)
                continue
            
        except Exception as e:
            print(f"Error initializing video {video_file.name}: {str(e)}")
            detector.logger.error(f"Error initializing video: {str(e)}", exc_info=True)
            continue
    
    print(f"\n{'='*50}")
    print(f"Processing complete!")
    print(f"Processed {len(video_files)} videos")
    print(f"Found {total_smiles} total smiles")
    print(f"{'='*50}")

if __name__ == "__main__":
    try:
        process_videos()
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        logging.error("Unexpected error", exc_info=True)