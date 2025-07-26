from src.config import DetectionConfig, MODEL_PATH, LABEL_BINARIZER_PATH
from src.model import CustomCNN
from src.detector import detect_common_objects, draw_bbox, DrowningDetector
import cv2
import numpy as np
import joblib
import torch
import time
from PIL import Image
import argparse
import os
import logging


def main():
    """Main function to run drowning detection."""
    # Define command-line arguments
    parser = argparse.ArgumentParser(description='Drowning Detection System')
    parser.add_argument('--source', required=True, help='Video source file name or camera index')
    parser.add_argument('--confidence', type=float, default=0.5, help='Object detection confidence threshold')
    parser.add_argument('--no-voice', action='store_true', help='Disable voice alerts')
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    print('Loading model and label binarizer...')
    try:
        lb = joblib.load(LABEL_BINARIZER_PATH)
        model = CustomCNN(num_classes=len(lb.classes_))
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        model.eval()
        print('Model loaded successfully!')
    except Exception as e:
        print(f'Error loading model: {e}')
        return
    
    def detect_drowning(source):
        """Detect drowning in video source."""
        is_drowning = False
        frame_count = 0
        
        # Input from camera or video file
        try:
            cap = cv2.VideoCapture(int(source))  # try webcam index
        except ValueError:
            cap = cv2.VideoCapture(f"videos/{source}")  # fallback to file
        
        if not cap.isOpened():
            print('Error: Could not open video source.')
            return
        
        print(f'Video opened successfully. Press "q" to quit.')
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print('End of video or failed to read frame.')
                break
            
            frame_count += 1
            
            # Apply object detection
            bbox, labels, confidences = detect_common_objects(
                frame, confidence=args.confidence
            )
            
            # Filter only person detections
            person_bbox = []
            person_labels = []
            person_conf = []
            
            for i, label in enumerate(labels):
                if label == 'person':
                    person_bbox.append(bbox[i])
                    person_labels.append(labels[i])
                    person_conf.append(confidences[i])
            
            # If only one person is detected, use model-based detection
            if len(person_bbox) == 1:
                model.eval()
                with torch.no_grad():
                    # Preprocess image
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_frame)
                    pil_image = pil_image.resize((DetectionConfig.IMAGE_SIZE, DetectionConfig.IMAGE_SIZE))
                    
                    # Convert to tensor
                    image_array = np.array(pil_image, dtype=np.float32) / 255.0
                    image_tensor = torch.from_numpy(
                        np.transpose(image_array, (2, 0, 1))
                    ).unsqueeze(0)
                    
                    # Get prediction
                    outputs = model(image_tensor)
                    _, preds = torch.max(outputs.data, 1)
                    
                    predicted_class = lb.classes_[preds.item()]
                    print(f"Frame {frame_count}: Swimming status - {predicted_class}")
                    
                    if predicted_class == 'drowning':
                        is_drowning = True
                        if DetectionConfig.ENABLE_VOICE_ALERTS and not args.no_voice:
                            os.system('say "Drowning detected!"')  # MacOS voice alert
                    else:
                        is_drowning = False
                
                out = draw_bbox(frame, person_bbox, person_labels, person_conf, is_drowning)
            
            # If multiple people detected, use proximity-based detection
            elif len(person_bbox) > 1:
                centers = []
                for bbox_coords in person_bbox:
                    center_x = (bbox_coords[0] + bbox_coords[2]) / 2
                    center_y = (bbox_coords[1] + bbox_coords[3]) / 2
                    centers.append([center_x, center_y])
                
                # Calculate distances between people
                min_distance = float('inf')
                for i in range(len(centers)):
                    for j in range(i + 1, len(centers)):
                        dist = np.sqrt(
                            (centers[i][0] - centers[j][0]) ** 2 + 
                            (centers[i][1] - centers[j][1]) ** 2
                        )
                        min_distance = min(min_distance, dist)
                
                # If people are too close, it might indicate drowning/rescue situation
                if min_distance < DetectionConfig.MIN_DISTANCE_THRESHOLD:
                    is_drowning = True
                    print(f"Frame {frame_count}: Multiple people detected - potential rescue situation")
                else:
                    is_drowning = False
                
                out = draw_bbox(frame, person_bbox, person_labels, person_conf, is_drowning)
            
            else:
                # No person detected
                out = frame
                is_drowning = False
            
            # Display output
            cv2.imshow(DetectionConfig.WINDOW_NAME, out)
            
            # Press "q" to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print('Detection stopped.')
    
    # Start detection
    detect_drowning(args.source)


if __name__ == "__main__":
    main()



