"""Object detection utilities for drowning detection system."""

import cv2
import os
import numpy as np
import requests
import subprocess
import sys
import threading
from typing import List, Tuple, Optional


def download_file(url: str, file_name: str, dest_dir: str) -> Optional[str]:
    """Download a file from URL to destination directory."""
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    full_path_to_file = os.path.join(dest_dir, file_name)

    if os.path.exists(full_path_to_file):
        return full_path_to_file

    print(f"Downloading {file_name} from {url}")

    try:
        r = requests.get(url, allow_redirects=True, stream=True)
        if r.status_code != requests.codes.ok:
            print("Error occurred while downloading file")
            return None
            
        with open(full_path_to_file, 'wb') as file:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
                    
        print(f"Successfully downloaded {file_name}")
        return full_path_to_file
        
    except Exception as e:
        print(f"Could not establish connection. Download failed: {e}")
        return None


def play_sound() -> None:
    """Play alarm sound when drowning is detected."""
    try:
        # Get the project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sound_path = os.path.join(project_root, 'sound', 'alarm.mp3')
        
        # Check if file exists
        if os.path.exists(sound_path):
            # Use system-specific audio player
            if sys.platform == "darwin":  # macOS
                subprocess.run(["afplay", sound_path], check=False)
            elif sys.platform.startswith("linux"):  # Linux
                subprocess.run(["aplay", sound_path], check=False)
            elif sys.platform == "win32":  # Windows
                import winsound
                winsound.PlaySound(sound_path, winsound.SND_FILENAME)
            else:
                print("Warning: Unsupported platform for audio playback")
        else:
            print(f"Warning: Sound file not found at {sound_path}")
    except Exception as e:
        print(f"Warning: Could not play sound - {e}")


def populate_class_labels() -> List[str]:
    """Load YOLO class labels."""
    project_dir = os.path.dirname(os.path.abspath(__file__))
    dest_dir = os.path.join(project_dir, '..', 'models', 'yolo')
    
    class_file_name = 'coco_classes.txt'
    class_file_abs_path = os.path.join(dest_dir, class_file_name)
    url = 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names'
    
    if not os.path.exists(class_file_abs_path):
        download_file(url=url, file_name=class_file_name, dest_dir=dest_dir)
    
    try:
        with open(class_file_abs_path, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        return classes
    except FileNotFoundError:
        # Fallback to common COCO classes
        return ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck']


def get_output_layers(net) -> List[str]:
    """Get output layer names from YOLO network."""
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


def draw_bbox(img: np.ndarray, bbox: List[List[int]], labels: List[str], 
              confidence: List[float], drowning: bool, write_conf: bool = False) -> np.ndarray:
    """Draw bounding boxes on image with drowning detection."""
    
    # Colors: Green for normal, Red for drowning, White for other
    COLORS = [(0, 255, 0), (0, 0, 255), (255, 255, 255)]
    
    for i, label in enumerate(labels):
        # If the person is drowning, the box will be drawn red instead of green
        if label == 'person' and drowning:
            color = COLORS[1]  # Red
            label = 'ALERT DROWNING'
            # Play sound in separate thread to avoid blocking
            threading.Thread(target=play_sound, daemon=True).start()
        else:
            color = COLORS[0]  # Green
            label = 'Normal'

        if write_conf:
            label += f' {confidence[i] * 100:.2f}%'

        # Draw rectangle
        cv2.rectangle(img, (bbox[i][0], bbox[i][1]), (bbox[i][2], bbox[i][3]), color, 2)
        
        # Put text
        cv2.putText(img, label, (bbox[i][0], bbox[i][1] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return img


def detect_common_objects(image: np.ndarray, confidence: float = 0.5, 
                         nms_thresh: float = 0.3) -> Tuple[List[List[int]], List[str], List[float]]:
    """Detect common objects using YOLO."""
    
    if image is None:
        return [], [], []
    
    height, width = image.shape[:2]
    scale = 0.00392
    
    # Setup model paths
    project_dir = os.path.dirname(os.path.abspath(__file__))
    dest_dir = os.path.join(project_dir, '..', 'models', 'yolo')
    
    config_file_name = 'yolov3.cfg'
    config_file_path = os.path.join(dest_dir, config_file_name)
    
    weights_file_name = 'yolov3.weights'
    weights_file_path = os.path.join(dest_dir, weights_file_name)
    
    # Try to initialize network (cache for performance)
    if not hasattr(detect_common_objects, 'net'):
        classes = populate_class_labels()
        detect_common_objects.classes = classes
        
        try:
            # Try YOLOv4 first (better compatibility)
            config_url = 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg'
            weights_url = 'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights'
            config_file_name = 'yolov4.cfg'
            weights_file_name = 'yolov4.weights'
            config_file_path = os.path.join(dest_dir, config_file_name)
            weights_file_path = os.path.join(dest_dir, weights_file_name)
            
            if not os.path.exists(config_file_path):
                download_file(url=config_url, file_name=config_file_name, dest_dir=dest_dir)
            
            if not os.path.exists(weights_file_path):
                print("Note: YOLOv4 weights are large (~250MB). This may take a while...")
                download_file(url=weights_url, file_name=weights_file_name, dest_dir=dest_dir)
            
            detect_common_objects.net = cv2.dnn.readNet(weights_file_path, config_file_path)
            print("Successfully loaded YOLOv4 model")
            
        except Exception as e:
            print(f"Warning: Could not load YOLO model ({e}). Using fallback detection.")
            # Fallback: Use OpenCV's built-in HOG + SVM person detector
            detect_common_objects.net = None
            detect_common_objects.hog = cv2.HOGDescriptor()
            detect_common_objects.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    net = detect_common_objects.net
    classes = detect_common_objects.classes
    
    if net is not None:
        # Use YOLO detection
        try:
            # Create blob from image
            blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            
            # Run forward pass
            outs = net.forward(get_output_layers(net))
            
            # Process detections
            class_ids = []
            confidences = []
            boxes = []
            
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    max_conf = scores[class_id]
                    
                    if max_conf > confidence:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = center_x - w // 2
                        y = center_y - h // 2
                        
                        class_ids.append(class_id)
                        confidences.append(float(max_conf))
                        boxes.append([x, y, w, h])
            
            # Apply Non-Maximum Suppression
            indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence, nms_thresh)
            
            bbox = []
            label = []
            conf = []
            
            if len(indices) > 0:
                for i in indices.flatten():
                    box = boxes[i]
                    x, y, w, h = box
                    bbox.append([x, y, x + w, y + h])
                    label.append(classes[class_ids[i]])
                    conf.append(confidences[i])
            
            return bbox, label, conf
            
        except Exception as e:
            print(f"YOLO detection failed: {e}. Using HOG fallback.")
            net = None
    
    # Fallback: Use HOG + SVM person detector
    if hasattr(detect_common_objects, 'hog'):
        hog = detect_common_objects.hog
        
        # Detect people using HOG
        (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)
        
        bbox = []
        label = []
        conf = []
        
        for (x, y, w, h) in rects:
            bbox.append([x, y, x + w, y + h])
            label.append('person')
            conf.append(0.8)  # Default confidence for HOG detection
        
        return bbox, label, conf
    
    # If all else fails, return empty results
    return [], [], []

"""
Drowning detection utilities and helper functions.
"""

import cv2
import numpy as np
import torch
from PIL import Image
import logging
from typing import Tuple, List, Optional
import time

from .config import DetectionConfig
from .model import CustomCNN


class DrowningDetector:
    """
    Main drowning detection class that handles video processing and classification.
    """
    
    def __init__(self, model: CustomCNN, label_binarizer, confidence_threshold: float = 0.5):
        """
        Initialize the drowning detector.
        
        Args:
            model (CustomCNN): Trained CNN model for classification
            label_binarizer: Fitted label binarizer for class names
            confidence_threshold (float): Confidence threshold for object detection
        """
        self.model = model
        self.label_binarizer = label_binarizer
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)
        
        # Detection statistics
        self.total_frames = 0
        self.drowning_frames = 0
        self.start_time = time.time()
    
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess frame for model inference.
        
        Args:
            frame (np.ndarray): Input frame from video
            
        Returns:
            torch.Tensor: Preprocessed tensor ready for model input
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        pil_image = Image.fromarray(rgb_frame)
        pil_image = pil_image.resize((DetectionConfig.IMAGE_SIZE, DetectionConfig.IMAGE_SIZE))
        
        # Normalize and convert to tensor
        image_array = np.array(pil_image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(np.transpose(image_array, (2, 0, 1))).unsqueeze(0)
        
        return image_tensor
    
    def classify_drowning(self, frame: np.ndarray) -> Tuple[str, float]:
        """
        Classify whether a person in the frame is drowning.
        
        Args:
            frame (np.ndarray): Input frame containing a person
            
        Returns:
            Tuple[str, float]: (predicted_class, confidence_score)
        """
        self.model.eval()
        
        with torch.no_grad():
            # Preprocess frame
            input_tensor = self.preprocess_frame(frame)
            
            # Get model prediction
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
            
            predicted_class = self.label_binarizer.classes_[prediction.item()]
            confidence_score = confidence.item()
            
        return predicted_class, confidence_score
    
    def calculate_person_distances(self, bboxes: List[List[int]]) -> float:
        """
        Calculate minimum distance between detected persons.
        
        Args:
            bboxes (List[List[int]]): List of bounding boxes [x1, y1, x2, y2]
            
        Returns:
            float: Minimum distance between any two persons
        """
        if len(bboxes) < 2:
            return float('inf')
        
        centers = []
        for bbox in bboxes:
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            centers.append([center_x, center_y])
        
        min_distance = float('inf')
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                distance = np.sqrt(
                    (centers[i][0] - centers[j][0]) ** 2 + 
                    (centers[i][1] - centers[j][1]) ** 2
                )
                min_distance = min(min_distance, distance)
        
        return min_distance
    
    def detect_drowning_in_frame(self, frame: np.ndarray, person_bboxes: List[List[int]]) -> Tuple[bool, str]:
        """
        Detect drowning in a single frame.
        
        Args:
            frame (np.ndarray): Input frame
            person_bboxes (List[List[int]]): List of person bounding boxes
            
        Returns:
            Tuple[bool, str]: (is_drowning, detection_method)
        """
        self.total_frames += 1
        
        if len(person_bboxes) == 0:
            return False, "no_person_detected"
        
        elif len(person_bboxes) == 1:
            # Single person: use CNN classification
            predicted_class, confidence = self.classify_drowning(frame)
            
            is_drowning = predicted_class == 'drowning'
            if is_drowning:
                self.drowning_frames += 1
                
            self.logger.info(f"CNN Classification: {predicted_class} (confidence: {confidence:.3f})")
            return is_drowning, "cnn_classification"
        
        else:
            # Multiple persons: use proximity-based detection
            min_distance = self.calculate_person_distances(person_bboxes)
            
            is_drowning = min_distance < DetectionConfig.MIN_DISTANCE_THRESHOLD
            if is_drowning:
                self.drowning_frames += 1
                
            self.logger.info(f"Proximity Detection: min_distance={min_distance:.1f}, threshold={DetectionConfig.MIN_DISTANCE_THRESHOLD}")
            return is_drowning, "proximity_detection"
    
    def get_detection_stats(self) -> dict:
        """
        Get detection statistics.
        
        Returns:
            dict: Dictionary containing detection statistics
        """
        elapsed_time = time.time() - self.start_time
        fps = self.total_frames / elapsed_time if elapsed_time > 0 else 0
        
        return {
            'total_frames': self.total_frames,
            'drowning_frames': self.drowning_frames,
            'drowning_percentage': (self.drowning_frames / self.total_frames * 100) if self.total_frames > 0 else 0,
            'elapsed_time': elapsed_time,
            'fps': fps
        }
