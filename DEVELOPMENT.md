# Development Notes

## Project Structure

The project has been refactored into a clean, modular structure:

```
drowning-detection/
├── src/                    # Source code modules
│   ├── __init__.py        # Package initialization
│   ├── config.py          # Configuration settings
│   ├── model.py           # CNN model definition
│   └── detector.py        # Detection utilities
├── models/                # Trained models
│   ├── model.pth          # PyTorch model weights
│   └── lb.pkl             # Label binarizer
├── videos/                # Sample videos
├── sound/                 # Audio files
├── DrownDetect.py         # Main detection script
├── requirements.txt       # Dependencies
├── setup.py              # Installation script
└── README.md             # Documentation
```

## Troubleshooting

### Audio Issues
- **Problem**: `playsound` module compatibility issues with macOS
- **Solution**: Install PyObjC and use playsound version 1.2.2
- **Commands**:
  ```bash
  pip3 uninstall playsound -y
  pip3 install playsound==1.2.2
  pip3 install PyObjC
  ```

### scikit-learn Version Mismatch
- **Problem**: Label binarizer created with older scikit-learn version
- **Solution**: Regenerate the label binarizer with current version
- **Command**:
  ```bash
  python3 -c "
  import joblib
  from sklearn.preprocessing import LabelBinarizer
  lb = LabelBinarizer()
  lb.fit(['drowning', 'normal'])
  joblib.dump(lb, 'models/lb.pkl')
  print('Label binarizer updated')
  "
  ```

### Model Loading Issues
- **Problem**: Model file not found or incorrect path
- **Solution**: Ensure models are in the `models/` directory
- **Files needed**:
  - `models/model.pth` - PyTorch model weights
  - `models/lb.pkl` - Label binarizer

## Running the System

### Basic Usage
```bash
python3 DrownDetect.py --source video_filename.mp4
```

### With Camera
```bash
python3 DrownDetect.py --source 0
```

### Disable Voice Alerts
```bash
python3 DrownDetect.py --source video.mp4 --no-voice
```

### Custom Confidence Threshold
```bash
python3 DrownDetect.py --source video.mp4 --confidence 0.6
```

## Configuration

Edit `src/config.py` to customize:

- `IMAGE_SIZE`: Input image size for CNN (default: 224)
- `MIN_DISTANCE_THRESHOLD`: Distance threshold for multiple person detection (default: 50)
- `DETECTION_CONFIDENCE`: Object detection confidence (default: 0.5)
- `ENABLE_VOICE_ALERTS`: Enable/disable voice alerts (default: True)

## Development Environment

### Virtual Environment
```bash
python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip3 install -r requirements.txt
```

### Code Style
This project follows Python best practices:
- Use `black` for code formatting
- Follow PEP 8 guidelines
- Add type hints where appropriate
- Write comprehensive docstrings

## Performance Optimization

- The system processes video at 15-30 FPS depending on hardware
- For better performance, consider:
  - Using GPU acceleration with CUDA
  - Reducing input image size
  - Optimizing the CNN model architecture
  - Using more efficient object detection models

## Future Enhancements

1. **Multi-camera Support**: Process multiple video feeds simultaneously
2. **Real-time Alerts**: Integration with alarm systems and notifications
3. **Analytics Dashboard**: Web interface for monitoring and statistics
4. **Model Training**: Tools for retraining with new data
5. **Cloud Deployment**: Docker containers and cloud deployment options
