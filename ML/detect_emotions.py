from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input
from keras import backend as K
from keras.models import load_model
from statistics import mode
import os
import sys
import cv2
import numpy as np
import tempfile
import json

# parameters for loading data and images
emotion_model_path = os.path.join('ML', 'models', 'emotion_model.hdf5')

# loading models
face_cascade_path = os.path.join('ML', 'models', 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(face_cascade_path)
emotion_classifier = load_model(emotion_model_path, compile=False)  # Avoid optimizer warnings

def detect_emotion():
    """Detect emotion from webcam without GUI display"""
    try:
        K.clear_session()

        # Add a small delay to ensure camera is released from browser
        import time
        time.sleep(1)  # 1 second delay

        
        #emotion_labels = get_labels('fer2013')
        emotion_labels = {0:'angry',1:'disgust',2:'fear',3:'happy', 4:'sad',5:'surprise',6:'neutral'}

        # hyper-parameters for bounding boxes shape
        frame_window = 10
        emotion_offsets = (20, 40)

        

        # getting input model shapes for inference
        emotion_target_size = emotion_classifier.input_shape[1:3]

        # starting lists for calculating modes
        emotion_window = []

        # Try different camera backends and indices
        cap = None
        for camera_index in range(3):
            cap = cv2.VideoCapture(camera_index, cv2.CAP_ANY)  # Try DirectShow backend first
            if cap.isOpened():
                print(f"Successfully opened camera {camera_index} with default backend", file=sys.stderr)
                break
            else:
                print(f"Error: Could not open camera {camera_index} with default backend", file=sys.stderr)
        #cap.release()
        
        
        # backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]  # Try DirectShow, Media Foundation, then any
        
        # for backend in backends:
        #     for camera_index in range(3):  # Try cameras 0, 1, 2
        #         cap = cv2.VideoCapture(camera_index, backend)
        #         if cap.isOpened():
        #             print(f"Successfully opened camera {camera_index} with backend {backend}")
        #             break
        #         cap.release()
        #     if cap and cap.isOpened():
        #         break

        if not cap or not cap.isOpened():
            print("Error: Could not open any camera with any backend", file=sys.stderr)
            return 'neutral', None

        frame_count = 0
        max_frames = 30  # Capture for about 2 seconds at 30fps

        # Store the last processed frame for visualization
        processed_frame = None

        while cap.isOpened() and frame_count < max_frames:
            ret, bgr_image = cap.read()
            if not ret:
                print(f"Warning: Could not read frame {frame_count}", file=sys.stderr)
                frame_count += 1
                continue

            print(f"Frame {frame_count}: Read successfully, shape: {bgr_image.shape}", file=sys.stderr)

            try:
                gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
                rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

                faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5,
                        minSize=(200, 200), flags=cv2.CASCADE_SCALE_IMAGE)

                print(f"Frame {frame_count}: Detected {len(faces)} faces", file=sys.stderr)

                for face_coordinates in faces:
                    x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
                    gray_face = gray_image[y1:y2, x1:x2]
                    try:
                        gray_face = cv2.resize(gray_face, (emotion_target_size))
                    except:
                        continue

                    gray_face = preprocess_input(gray_face, True)
                    gray_face = np.expand_dims(gray_face, 0)
                    gray_face = np.expand_dims(gray_face, -1)
                    emotion_prediction = emotion_classifier.predict(gray_face, verbose=0)  # Suppress TensorFlow warnings
                    emotion_probability = np.max(emotion_prediction)
                    emotion_label_arg = np.argmax(emotion_prediction)
                    emotion_text = emotion_labels[emotion_label_arg]

                    emotion_window.append(emotion_text)

                    # Draw bounding box and emotion label on the BGR image (cv2 expects BGR)
                    if emotion_text == 'angry':
                        color = emotion_probability * np.asarray((0, 0, 255))  # BGR format
                    elif emotion_text == 'sad':
                        color = emotion_probability * np.asarray((255, 0, 0))  # BGR format
                    elif emotion_text == 'happy':
                        color = emotion_probability * np.asarray((0, 255, 255))  # BGR format
                    elif emotion_text == 'surprise':
                        color = emotion_probability * np.asarray((255, 255, 0))  # BGR format
                    elif emotion_text == 'fear':
                        color = emotion_probability * np.asarray((238,130,238))  # BGR format
                    elif emotion_text == 'disgust':
                        color = emotion_probability * np.asarray((0,147,20))  # BGR format
                    else:
                        color = emotion_probability * np.asarray((255, 255, 255))  # BGR format

                    color = color.astype(int)
                    color = color.tolist()

                    draw_bounding_box(face_coordinates, bgr_image, color)
                    draw_text(face_coordinates, bgr_image, emotion_text,
                              color, 0, -45, 1, 1)

                    if len(emotion_window) > frame_window:
                        emotion_window.pop(0)

                # Store the processed frame (already in BGR format)
                if frame_count > max_frames - 10:  # Store last 10 frames
                    processed_frame = bgr_image.copy()

                # Also store any frame that has face detections
                if len(faces) > 0 and (processed_frame is None or frame_count > max_frames - 5):
                    processed_frame = bgr_image.copy()

                # Fallback: store at least one frame if we haven't stored any yet
                if processed_frame is None and frame_count == max_frames // 2:
                    processed_frame = bgr_image.copy()

            except Exception as e:
                print(f"Error processing frame {frame_count}: {str(e)}", file=sys.stderr)

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()
        K.clear_session()

        print(f"Processed frame captured: {processed_frame is not None}", file=sys.stderr)
        if processed_frame is not None:
            print(f"Processed frame shape: {processed_frame.shape}", file=sys.stderr)
            print(f"Processed frame type: {processed_frame.dtype}", file=sys.stderr)

        if emotion_window:
            print("emotion_window:", emotion_window, file=sys.stderr)
            try:
                detected_emotion = mode(emotion_window)
            except:
                detected_emotion = emotion_window[-1] if emotion_window else 'neutral'
        else:
            print("emotion_window is empty", file=sys.stderr)
            detected_emotion = 'neutral'

        return detected_emotion, processed_frame

    except Exception as e:
        print(f"Error in emotion detection: {str(e)}", file=sys.stderr)
        # Suppress TensorFlow warnings in logs
        import logging
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
        return 'neutral', None
    
if __name__ == "__main__":
    try:
        print("Starting emotion detection subprocess", file=sys.stderr, flush=True)
        detected_emotion, processed_frame = detect_emotion()
        print(f"Detection completed: {detected_emotion}", file=sys.stderr, flush=True)

        #Save processed frame to a temporary file if available
        temp_frame_path = None
        if processed_frame is not None:
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                cv2.imwrite(temp_file.name, processed_frame)
                temp_frame_path = temp_file.name
            print(f"Frame saved to: {temp_frame_path}", file=sys.stderr, flush=True)

        # Output result as JSON
        result = {
            'emotion': detected_emotion,
            'frame_path': temp_frame_path
        }
        print(json.dumps(result), flush=True)

    except Exception as e:
        print(f"Exception in subprocess: {str(e)}", file=sys.stderr, flush=True)
        # Output error as JSON
        result = {
            'emotion': 'neutral',
            'frame_path': None,
            'error': str(e)
        }
        print(json.dumps(result), flush=True)