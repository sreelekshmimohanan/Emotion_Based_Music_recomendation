from django.http import HttpResponse
from django.shortcuts import render
from django.shortcuts import redirect
# FILE UPLOAD AND VIEW
from  django.core.files.storage import FileSystemStorage
# SESSION
from django.conf import settings
from .models import *

# ML imports for emotion detection
import cv2
import numpy as np
import h5py
from keras.models import load_model
from statistics import mode
import os
import sys

# Add ML path to sys.path
sys.path.append(os.path.join(settings.BASE_DIR, 'ML'))
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input
import statistics
from keras import backend as K

def first(request):
    return render(request,'index.html')
def index(request):
    return render(request,'index.html')
def addreg(request):
    if request.method=="POST":
        a=request.POST.get('name')
        b=request.POST.get('phone_number')
        c=request.POST.get('email')
        d=request.POST.get('password')
        e=regtable(name=a,phone_number=b,email=c,password=d)
        e.save()
    return redirect(login) 

def register(request):
    return render(request,'register.html')

def login(request):
    return render(request,'login.html')

def addlogin(request):
    email = request.POST.get('email')
    password = request.POST.get('password')
    if email == 'admin@gmail.com' and password =='admin':
        request.session['admin'] = 'admin'
        return render(request,'index.html')

    elif regtable.objects.filter(email=email,password=password).exists():
            userdetails=regtable.objects.get(email=request.POST['email'], password=password)
            request.session['uid'] = userdetails.id
            return render(request,'index.html')

    else:
        return render(request, 'login.html', {'message':'Invalid Email or Password'})
    



def v_users(request):
    user=regtable.objects.all()
    return render(request,'viewusers.html',{'result':user})





def profile(request):
    uid = request.session.get('uid')
    if not uid:
        return redirect(login)
    try:
        user = regtable.objects.get(id=uid)
    except regtable.DoesNotExist:
        return redirect(login)
    return render(request, 'profile.html', {'user': user})

def logout(request):
    session_keys=list(request.session.keys())
    for key in session_keys:
        del request.session[key]
    return redirect(first)

def add_music(request):
    if not request.session.get('admin'):
        return redirect(login)
    if request.method == "POST":
        name = request.POST.get('name')
        description = request.POST.get('description')
        emotion = request.POST.get('emotion')
        language = request.POST.get('language')
        music_file = request.FILES.get('music_file')

        if music_file:
            music = Music(name=name, description=description, emotion=emotion, language=language, music_file=music_file)
            music.save()
            return render(request, 'add_music.html', {'message': 'Music added successfully!'})
        else:
            return render(request, 'add_music.html', {'message': 'Please upload a music file.'})

    return render(request, 'add_music.html')

def emotion_recommend(request):
    if not request.session.get('uid'):
        return redirect(login)

    # Get all music grouped by emotion
    music_by_emotion = {}
    for emotion_choice in Music.EMOTION_CHOICES:
        emotion_value, emotion_display = emotion_choice
        music_by_emotion[emotion_display] = Music.objects.filter(emotion=emotion_value)

    return render(request, 'emotion_recommend.html', {'music_by_emotion': music_by_emotion})

def view_music(request):
    if not request.session.get('uid'):
        return redirect(login)

    # Get all music ordered by upload date (newest first)
    all_music = Music.objects.all().order_by('-uploaded_at')

    return render(request, 'view_music.html', {'music_list': all_music})

def detect_emotion():
    """Detect emotion from webcam without GUI display"""
    try:
        K.clear_session()

        # Add a small delay to ensure camera is released from browser
        import time
        time.sleep(1)  # 1 second delay

        # parameters for loading data and images
        emotion_model_path = os.path.join(settings.BASE_DIR, 'ML', 'models', 'emotion_model.hdf5')
        emotion_labels = get_labels('fer2013')

        # hyper-parameters for bounding boxes shape
        frame_window = 10
        emotion_offsets = (20, 40)

        # loading models
        face_cascade_path = os.path.join(settings.BASE_DIR, 'ML', 'models', 'haarcascade_frontalface_default.xml')
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        emotion_classifier = load_model(emotion_model_path, compile=False)  # Avoid optimizer warnings

        # getting input model shapes for inference
        emotion_target_size = emotion_classifier.input_shape[1:3]

        # starting lists for calculating modes
        emotion_window = []

        # Try different camera backends and indices
        cap = None
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]  # Try DirectShow, Media Foundation, then any
        
        for backend in backends:
            for camera_index in range(3):  # Try cameras 0, 1, 2
                cap = cv2.VideoCapture(camera_index, backend)
                if cap.isOpened():
                    print(f"Successfully opened camera {camera_index} with backend {backend}")
                    break
                cap.release()
            if cap and cap.isOpened():
                break

        if not cap or not cap.isOpened():
            print("Error: Could not open any camera with any backend")
            return 'neutral', None

        frame_count = 0
        max_frames = 60  # Capture for about 2 seconds at 30fps

        # Store the last processed frame for visualization
        processed_frame = None

        while cap.isOpened() and frame_count < max_frames:
            ret, bgr_image = cap.read()
            if not ret:
                print(f"Warning: Could not read frame {frame_count}")
                frame_count += 1
                continue

            print(f"Frame {frame_count}: Read successfully, shape: {bgr_image.shape}")

            try:
                gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
                rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

                faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=3,
                        minSize=(20, 20), flags=cv2.CASCADE_SCALE_IMAGE)

                print(f"Frame {frame_count}: Detected {len(faces)} faces")

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
                print(f"Error processing frame {frame_count}: {str(e)}")

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()
        K.clear_session()

        print(f"Processed frame captured: {processed_frame is not None}")
        if processed_frame is not None:
            print(f"Processed frame shape: {processed_frame.shape}")
            print(f"Processed frame type: {processed_frame.dtype}")

        if emotion_window:
            print("emotion_window:", emotion_window)
            try:
                detected_emotion = mode(emotion_window)
            except:
                detected_emotion = emotion_window[-1] if emotion_window else 'neutral'
        else:
            print("emotion_window is empty")
            detected_emotion = 'neutral'

        return detected_emotion, processed_frame

    except Exception as e:
        print(f"Error in emotion detection: {str(e)}")
        # Suppress TensorFlow warnings in logs
        import logging
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
        return 'neutral', None

def emotion_based_recommend(request):
    if not request.session.get('uid'):
        return redirect(login)

    if request.method == 'POST':
        try:
            # Detect emotion and get processed frame
            detected_emotion, processed_frame = detect_emotion()

            # Determine if faces were detected
            faces_detected = detected_emotion != 'neutral' or processed_frame is not None

            # Save processed frame temporarily for display
            processed_image_path = None
            if processed_frame is not None:
                import uuid
                from datetime import datetime, timedelta

                # Clean up old processed images (older than 1 hour)
                processed_dir = os.path.join(settings.MEDIA_ROOT, 'processed')
                if os.path.exists(processed_dir):
                    for filename in os.listdir(processed_dir):
                        filepath = os.path.join(processed_dir, filename)
                        if os.path.isfile(filepath):
                            # Check if file is older than 1 hour
                            file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                            if datetime.now() - file_time > timedelta(hours=1):
                                try:
                                    os.remove(filepath)
                                except:
                                    pass  # Ignore errors when deleting old files

                filename = f"processed_emotion_{uuid.uuid4().hex}.jpg"
                processed_image_path = os.path.join(settings.MEDIA_ROOT, 'processed', filename)
                os.makedirs(os.path.dirname(processed_image_path), exist_ok=True)

                # Save the processed image
                save_result = cv2.imwrite(processed_image_path, processed_frame)

                if save_result and os.path.exists(processed_image_path):
                    processed_image_url = f"/media/processed/{filename}"
                else:
                    processed_image_url = None
            else:
                processed_image_url = None

            # Map detected emotion to database emotion choices
            emotion_mapping = {
                'angry': 'angry',
                'disgust': 'disgust',
                'fear': 'fear',
                'happy': 'happy',
                'sad': 'sad',
                'surprise': 'surprise',
                'neutral': 'neutral'
            }

            db_emotion = emotion_mapping.get(detected_emotion, 'neutral')

            # Get recommended music
            recommended_music = Music.objects.filter(emotion=db_emotion).order_by('-uploaded_at')

            return render(request, 'emotion_based_recommend.html', {
                'detected_emotion': detected_emotion,
                'recommended_music': recommended_music,
                'emotion_display': dict(Music.EMOTION_CHOICES)[db_emotion],
                'processed_image_url': processed_image_url,
                'faces_detected': faces_detected
            })

        except Exception as e:
            print(f"Error in emotion_based_recommend: {str(e)}")
            # Fallback to neutral emotion if detection fails
            recommended_music = Music.objects.filter(emotion='neutral').order_by('-uploaded_at')
            return render(request, 'emotion_based_recommend.html', {
                'detected_emotion': 'neutral',
                'recommended_music': recommended_music,
                'emotion_display': 'Neutral',
                'processed_image_url': None,
                'error_message': 'Emotion detection encountered an issue. Showing neutral music recommendations.',
                'faces_detected': False
            })

    return render(request, 'emotion_detect.html')