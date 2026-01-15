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
import os
import sys
import subprocess
import json


# Add ML path to sys.path
sys.path.append(os.path.join(settings.BASE_DIR, 'ML'))

import statistics
from keras import backend as K
# from ML.detect_emotions import detect_emotion  # Removed direct import

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



def emotion_based_recommend(request):
    if not request.session.get('uid'):
        return redirect(login)

    if request.method == 'POST':
        try:
            # Detect emotion using subprocess
            #script_path = os.path.join(settings.BASE_DIR, 'detect_emotion_subprocess.py')
            print("Starting subprocess for emotion detection", file=sys.stderr, flush=True)
            script_path = os.path.join(settings.BASE_DIR, 'ML', 'detect_emotions.py')
            result = subprocess.run([sys.executable, script_path], capture_output=True, text=True, timeout=90)
            
            print(f"Subprocess returncode: {result.returncode}")
            print(f"Subprocess stdout: {repr(result.stdout)}")
            print(f"Subprocess stderr: {repr(result.stderr)}")
            
            if result.returncode == 0:
                output = json.loads(result.stdout.strip())
                detected_emotion = output.get('emotion', 'neutral')
                temp_frame_path = output.get('frame_path')
                
                if temp_frame_path and os.path.exists(temp_frame_path):
                    processed_frame = cv2.imread(temp_frame_path)
                    os.unlink(temp_frame_path)  # Clean up temp file
                else:
                    processed_frame = None
            else:
                print(f"Subprocess error: {result.stderr}")
                detected_emotion = 'neutral'
                processed_frame = None

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