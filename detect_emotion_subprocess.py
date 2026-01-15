#!/usr/bin/env python
"""
Script to detect emotion using subprocess for isolation
"""
import os
import sys
import django
import json
import tempfile
import cv2

print("Script started", file=sys.stderr, flush=True)

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'emotion.settings')
print(f"DJANGO_SETTINGS_MODULE set to: {os.environ.get('DJANGO_SETTINGS_MODULE')}", file=sys.stderr, flush=True)

try:
    django.setup()
    print("Django setup successful", file=sys.stderr, flush=True)
except Exception as e:
    print(f"Django setup failed: {str(e)}", file=sys.stderr, flush=True)
    sys.exit(1)

# Add paths
script_dir = os.path.dirname(__file__)
ml_path = os.path.join(script_dir, 'ML')
sys.path.append(ml_path)
print(f"Added to sys.path: {ml_path}", file=sys.stderr, flush=True)

try:
    from ML.detect_emotions import detect_emotion
    print("Import successful", file=sys.stderr, flush=True)
except Exception as e:
    print(f"Import failed: {str(e)}", file=sys.stderr, flush=True)
    sys.exit(1)

if __name__ == "__main__":
    try:
        print("Starting emotion detection subprocess", file=sys.stderr, flush=True)
        detected_emotion, processed_frame = detect_emotion()
        print(f"Detection completed: {detected_emotion}", file=sys.stderr, flush=True)

        # Save processed frame to a temporary file if available
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