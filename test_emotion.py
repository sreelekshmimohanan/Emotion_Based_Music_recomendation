#!/usr/bin/env python
"""
Test script for emotion detection functionality
"""
import os
import sys
import django

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'emotion.settings')
django.setup()

from emotion.views import detect_emotion

if __name__ == "__main__":
    print("Testing emotion detection...")
    try:
        emotion, frame = detect_emotion()
        print(f"Detected emotion: {emotion}")
        print(f"Frame captured: {frame is not None}")
        print("Test completed successfully!")
    except Exception as e:
        print(f"Error during testing: {str(e)}")