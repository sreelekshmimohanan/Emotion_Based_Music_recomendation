from django.db import models



class regtable(models.Model):
    name=models.CharField(max_length=150)
    phone_number=models.CharField(max_length=120)
    email=models.CharField(max_length=120)
    password=models.CharField(max_length=120)

class Music(models.Model):
    EMOTION_CHOICES = [
        ('happy', 'Happy'),
        ('sad', 'Sad'),
        ('angry', 'Angry'),
        ('fear', 'Fear'),
        ('surprise', 'Surprise'),
        ('disgust', 'Disgust'),
        ('neutral', 'Neutral'),
    ]

    LANGUAGE_CHOICES = [
        ('english', 'English'),
        ('malayalam', 'Malayalam'),
    ]

    name = models.CharField(max_length=200)
    description = models.TextField()
    music_file = models.FileField(upload_to='music/')
    emotion = models.CharField(max_length=20, choices=EMOTION_CHOICES)
    language = models.CharField(max_length=20, choices=LANGUAGE_CHOICES)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name 
