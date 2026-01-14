"""emotion URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from . import views
from django.conf.urls import url
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('',views.first),
    path('index/',views.index),
    path('register/addreg',views.addreg),
    path('register/',views.register),
    path('login/addlogin',views.addlogin),
    path('login/',views.login),
    path('addlogin/',views.addlogin),

    path('v_users/',views.v_users),

    path('profile/',views.profile, name='profile'),

    path('logout/',views.logout),
    path('add_music/', views.add_music, name='add_music'),
    path('emotion_recommend/', views.emotion_recommend, name='emotion_recommend'),
    path('view_music/', views.view_music, name='view_music'),
    path('emotion_detect/', views.emotion_based_recommend, name='emotion_detect'),
    path('admin/', admin.site.urls),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
