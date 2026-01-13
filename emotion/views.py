from django.http import HttpResponse
from django.shortcuts import render
from django.shortcuts import redirect
# FILE UPLOAD AND VIEW
from  django.core.files.storage import FileSystemStorage
# SESSION
from django.conf import settings
from .models import *

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