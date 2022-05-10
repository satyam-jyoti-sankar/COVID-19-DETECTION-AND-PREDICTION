from django.shortcuts import render
from django.http import HttpResponse
import datetime
from pathlib import Path
import os

from .program1 import *
from .program2 import *
from .program3 import *
# image upload 
from django.forms.forms import Form
from django.shortcuts import render
from .forms import ImageForm
from .models import *

# for directory path
SERVER_URL = str("http://127.0.0.1:8000")
BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = os.path.join(BASE_DIR,'templates')
STATIC_ROOT = os.path.join(BASE_DIR, 'csv_files')
DATA_SETS= os.path.join(BASE_DIR, 'data_sets')
COVID_FILE_PATH = STATIC_ROOT+'\covid_19_data.csv'

# function for homepage
def home(request):
    registerDone = RegisterForm(request)
    subscribeDone = subscribeForm(request)
    date=datetime.datetime.now()
    var = 'Kanhu Charan Swain'
    my_dict={'date_msg':date,'graph':var,'STATIC_DIR':STATIC_ROOT}
    return render(request,'templateApp/index.html',context=my_dict)

# for covid_predict page
def covid_predict(request):
    registerDone = RegisterForm(request)
    subscribeDone = subscribeForm(request)
    chart1= get_plot1() #Distributions plot for Active Cases
    chart2= get_plot2() #Distribution plot  for Closed Cases
    chart3= get_plot3() #Weekly Progress of different types of cases
    chart4= get_plot4() #Daily increase
    chart5= get_plot5() #Top 15 countries as per number of confirmed cases
    chart6= get_plot6() #Weekly Progress of different types of cases for india
    chart7= get_plot7() #graph for holt prediction
    chart8= get_plot8() #prediction using lineaer regression
    chart9= get_plot9() #prediction using SVM

    my_dict={'chart1':chart1,'chart2':chart2,'chart3':chart3,'chart4':chart4,'chart5':chart5,'chart6':chart6,'chart7':chart7,'chart8':chart8,'chart9':chart9}
    return render(request,'templateApp/covid_predict.html',context=my_dict)

# for covid_test page With image upload
def covid_test(request):
    registerDone = RegisterForm(request)
    subscribeDone = subscribeForm(request)
    date=datetime.datetime.now()    
    var = 'Kanhu Charan Swain'
    my_dict={'date_msg':date,'graph':var,'STATIC_DIR':STATIC_ROOT}

    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            # Get the current instance object to display in the template
            img_obj = form.instance

            # upload image path
            UPLOAD_IMAGE_PATH =str(img_obj.photo.url)

            # full path of upload image
            UPLOAD_IMAGE_FULL_PATH = SERVER_URL + UPLOAD_IMAGE_PATH

            # function for send image and get result for covid +/-
            myimg = get_covid_result(UPLOAD_IMAGE_FULL_PATH)
            result = myimg['result']
            covid_chance = myimg['covid_chance']
            normal_chance = myimg['normal_chance']

           
            return render(request, 'templateApp/covid_test.html', {'form': form, 'img_obj': img_obj,'result':result,'covid_chance':covid_chance,'normal_chance':normal_chance})
    else:
        form = ImageForm()
    return render(request, 'templateApp/covid_test.html', {'form': form})




# for world data  page
def world_data(request):
    registerDone = RegisterForm(request)
    subscribeDone = subscribeForm(request)
    my_dict={'chart2':'var1'}
    return render(request,'templateApp/world_data.html',context=my_dict)





# for about page
def about(request):
    registerDone = RegisterForm(request)
    subscribeDone = subscribeForm(request)
    my_dict={'chart2':'var1'}
    return render(request,'templateApp/about.html',context=my_dict)

# for contact page
def contact(request):
    registerDone = RegisterForm(request)
    subscribeDone = subscribeForm(request)
    contactDone  = contactForm(request)
    my_dict={'chart2':'var1','base_url':BASE_DIR}
    return render(request,'templateApp/contact.html',context=my_dict)

# for blog page
# def blog(request):
#     registerDone = RegisterForm(request)
#     subscribeDone = subscribeForm(request)
#     my_dict={'chart2':'var1'}
#     return render(request,'templateApp/blog.html',context=my_dict)


# for about page
def privacy(request):
    registerDone = RegisterForm(request)
    subscribeDone = subscribeForm(request)
    my_dict={'chart2':'var1'}
    return render(request,'templateApp/privacy_policy.html',context=my_dict)

# for about page
def terms(request):
    registerDone = RegisterForm(request)
    subscribeDone = subscribeForm(request)
    my_dict={'chart2':'var1'}
    return render(request,'templateApp/terms_and_conditions.html',context=my_dict)

# contact form data
def contactForm(request):
    if request.method=='POST' and 'contactSubmit' in request.POST:
        get_name = request.POST['name']
        get_email = request.POST['email']
        get_phone = request.POST['phone']
        get_message = request.POST['message']
        ins = Contact(name=get_name, email=get_email, phone=get_phone,message=get_message)
        ins.save()

# register form data
def RegisterForm(request):
    if request.method=='POST' and 'registerBtn' in request.POST:
        get_name = request.POST['name']
        get_email = request.POST['email']
        get_phone = request.POST['phone']
        get_city = request.POST['city']
        ins = Register(name=get_name, email=get_email, phone=get_phone,city=get_city)
        ins.save()

# subsribe email 
def subscribeForm(request):
    if request.method=='POST' and 'subscribeBtn' in request.POST:
        get_email = request.POST['email']
        ins = Subscribe(email=get_email)
        ins.save()
  
  





# def viewgraph(request):
    # chart1= get_plot1()
    # chart2= get_plot2()
    # chart3= get_plot3()
    # chart4= get_plot4()
    # chart5= get_plot5()
    # chart6= get_plot6()
    # my_dict={'chart1':chart1,'chart2':chart2,'chart3':chart3,'chart4':chart4,'chart5':chart5,'chart6':chart6}
    # return render(request,'templateApp/wish.html',context=my_dict)

# def cnn(request):
    # cnn1= test_cnn()

    # UPLOAD_IMAGE_FULL_PATH = SERVER_URL
    # myimg = test_cnn(UPLOAD_IMAGE_FULL_PATH)
  
    # my_dict={'chart1':myimg}
    # return render(request,'templateApp/cnn.html',context=my_dict)

# def test(request):
#     img_path = "data_sets/check/1.jpg" 
#     var1 = covid_result(img_path)
#     my_dict={'result':var1}
#     return render(request,'templateApp/test.html',context=my_dict)


# def SomeFunction(request):
    # result = request.GET.get('foo')
    # foo2 = request.GET.get('bar')
    # return render(request,context=foo)
    # return HttpResponse(result)


