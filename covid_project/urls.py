from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from covid_app import views 

urlpatterns = [
    path('admin/', admin.site.urls),

    path('', views.home), #home page

    path('predict_covid/', views.covid_predict), #predict covid showa graph

    path('test_covid/', views.covid_test), #test covid upload photo test

    path('about/', views.about), #about our project

    path('contact/', views.contact), #for contact page map ans contact form

    # path('blog/', views.blog), #for blog

    path('privacy-policy/', views.privacy), #for privacy policy

    path('terms-and-conditions/', views.terms), #terms and conditions

    path('world-data/', views.world_data), #terms and conditions


]+static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)   #url pattern for media files
