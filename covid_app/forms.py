from django import forms
from .models import *

#For Image upload form 
class ImageForm(forms.ModelForm):
    class Meta:
        model=Image
        fields= ('photo',) #total fields we have taken 
        labels = { 'photo' : ''} #used for remove lable
        widgets= {
            'photo' : forms.FileInput(attrs={'id' :'fileup'}), #attrs used to add id,class to form input fields
        } 

#For contact form 
class ContactForm(forms.ModelForm):
    class Meta:
        model = Contact
        fields = '__all__'   

#For Register form 
class RegisterForm(forms.ModelForm):
    class Meta:
        model = Register
        fields = '__all__'  

#For Subscribe form 
class SubscribeForm(forms.ModelForm):
    class Meta:
        model = Subscribe
        fields = '__all__'   