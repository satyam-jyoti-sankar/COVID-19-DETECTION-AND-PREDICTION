from django.contrib import admin
from .models import *

# models for image upload
@admin.register(Image)
class ImageAdmin(admin.ModelAdmin):
    list_display=['id','photo','date']


#model for contact form
@admin.register(Contact)
class ContactAdmin(admin.ModelAdmin):
    list_display=['id','name','email','phone','message','date']


#model for register form
@admin.register(Register)
class ContactAdmin(admin.ModelAdmin):
    list_display=['id','name','email','phone','city','date']


#model for Subscribe form
@admin.register(Subscribe)
class SubscribeAdmin(admin.ModelAdmin):
    list_display=['id','email','date']