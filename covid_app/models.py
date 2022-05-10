from django.db import models

# Create your models here.
class Image(models.Model):
    photo=models.ImageField(upload_to='uploads')
    date=models.DateTimeField(auto_now_add=True)


# model for contact form
class Contact(models.Model):
    name = models.CharField(max_length=255)
    email = models.EmailField()
    phone = models.CharField(max_length=10,default="")
    message = models.TextField()
    date=models.DateTimeField(auto_now_add=True)

# model for register form
class Register(models.Model):
    name = models.CharField(max_length=255)
    email = models.EmailField()
    city = models.CharField(max_length=255)
    phone = models.CharField(max_length=10,default="")
    date=models.DateTimeField(auto_now_add=True)


# model for subscribe by email
class Subscribe(models.Model):
    email = models.EmailField()
    date = models.DateTimeField(auto_now_add=True)