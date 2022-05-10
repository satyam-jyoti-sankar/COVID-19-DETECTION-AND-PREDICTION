# Generated by Django 3.2 on 2021-07-15 18:26

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('covid_app', '0003_remove_image_name'),
    ]

    operations = [
        migrations.CreateModel(
            name='ContactNew',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255)),
                ('email', models.EmailField(max_length=254)),
                ('phone', models.CharField(default='', max_length=10)),
                ('message', models.TextField()),
            ],
        ),
    ]
