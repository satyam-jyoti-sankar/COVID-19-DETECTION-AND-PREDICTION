# Generated by Django 3.2 on 2021-07-04 16:46

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('covid_app', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='image',
            name='name',
            field=models.CharField(default='kanha', max_length=700),
            preserve_default=False,
        ),
    ]