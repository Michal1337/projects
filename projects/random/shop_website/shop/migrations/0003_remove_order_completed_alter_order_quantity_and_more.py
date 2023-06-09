# Generated by Django 4.1.5 on 2023-01-21 15:11

import django.core.validators
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('shop', '0002_order_remove_cartdetails_cart_and_more'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='order',
            name='completed',
        ),
        migrations.AlterField(
            model_name='order',
            name='quantity',
            field=models.IntegerField(validators=[django.core.validators.MinValueValidator(0)]),
        ),
        migrations.AlterField(
            model_name='product',
            name='image',
            field=models.ImageField(blank=True, upload_to='static/images/'),
        ),
    ]
