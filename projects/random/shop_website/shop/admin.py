from django.contrib import admin

from .models import *

admin.site.register(Product)
admin.site.register(Comment)
admin.site.register(Review)
admin.site.register(Order)
