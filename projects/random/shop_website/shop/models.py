from django.contrib.auth.models import User
from django.core.validators import MinValueValidator
from django.db import models
from django_measurement.models import MeasurementField
from djmoney.models.fields import MoneyField
from measurement.measures import Weight


# Create your models here.

class Product(models.Model):
    class ProductType(models.TextChoices):
        GPU = 'GPU', 'Graphics card'
        CPU = 'CPU', 'Processor'
        RAM = 'RAM', 'Random-access memory'
        MB = 'MB', 'Motherboard'
        FAN = 'FAN', 'Cooling Fan'
        PU = 'PU', 'Power supply unit'

    class ConditionType(models.TextChoices):
        BNEW = 'BNEW', 'Brand new'
        NEW = 'NEW', 'New'
        USED_GC = 'USED_GC', 'Used - Good condition'
        USED_PC = 'USED_PC', 'Used - Poor condition'
        BROKEN = 'Broken', 'Broken'

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=128)
    type = models.CharField(choices=ProductType.choices, max_length=32)
    description = models.TextField(max_length=2000)
    image = models.ImageField(blank=True, upload_to="static/images/")
    price = MoneyField(decimal_places=2, default=0, default_currency='PLN', max_digits=7)
    weight = MeasurementField(measurement=Weight, default=0, max_length=6)
    condition = models.CharField(choices=ConditionType.choices, max_length=32)
    available_num = models.IntegerField(validators=[MinValueValidator(0)])
    available_from = models.DateTimeField()

    def __str__(self):
        return self.user.username + "'s " + self.name

    def get_condition(self):
        return self.ConditionType[self.condition].label

    def get_type(self):
        return self.ProductType[self.type].label


class RatingType(models.TextChoices):
    THEBEST = 'THEBEST', 'The best'
    GOOD = 'GOOD', 'Good'
    MEDIUM = 'MEDIUM', 'Medium'
    BAVERAGE = 'BAVERAGE', 'Below average'
    BAD = 'BAD', 'Bad'


class Comment(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    author = models.ForeignKey(User, on_delete=models.CASCADE, related_name="commentauthor")
    title = models.CharField(max_length=128)
    content = models.TextField(max_length=2000)
    rating = models.CharField(choices=RatingType.choices, max_length=32)
    pub_date = models.DateTimeField()

    def __str__(self):
        return self.author.username + " about " + self.user.username

    def get_rating(self):
        return RatingType[self.rating].label


class Review(models.Model):
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    title = models.CharField(max_length=128)
    content = models.TextField(max_length=2000)
    rating = models.CharField(choices=RatingType.choices, max_length=32)
    pub_date = models.DateTimeField()

    def __str__(self):
        return self.author.username + " about " + self.product.name

    def get_rating(self):
        return RatingType[self.rating].label


class Order(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    quantity = models.IntegerField(validators=[MinValueValidator(0)])
    total_price = MoneyField(decimal_places=2, default=0, default_currency='PLN', max_digits=10)
    date = models.DateTimeField()

    def __str__(self):
        return self.user.username + "'s order of " + self.product.name
