import re

from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.forms import ModelForm
from djmoney.forms.fields import MoneyField

from .models import User, Product, Comment, Review, Order


class RegisterForm(UserCreationForm):
    class Meta:
        model = User
        fields = ['username', 'first_name', 'last_name', 'email', 'password1', 'password2']

    def is_valid_custom(self):
        cd = self.cleaned_data

        username = cd.get("username")
        first_name = cd.get("first_name")
        last_name = cd.get("last_name")

        pattern_username = r"^[A-Za-z0-9]*"
        pattern_first_name = r"^[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]*"

        match1 = re.search(pattern_username, username)
        match2 = re.search(pattern_first_name, first_name)
        match3 = re.search(pattern_first_name, last_name)

        if len(username) > 4 and len(first_name) > 2 and len(
                last_name) > 2 and match1 is not None and match2 is not None and match3 is not None:
            return True
        return False


class AddProductForm(ModelForm):
    class Meta:
        model = Product
        fields = ['name', 'type', 'price', 'description', 'image', 'weight', 'condition', 'available_num']

    def is_valid_custom(self):
        cd = self.cleaned_data

        name = cd.get("name")
        price = cd.get("price")
        desc = cd.get("description")
        weight = cd.get("weight")
        avlb = cd.get("available_num")

        if len(name) > 6 and len(desc) > 20 and price.amount > 0 and weight.value > 0 and avlb > 0 and len(
                name) < 128 and len(desc) < 2000:
            return True
        return False


class AddReviewForm(ModelForm):
    class Meta:
        model = Review
        fields = ['title', 'content', 'rating']


class AddCommentForm(ModelForm):
    class Meta:
        model = Comment
        fields = ['title', 'content', 'rating']

    def is_valid_custom(self):
        cd = self.cleaned_data

        title = cd.get("title")
        content = cd.get("content")

        if len(title) > 5 and len(content) > 10 and len(title) < 128 and len(content) < 2000:
            return True
        return False


class AddOrderForm(ModelForm):
    class Meta:
        model = Order
        fields = ['quantity', 'total_price']


class SearchForm(forms.Form):
    name = forms.CharField(max_length=128)
    min_price = MoneyField(decimal_places=2, default_currency='PLN', max_digits=7)
    max_price = MoneyField(decimal_places=2, default_currency='PLN', max_digits=7)

    def is_valid_custom(self):
        cd = self.cleaned_data

        name = cd.get("name")
        min_price = cd.get("min_price")
        max_price = cd.get("max_price")

        if len(name) < 128 and 0 <= min_price.amount and min_price.amount <= max_price.amount:
            return True
        return False
