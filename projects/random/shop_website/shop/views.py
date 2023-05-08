import datetime as dt
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import PasswordResetForm, SetPasswordForm
from django.contrib.sites.shortcuts import get_current_site
from django.core.mail import EmailMessage
from django.db.models.query import Q
from django.shortcuts import render, redirect
from django.template.loader import render_to_string
from django.utils.encoding import force_bytes, force_str
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.db.models import Sum
from .forms import *
from .models import *
from .tokens import account_activation_token


def activateEmail(request, user, email):
    mail_subject = "Activate your shop account"
    message = render_to_string("activate_account.html", {
        'user': user.username,
        'domain': get_current_site(request).domain,
        'uid': urlsafe_base64_encode(force_bytes(user.pk)),
        'token': account_activation_token.make_token(user),
        'protocol': 'https' if request.is_secure() else 'http'
    })
    email = EmailMessage(mail_subject, message, to=[email])
    if email.send():
        messages.success(request, "email sent")
    else:
        messages.error(request, "error, not sent")


def activate(request, uidb64, token):
    try:
        uid = force_str(urlsafe_base64_decode(uidb64))
        user = User.objects.get(pk=uid)
    except:
        user = None

    if user is not None and account_activation_token.check_token(user, token):
        user.is_active = True
        user.save()

    return redirect('login')


def register_view(request):
    form = RegisterForm()
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        context = {'form': form}
        if form.is_valid() and form.is_valid_custom():
            try:
                test_user1 = User.objects.get(username=form.cleaned_data["username"])
                test_user2 = User.objects.get(email=form.cleaned_data["email"])
            except:
                test_user1 = None
                test_user2 = None
            if test_user1 is not None or test_user2 is not None:
                messages.error(request, "Username or email already used")
                return render(request, 'register.html', context=context)
            user = form.save(commit=False)
            user.is_active = False
            user.save()
            activateEmail(request, user, form.cleaned_data.get('email'))
            return redirect('home')

    context = {'form': form}
    return render(request, 'register.html', context=context)


def password_reset_request_view(request):
    form = PasswordResetForm()
    if request.method == 'POST':
        form = PasswordResetForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data['email']
            user = User.objects.filter(Q(email=email)).first()
            if user:
                subject = "Reset password request"
                message = render_to_string("reset_password.html", {
                    'user': user,
                    'domain': get_current_site(request).domain,
                    'uid': urlsafe_base64_encode(force_bytes(user.pk)),
                    'token': account_activation_token.make_token(user),
                    'protocol': 'https' if request.is_secure() else 'http'
                })
                email = EmailMessage(subject, message, to=[user.email])
                if email.send():
                    messages.success(request, "udalo sie, zresetuj haslo")
                else:
                    messages.error(request, "nie udalo sie")
            return redirect('home')

    context = {'form': form}
    return render(request, 'password_reset.html', context=context)


def passwordResetConfirm(request, uidb64, token):
    try:
        uid = force_str(urlsafe_base64_decode(uidb64))
        user = User.objects.get(pk=uid)
    except:
        user = None

    if user is not None and account_activation_token.check_token(user, token):
        if request.method == 'POST':
            form = SetPasswordForm(user, request.POST)
            if form.is_valid():
                form.save()
                messages.success(request, "password changed")
                return redirect('home')
            else:
                for error in list(form.errors.values()):
                    messages.error(request, error)

        form = SetPasswordForm(user)
        context = {'form': form}
        return render(request, 'password_reset_confirm.html', context=context)
    else:
        messages.error(request, "link has expired")

    return redirect('home')

def login_view(request):
    context = {}
    if request.method == "POST":
        username = request.POST['username']
        password = request.POST['password']

        pattern = r"^[A-Za-z0-9]*"
        if re.search(pattern, username) is None:
            return redirect('login')
        user = authenticate(request, username=username, password=password)
        if user is not None and user.is_active:
            login(request, user)
            return redirect('home')
        else:
            messages.success(request, "There was na error!")
            return redirect('login')
    else:
        return render(request, 'login.html', context=context)


@login_required(login_url='http://127.0.0.1:8000/shop/login/')
def logout_view(request):
    if request.method == "POST":
        logout(request)
        return redirect('home')
    return render(request, 'logout.html')


@login_required(login_url='http://127.0.0.1:8000/shop/login/')
def addproduct_view(request, username):
    active_user = User.objects.get(username=username)
    form = AddProductForm()

    if request.method == 'POST':
        form = AddProductForm(request.POST)
        if form.is_valid() and form.is_valid_custom():
            product = Product(user=active_user, name=form.cleaned_data["name"], type=form.cleaned_data["type"],
                              description=form.cleaned_data["description"], image=form.cleaned_data["image"],
                              price=form.cleaned_data["price"], weight=form.cleaned_data["weight"],
                              condition=form.cleaned_data["condition"],
                              available_num=form.cleaned_data["available_num"], available_from=dt.datetime.now())
            product.save()

    context = {'form': form}
    return render(request, 'addproduct.html', context=context)


def home_view(request):
    products = Product.objects.filter(available_num__gt=0).order_by('-price')[:15]
    context = {'list_of_products': products}
    return render(request, 'home.html', context=context)


@login_required(login_url='http://127.0.0.1:8000/shop/login/')
def user_view(request, username):
    user = User.objects.get(username=username)
    active_user = request.user
    products = Product.objects.filter(user=user).order_by('-price')
    comments = Comment.objects.filter(user=user).order_by('-pub_date')

    if request.method == 'POST':
        form = AddCommentForm(request.POST)
        if form.is_valid() and form.is_valid_custom():
            comment = Comment(user=user, author=active_user, title=form.cleaned_data["title"],
                              content=form.cleaned_data["content"], rating=form.cleaned_data["rating"],
                              pub_date=dt.datetime.now())
            comment.save()

    form = AddCommentForm()
    context = {'list_of_products': products, 'list_of_comments': comments, 'form': form, 'user': user}
    return render(request, 'user.html', context=context)


def product_view(request, product_id):
    active_user = request.user
    product = Product.objects.get(id=product_id)
    reviews = Review.objects.filter(product=product).order_by('-pub_date')
    formReview = AddReviewForm()
    formOrder = AddOrderForm()

    if request.method == 'POST' and 'addReview' in request.POST:
        form = AddReviewForm(request.POST)
        if form.is_valid():
            review = Review(product=product, author=active_user, title=form.cleaned_data["title"],
                            content=form.cleaned_data["content"], rating=form.cleaned_data["rating"],
                            pub_date=dt.datetime.now())
            review.save()

    if request.method == 'POST' and 'addToOrder' in request.POST:
        form = AddOrderForm(request.POST)
        if form.is_valid():
            if form.cleaned_data["total_price"] == form.cleaned_data["quantity"] * product.price and form.cleaned_data[
                "quantity"] <= product.available_num:
                order = Order(user=active_user, product=product, quantity=form.cleaned_data["quantity"],
                              total_price=form.cleaned_data["total_price"], date=dt.datetime.now())
                order.save()
                product.available_num -= form.cleaned_data["quantity"]
                product.save()
            else:
                messages.error(request, "There was an error")

    context = {'product': product, 'list_of_reviews': reviews, 'request': request, 'formReview': formReview,
               'formOrder': formOrder}
    return render(request, 'product.html', context=context)

@login_required(login_url='http://127.0.0.1:8000/shop/login/')
def cart_view(request, username):
    user = User.objects.get(username=username)
    list_of_orders = Order.objects.filter(user=user).order_by('-date')
    sum_totalprice = list_of_orders.aggregate(Sum('total_price'))["total_price__sum"]
    context = {'list_of_orders': list_of_orders, 'sum_totalprice': sum_totalprice}
    return render(request, 'cart.html', context=context)


def search_view(request):
    form = SearchForm()
    if request.method == 'POST':
        form = SearchForm(request.POST)
        if form.is_valid() and form.is_valid_custom():
            products = Product.objects.filter(name__contains=form.cleaned_data["name"],
                                              price__gte=form.cleaned_data["min_price"],
                                              price__lte=form.cleaned_data["max_price"], available_num__gt=0).order_by(
                '-price')
            print(products)
            context = {'form': form, 'list_of_products': products}
            return render(request, 'search.html', context=context)

    context = {'form': form}
    return render(request, 'search.html', context=context)
