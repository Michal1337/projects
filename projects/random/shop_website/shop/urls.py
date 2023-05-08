from django.urls import path
from . import views

urlpatterns = [
    path('home/', views.home_view, name='home'),
    path('user/<str:username>/', views.user_view, name='user'),
    path('product/<int:product_id>/', views.product_view, name='product'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('register/', views.register_view, name='register'),
    path('activate/<uidb64>/<token>/', views.activate, name='activate'),
    path('password_reset/', views.password_reset_request_view, name='password_reset'),
    path('reset/<uidb64>/<token>/', views.passwordResetConfirm, name='password_reset_confirm'),
    path('user/<str:username>/addProduct/', views.addproduct_view, name='addproduct'),
    path('user/<str:username>/cart/', views.cart_view, name='cart'),
    path('search/', views.search_view, name='search')
]