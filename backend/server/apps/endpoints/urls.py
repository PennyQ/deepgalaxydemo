from django.conf.urls import url, include
from django.urls import path

from .views import HomePageView, CreatePostView


urlpatterns = [
    path('', HomePageView.as_view(), name='home'),
    path('post/', CreatePostView.as_view(), name='add_post') # new
]
