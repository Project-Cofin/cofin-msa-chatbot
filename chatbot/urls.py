from django.conf.urls import url

from chatbot import views

urlpatterns = [
    url(r'upload', views.upload),
    url(r'find-by-detail', views.find_by_detail),
    url(r'find-all', views.find_all),
]