from django.contrib import admin
from django.urls import path
from recommender import views

urlpatterns = [
    path("login" ,views.login,name="login"),
    path("" ,views.home,name="home"),
    path("home" ,views.home,name="home"),
    path("bodymass",views.bodymass,name="bodymass"),
    path("dietplanner",views.index,name="dietplanner"),
    path("diet",views.diet,name="diet"),
    path("detect",views.detect,name="detect"),
    path('video_feed', views.video_feed, name='video_feed')
]