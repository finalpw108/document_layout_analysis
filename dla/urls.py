from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    # path('test/',views.document_layout_analysis,name='document_layout_analysis'),
    path('home/',views.home,name='home'),
    path('convert/',views.convertPdf,name='convert'),
]