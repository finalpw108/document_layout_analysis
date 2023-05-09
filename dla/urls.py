from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    # path('test/',views.document_layout_analysis,name='document_layout_analysis'),
    path('home/',views.home,name='home'),
    # path('convert/',views.convertPdf,name='convert'),
    path('signup/',views.signup,name='signup'),
    path('login/',views.login,name="login"),
    path('logout/',views.logout,name="logout"),
    path('dashboard/',views.dashboard,name="dashboard"),
    path('sectionPages/<jsonFile>/<int:id>/',views.sectionPages,name="sectionPages"),
]