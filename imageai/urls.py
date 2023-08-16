from django.urls import path
from .views import ImageAiViewSet

urlpatterns = [
    path('identify', ImageAiViewSet.as_view({'post': 'create'})),
    path('identify', ImageAiViewSet.as_view({'get': 'get'}))
]