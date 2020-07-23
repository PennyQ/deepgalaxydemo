# from django.conf.urls import url, include
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static # new

from apps.endpoints.urls import urlpatterns as endpoints_urlpatterns

# import settings
# from django.conf.urls import url

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('apps.endpoints.urls')), # new
    # url(r'^media/(?P<path>.*)$', 'django.views.static.serve', {'document_root': settings.MEDIA_ROOT}),
]

urlpatterns += endpoints_urlpatterns

if settings.DEBUG: # new
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)