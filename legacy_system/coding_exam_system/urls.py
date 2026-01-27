"""
URL configuration for coding_exam_system project.
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.http import HttpResponse

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('exams.urls')),
    path('', include('exams.urls')),
    # Add favicon route to prevent 404 errors
    path('favicon.ico', lambda request: HttpResponse(status=204, content_type='image/x-icon')),
    # MongoDB test endpoint
    path('test-mongo/', lambda request: JsonResponse({'status': 'MongoDB test endpoint'}), name='test_mongo'),
    # URL debug endpoint
    path('debug-urls/', lambda request: JsonResponse({
        'available_urls': [
            '/api/problems/',
            '/api/sessions/',
            '/api/sessions/{id}/submit/',
            '/api/execute/'
        ]
    }), name='debug_urls'),
]

# Serve static and media files in development
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT) 