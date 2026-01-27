from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'problems', views.ProblemViewSet, basename='problems')
router.register(r'sessions', views.ExamSessionViewSet, basename='sessions')
#router.register(r'submissions', views.SubmissionViewSet, basename='submissions')

urlpatterns = [
    path('', views.index, name='index'),
    path('login/', views.loginview.as_view(), name='login'),
    path('register/', views.registerview.as_view(), name='register'),
    path('faculty-login/', views.faculty_loginview.as_view(), name='faculty_login'),
    path('faculty-register/', views.faculty_registerview.as_view(), name='faculty_register'),
    path('api/login/', views.loginview.as_view(), name='api_login'),
    path('api/register/', views.registerview.as_view(), name='api_register'),
    path('api/faculty-login/', views.faculty_loginview.as_view(), name='api_faculty_login'),
    path('api/faculty-register/', views.faculty_registerview.as_view(), name='api_faculty_register'),
    path('student_dashboard', views.frontend_view, name='frontend'),
    path('api/', include(router.urls)),
    path('api/execute/', views.CodeExecutionView.as_view(), name='api_code_execute'),
    path('api/sessions/<session_id>/submit/', views.ExamSessionViewSet.as_view({'post': 'submit'}), name='api_session_submit'),
    path('api/sessions/<session_id>/event/', views.ExamSessionViewSet.as_view({'post': 'event'}), name='api_session_event'),
    path('api/sessions/<session_id>/', views.ExamSessionViewSet.as_view({'get': 'retrieve'}), name='api_session_retrieve'),
    path('api/sessions/<session_id>/verification/', views.ExamSessionViewSet.as_view({'get': 'verification_status'}), name='api_session_verification'),
    path('api/sessions/<session_id>/proctoring/', views.ExamSessionViewSet.as_view({'get': 'proctoring_status'}), name='api_session_proctoring'),
    
    # NEW: Endpoint for receiving frames from Frontend
    path('api/sessions/<session_id>/frame/', views.ExamSessionViewSet.as_view({'post': 'receive_frame'}), name='api_session_frame'),

    path('api/test-verification/', views.test_verification, name='test_verification'),
]
