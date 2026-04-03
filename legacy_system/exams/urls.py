from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'problems', views.ProblemViewSet, basename='problems')
router.register(r'sessions', views.ExamSessionViewSet, basename='sessions')
router.register(r'mcqs', views.MCQQuestionViewSet, basename='mcqs')
router.register(r'contests', views.ContestViewSet, basename='contests')

#router.register(r'submissions', views.SubmissionViewSet, basename='submissions')

urlpatterns = [
    path('', views.index, name='index'),
    path('login/', views.loginview.as_view(), name='login'),
    path('register/', views.registerview.as_view(), name='register'),
    path('faculty-login/', views.faculty_loginview.as_view(), name='faculty_login'),
    path('faculty-register/', views.faculty_registerview.as_view(), name='faculty_register'),
    path('login/', views.loginview.as_view(), name='api_login'),
    path('register/', views.registerview.as_view(), name='api_register'),
    path('faculty-login/', views.faculty_loginview.as_view(), name='api_faculty_login'),
    path('faculty-register/', views.faculty_registerview.as_view(), name='api_faculty_register'),
    path('student_dashboard', views.frontend_view, name='frontend'),
    path('', include(router.urls)),
    path('execute/', views.CodeExecutionView.as_view(), name='api_code_execute'),
    path('sessions/<session_id>/submit/', views.ExamSessionViewSet.as_view({'post': 'submit'}), name='api_session_submit'),
    path('sessions/<session_id>/event/', views.ExamSessionViewSet.as_view({'post': 'event'}), name='api_session_event'),
    path('sessions/<session_id>/', views.ExamSessionViewSet.as_view({'get': 'retrieve'}), name='api_session_retrieve'),
    path('sessions/<session_id>/verification/', views.ExamSessionViewSet.as_view({'get': 'verification_status'}), name='api_session_verification'),
    path('sessions/<session_id>/proctoring/', views.ExamSessionViewSet.as_view({'get': 'proctoring_status'}), name='api_session_proctoring'),
    
    # NEW: Endpoint for receiving frames from Frontend
    path('sessions/<session_id>/frame/', views.ExamSessionViewSet.as_view({'post': 'receive_frame'}), name='api_session_frame'),

    path('faculty/', views.faculty_dashboard, name='faculty_dashboard'),
    path('sessions/<session_id>/submit_mcqs/', views.ExamSessionViewSet.as_view({'post': 'submit_mcqs'}), name='api_session_submit_mcqs'),
    path('test-verification/', views.test_verification, name='test_verification'),
    path('dashboard_stats/', views.dashboard_stats, name='dashboard_stats'),
    path('proctoring_pulse/', views.proctoring_pulse, name='proctoring_pulse'),
    path('auth/session-check/', views.AuthSessionCheckView.as_view(), name='api_session_check'),
]

