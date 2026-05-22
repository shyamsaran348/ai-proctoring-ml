import os
import uuid
import time
import threading
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'coding_exam_system.settings')
django.setup()

from rest_framework.test import APIClient
from django.conf import settings
if 'testserver' not in settings.ALLOWED_HOSTS:
    settings.ALLOWED_HOSTS.append('testserver')

from django.contrib.auth.models import User
from exams.models import ExamSession, Contest
from exams.views import PROCTORING_INSTANCES

def stress_test_concurrency():
    print("=== CONCURRENCY STRESS TEST START ===")
    client = APIClient()
    
    # 1. Setup Session
    user, _ = User.objects.get_or_create(username="stress_student")
    from django.utils import timezone
    import datetime
    start = timezone.now()
    end = start + datetime.timedelta(hours=2)
    
    contest, _ = Contest.objects.get_or_create(
        title="Stress Exam", 
        defaults={
            'contest_type': 'hybrid',
            'start_time': start,
            'end_time': end,
            'duration_minutes': 120,
            'created_by': user
        }
    )
    
    s_id = str(uuid.uuid4())
    session = ExamSession.objects.create(student=user, contest=contest, session_id=s_id, status='active')
    
    # Initialize Engine but MOCK the processing time to be slow (0.5s)
    from exams.services.ml_adapter import MLProctoringAdapter
    class SlowProctor(MLProctoringAdapter):
        def process_external_frame(self, frame_data, audio_volume=0.0):
            try:
                self.is_processing = True
                time.sleep(0.5) # Simulate heavy ML work
            finally:
                self.is_processing = False

    proctor = SlowProctor()
    PROCTORING_INSTANCES[s_id] = proctor
    
    dummy_frame = "data:image/jpeg;base64,/9j/4AAQSkZJRg=="
    
    print("Hammering the server with 20 frames in 0.1s...")
    start_time = time.time()
    for i in range(20):
        client.post(f'/api/sessions/{s_id}/frame/', {'frame': dummy_frame}, format='json')
    
    duration = time.time() - start_time
    print(f"Hammering took {duration:.2f}s")
    
    # If throttling works, it should HAVE NOT spawned 20 threads.
    # It should have returned immediately for the 19 redundant frames.
    
    print(f"Current Thread Count: {threading.active_count()}")
    
    # Verify that the proctor is still processing the FIRST frame (since 0.1s < 0.5s)
    if proctor.is_processing:
        print("✅ SUCCESS: Thread Throttling Active. Redundant frames were skipped.")
    else:
        print("❌ FAILURE: Thread Throttling failed to catch the Hammer.")

    print("\n=== STRESS TEST COMPLETE ===")

if __name__ == "__main__":
    stress_test_concurrency()
