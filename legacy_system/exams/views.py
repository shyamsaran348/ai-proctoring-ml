from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.shortcuts import redirect
from django.contrib.auth.models import User
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.contrib.auth import authenticate, login
from django.contrib.auth.hashers import make_password
from django.views import View
from django.conf import settings
from rest_framework import status
from rest_framework import viewsets, status
from rest_framework.decorators import action, api_view, parser_classes
from rest_framework.parsers import JSONParser, MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from rest_framework.views import APIView
from rest_framework.authentication import SessionAuthentication
from .models import UserProfile
import json
import time
import uuid
import subprocess
import tempfile
import os
import threading



class NoCSRFSessionAuthentication(SessionAuthentication):
    """
    Custom authentication class that doesn't enforce CSRF for API endpoints
    """
    def enforce_csrf(self, request):
        return  # Skip CSRF enforcement

from .models import Problem, TestCase, ExamSession, Submission, TestResult, Language
from .serializers import (
    ProblemSerializer, TestCaseSerializer, ExamSessionSerializer,
    SubmissionSerializer, CodeExecutionSerializer, CodeExecutionResponseSerializer,
    RegisterSerializer
)
from .services.capture_reference import capture_reference_image
from .services.generate_embedding import generate_and_store_embedding
from .services.verify_identity import verify_identity
# from .services.proctoring_system import ProctoringSystem <-- DEPRECATED
from .services.ml_adapter import MLProctoringAdapter as ProctoringSystem # Adapter Pattern

PROCTORING_INSTANCES = {}
VERIFICATION_STATUS = {}

def _safe_json_loads(s, fallback):
    if s is None:
        return fallback
    if isinstance(s, (list, dict, int, float, bool)):
        return s
    try:
        return json.loads(s)
    except Exception:
        return fallback


class CodeExecutionView(View):
    @method_decorator(csrf_exempt)
    def dispatch(self, *args, **kwargs):
        return super().dispatch(*args, **kwargs)

    def post(self, request):
        try:
            data = json.loads(request.body)
            code = data.get('code', '')
            language = data.get('language', 'javascript')
            test_cases = data.get('test_cases', [])

            if not code:
                return JsonResponse({'error': 'Code is required'}, status=400)

            results = self.execute_code(code, language, test_cases)
            return JsonResponse({'success': True, 'results': results, 'execution_time': 0.1})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    def execute_code(self, code, language, test_cases):
        if language.lower() in ('javascript', 'js', 'node'):
            return self.execute_javascript(code, test_cases)
        elif language.lower() in ('python', 'py'):
            return self.execute_python(code, test_cases)
        # default to JS
        return self.execute_javascript(code, test_cases)

    def execute_javascript(self, code, test_cases):
        results = []
        for tc in test_cases:
            try:
                test_input = _safe_json_loads(tc.get('input_data', tc.get('input')), [])
                expected = _safe_json_loads(tc.get('expected_output', tc.get('expected')), None)
                with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
                    wrapped_code = f"""
{code}

const testInput = {json.dumps(test_input)};
const expectedOutput = {json.dumps(expected)};
(async () => {{
  try {{
    const out = (typeof solve === 'function') ? await solve(...testInput) : null;
    console.log(JSON.stringify({{
      success: true, result: out, expected: expectedOutput, passed: JSON.stringify(out) === JSON.stringify(expectedOutput)
    }}));
  }} catch (error) {{
    console.log(JSON.stringify({{ success: false, error: String(error) }}));
  }}
}})();
"""
                    f.write(wrapped_code)
                    temp_file = f.name

                result = subprocess.run(['node', temp_file], capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    try:
                        output = json.loads(result.stdout.strip().splitlines()[-1])
                        results.append({
                            'name': tc.get('name', 'Test'),
                            'input': test_input, 'expected': expected,
                            'actual': output.get('result'),
                            'passed': bool(output.get('passed')),
                            'error': output.get('error')
                        })
                    except json.JSONDecodeError:
                        results.append({'name': tc.get('name', 'Test'), 'input': test_input,
                                        'expected': expected, 'actual': result.stdout.strip(),
                                        'passed': False, 'error': 'Invalid output format'})
                else:
                    results.append({'name': tc.get('name', 'Test'), 'input': test_input,
                                    'expected': expected, 'actual': None,
                                    'passed': False, 'error': result.stderr.strip()})
                os.unlink(temp_file)
            except subprocess.TimeoutExpired:
                results.append({'name': tc.get('name', 'Test'), 'input': test_input,
                                'expected': expected, 'actual': None, 'passed': False,
                                'error': 'Execution timeout'})
            except Exception as e:
                results.append({'name': tc.get('name', 'Test'), 'input': test_input,
                                'expected': expected, 'actual': None, 'passed': False, 'error': str(e)})
        return results

    def execute_python(self, code, test_cases):
        results = []
        for tc in test_cases:
            try:
                test_input = _safe_json_loads(tc.get('input_data', tc.get('input')), [])
                expected = _safe_json_loads(tc.get('expected_output', tc.get('expected')), None)
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    wrapped_code = f"""
{code}

import json
test_input = {json.dumps(test_input)}
expected_output = {json.dumps(expected)}
try:
    out = solve(*test_input) if 'solve' in globals() else None
    print(json.dumps({{
        "success": True,
        "result": out,
        "expected": expected_output,
        "passed": json.dumps(out, sort_keys=True) == json.dumps(expected_output, sort_keys=True)
    }}))
except Exception as e:
    print(json.dumps({{ "success": False, "error": str(e) }}))
"""
                    f.write(wrapped_code)
                    temp_file = f.name

                result = subprocess.run(['python', temp_file], capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    try:
                        output = json.loads(result.stdout.strip().splitlines()[-1])
                        results.append({
                            'name': tc.get('name', 'Test'),
                            'input': test_input, 'expected': expected,
                            'actual': output.get('result'),
                            'passed': bool(output.get('passed')),
                            'error': output.get('error')
                        })
                    except json.JSONDecodeError:
                        results.append({'name': tc.get('name', 'Test'), 'input': test_input,
                                        'expected': expected, 'actual': result.stdout.strip(),
                                        'passed': False, 'error': 'Invalid output format'})
                else:
                    results.append({'name': tc.get('name', 'Test'), 'input': test_input,
                                    'expected': expected, 'actual': None,
                                    'passed': False, 'error': result.stderr.strip()})
                os.unlink(temp_file)
            except subprocess.TimeoutExpired:
                results.append({'name': tc.get('name', 'Test'), 'input': test_input,
                                'expected': expected, 'actual': None, 'passed': False,
                                'error': 'Execution timeout'})
            except Exception as e:
                results.append({'name': tc.get('name', 'Test'), 'input': test_input,
                                'expected': expected, 'actual': None, 'passed': False, 'error': str(e)})
        return results


class ProblemViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Problem.objects.filter(is_active=True)
    serializer_class = ProblemSerializer
    permission_classes = [AllowAny]
    authentication_classes = [NoCSRFSessionAuthentication]
    
    def list(self, request, *args, **kwargs):
        """Override list to include completion status for each problem"""
        response = super().list(request, *args, **kwargs)
        student_id = request.query_params.get('student_id') or (request.user.username if request.user and request.user.is_authenticated else None)
        
        if student_id:
            user = User.objects.filter(username=student_id).first()
            if user:
                # Get all completed sessions for this user
                completed_problem_ids = set()
                completed_sessions = ExamSession.objects.filter(
                    status='completed',
                    is_submitted=True
                )
                for session in completed_sessions:
                    if session.submissions.filter(user=user).exists():
                        if session.problem:
                            completed_problem_ids.add(session.problem.id)
                
                # Add completion status to each problem
                for problem in response.data:
                    problem['is_completed'] = problem['id'] in completed_problem_ids
            else:
                for problem in response.data:
                    problem['is_completed'] = False
        else:
            for problem in response.data:
                problem['is_completed'] = False
        
        return response

    @action(detail=True, methods=['post'], permission_classes=[AllowAny])
    def start_exam(self, request, pk=None):
        problem = get_object_or_404(Problem, pk=pk)
        student_id = request.data.get('student_id') or (request.user.username if request.user and request.user.is_authenticated else None)
        if not student_id:
            return Response({'error': 'student_id is required'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Check if student has already completed this problem
        user = User.objects.filter(username=student_id).first()
        if user:
            completed_sessions = ExamSession.objects.filter(
                problem=problem,
                status='completed',
                is_submitted=True
            )
            # Check if any completed session belongs to this student (via submissions)
            for session in completed_sessions:
                if session.submissions.filter(user=user).exists():
                    return Response({
                        'error': 'You have already completed this exam',
                        'completed': True
                    }, status=status.HTTP_400_BAD_REQUEST)
        
        # Create session data for serializer
        
        session_data = {
            'problem_id': problem.id,
            'time_remaining': problem.time_limit_seconds * 60,  # Convert to seconds
            'status': 'active'
        }
        
        # Use serializer to create the session
        serializer = ExamSessionSerializer(data=session_data)
        if serializer.is_valid():
            session = serializer.save(
                session_id=str(uuid.uuid4()),
                contest=None  # No contest for practice problems
            )
            # Initialize verification status
            VERIFICATION_STATUS[session.session_id] = 'pending'

            # Run verification in background to avoid blocking HTTP
            # UPDATE: Verification moved to frontend (Webcam Capture)
            VERIFICATION_STATUS[session.session_id] = 'pending_frontend'
            
            data = ExamSessionSerializer(session).data
            data.update({'verification_status': 'pending_frontend'})
            return Response(data, status=status.HTTP_201_CREATED)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class ExamSessionViewSet(viewsets.ModelViewSet):
    queryset = ExamSession.objects.all()
    serializer_class = ExamSessionSerializer
    permission_classes = [AllowAny]
    authentication_classes = [NoCSRFSessionAuthentication]
    lookup_field = 'session_id'  # <-- crucial

    def create(self, request):
        serializer = self.get_serializer(data=request.data)
        if serializer.is_valid():
            session_id = str(uuid.uuid4())
            exam_session = ExamSession.objects.create(
                session_id=session_id,
                problem_id=serializer.validated_data['problem_id'],
                time_remaining=serializer.validated_data.get('time_remaining', 300),
                status='active'
            )
            return Response(ExamSessionSerializer(exam_session).data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=True, methods=['post'])
    def submit(self, request, session_id=None):
        try:
            exam_session = self.get_object()
        except Exception as e:
            return Response({'error': f'Session not found: {str(e)}'}, status=status.HTTP_404_NOT_FOUND)

        code = request.data.get('code')
        lang_name = request.data.get('language', 'javascript')
        if not code:
            return Response({'error': 'Code is required'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Debug logging
        print(f"Submit request - Session: {session_id}, Language: {lang_name}, Code length: {len(code) if code else 0}")

        try:
            lang_obj = Language.objects.get(name__iexact=lang_name)
        except Language.DoesNotExist:
            # Try to create default languages if they don't exist
            if lang_name.lower() in ['javascript', 'js']:
                lang_obj, created = Language.objects.get_or_create(
                    name='javascript',
                    defaults={
                        'display_name': 'JavaScript',
                        'docker_image': 'node:latest',
                        'execute_command': 'node',
                        'file_extension': '.js',
                        'default_code': 'function solve(...args) {\n  return null;\n}'
                    }
                )
            elif lang_name.lower() in ['python', 'py']:
                lang_obj, created = Language.objects.get_or_create(
                    name='python',
                    defaults={
                        'display_name': 'Python',
                        'docker_image': 'python:latest',
                        'execute_command': 'python',
                        'file_extension': '.py',
                        'default_code': 'def solve(*args):\n    return None'
                    }
                )
            else:
                return Response({'error': f'Language "{lang_name}" not supported'}, status=status.HTTP_400_BAD_REQUEST)

        # Get user from request if available
        user_obj = None
        if request.user and request.user.is_authenticated:
            user_obj = request.user
        else:
            # Try to get user from student_id in request data
            student_id = request.data.get('student_id')
            if student_id:
                user_obj = User.objects.filter(username=student_id).first()
        
        try:
            submission = Submission.objects.create(
                exam_session=exam_session,
                user=user_obj,  # Link to user if available
                problem=exam_session.problem,
                contest=exam_session.contest,
                code=code,
                language=lang_obj,
                status='running',
                max_score=exam_session.problem.points if exam_session.problem else 0
            )
        except Exception as e:
            return Response({'error': f'Failed to create submission: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)

        # Build test data
        try:
            test_cases = exam_session.problem.test_cases.filter(is_active=True).order_by('order', 'id')
            test_data = []
            for tc in test_cases:
                test_data.append({
                    'name': tc.name,
                    'input': tc.input_data,
                    'expected': tc.expected_output
                })
        except Exception as e:
            return Response({'error': f'Failed to process test cases: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)

        # Execute
        code_executor = CodeExecutionView()
        results = code_executor.execute_code(code, lang_name, test_data)

        # Aggregate + persist
        passed = sum(1 for r in results if r.get('passed'))
        total = len(results)
        submission.results_summary = results
        submission.status = 'accepted' if passed == total and total > 0 else 'wrong_answer'
        submission.score = int((passed / total) * submission.max_score) if total else 0
        submission.save()

        # Detailed rows
        try:
            for i, result in enumerate(results):
                if i < len(test_cases):
                    tc = test_cases[i]
                    TestResult.objects.create(
                        submission=submission,
                        test_case=tc,
                        input_data=json.dumps(_safe_json_loads(result['input'], [])),
                        expected_output=json.dumps(_safe_json_loads(result['expected'], None)),
                        actual_output=json.dumps(result.get('actual')),
                        is_passed=bool(result.get('passed')),
                        execution_time=None,
                        error_message=str(result.get('error') or '')
                    )
        except Exception as e:
            print(f"Warning: Failed to create test results: {str(e)}")
            # Continue execution even if test results fail

        # Mark session as completed
        exam_session.is_submitted = True
        exam_session.status = 'completed'
        from django.utils import timezone
        exam_session.end_time = timezone.now()
        exam_session.save()

        # Stop background proctoring if running (but keep instance for a while)
        try:
            proctor = PROCTORING_INSTANCES.get(exam_session.session_id)
            if proctor:
                anomalies = proctor.stop_monitoring()
                if isinstance(anomalies, list):
                    print(f"Proctoring anomalies for {exam_session.session_id}: {len(anomalies)}")
                # Mark as completed but don't remove immediately
                proctor.is_completed = True
        except Exception as e:
            print(f"Warning: Failed to stop proctoring for session {exam_session.session_id}: {str(e)}")

        return Response(SubmissionSerializer(submission).data, status=status.HTTP_200_OK)

    @action(detail=True, methods=['get'])
    def verification_status(self, request, session_id=None):
        try:
            self.get_object()  # ensure exists
        except Exception:
            return Response({'error': 'Session not found'}, status=status.HTTP_404_NOT_FOUND)
        status_str = VERIFICATION_STATUS.get(session_id, 'unknown')
        return Response({'verification_status': status_str})

    @action(detail=True, methods=['post'])
    def verify_face_snapshot(self, request, session_id=None):
        """
        Received a snapshot from the frontend.
        1. Save it (Session Reference).
        2. Verify against Registered Profile (Identity Check).
        3. If Pass: Start Proctoring using Session Reference (Webcam vs Webcam).
        """
        import base64
        from django.conf import settings
        
        session = self.get_object()
        student_id = request.data.get('student_id')
        if not student_id:
             return Response({'error': 'student_id is required'}, status=400)
        
        # 1. Get Image Data
        image_data = request.data.get('image')
        if not image_data:
            return Response({'error': 'No image data provided'}, status=400)
            
        try:
            # Decode
            if 'base64,' in image_data:
                image_data = image_data.split('base64,')[1]
            img_bytes = base64.b64decode(image_data)
            
            # Save Session Reference
            # New Structure: media/sessions/{username}/{session_id}/session_ref.jpg
            filename = "session_ref.jpg"
            save_dir = os.path.join(settings.MEDIA_ROOT, 'sessions', student_id, str(session.session_id))
            os.makedirs(save_dir, exist_ok=True)
            session_ref_path = os.path.join(save_dir, filename)
            
            with open(session_ref_path, 'wb') as f:
                f.write(img_bytes)
                
            print(f"Saved session reference: {session_ref_path}")
            
            # --- Initialize ML Engine ---
            from .services.ml_adapter import create_engine
            engine = create_engine()

            # 2. Get Registered Photo (Scrubbed Path)
            # Structure: media/students/{student_id}/reference.jpg
            
            # 1. Try Exact Match First
            student_ref_dir = os.path.join(settings.MEDIA_ROOT, 'students', student_id)
            reg_path = os.path.join(student_ref_dir, 'reference.jpg')
            
            if not os.path.exists(reg_path):
                # 2. Try Walking to find case-insensitive match
                # Often username is "Shyam" but folder is "shyam" or vice versa
                students_root = os.path.join(settings.MEDIA_ROOT, 'students')
                found_dir = None
                if os.path.exists(students_root):
                     for d in os.listdir(students_root):
                         if d.lower() == student_id.lower():
                             found_dir = d
                             break
                
                if found_dir:
                    reg_path = os.path.join(students_root, found_dir, 'reference.jpg')
            
            found_reg = os.path.exists(reg_path)
            
            # --- DEBUG LOGGING ---
            print(f"\n[VERIFY DEBUG]")
            print(f"  > CWD:                 '{os.getcwd()}'")
            print(f"  > MEDIA_ROOT:          '{settings.MEDIA_ROOT}'")
            print(f"  > Incoming Student ID: '{student_id}'")
            print(f"  > Looking for Path:    '{reg_path}'")
            print(f"  > Exists?              {found_reg}")
            if not found_reg:
                try:
                    std_root = os.path.join(settings.MEDIA_ROOT, 'students')
                    print(f"  > Students Dir ({std_root}) Contents: {os.listdir(std_root) if os.path.exists(std_root) else 'DIR NOT FOUND'}")
                except Exception as e:
                    print(f"  > Error listing dir: {e}")
            print("--------------------\n")
            # ---------------------

            if not found_reg:
                print(f"[Verification] ERROR: No reference photo found at {reg_path}")
            
            print(f"Verifying Session Ref {session_ref_path} vs Registered {reg_path}")
            
            verified = False
            sim = 0.0
            
            from .services.ml_adapter import create_engine
            if found_reg:
                emb1 = engine.uc1.get_embedding(reg_path)
                emb2 = engine.uc1.get_embedding(session_ref_path)
                
                if emb1 is not None and emb2 is not None:
                    sim = engine.uc1.compute_similarity(emb1, emb2)
                    verified = sim > 0.45 # Lowered to 0.45 for better user experience
                    print(f"Verification Sim: {sim:.4f} -> {verified}")
                else:
                    print("Face not detected in one of the images.")
                    verified = False
                    return Response({'verified': False, 'similarity': 0.0, 'error': 'No face detected in camera or profile photo.'}, status=200)

            else:
                print("Retrieval Error: No Reference Photo Found for User.")
                # STRICT MODE: If account has no photo, they cannot take the exam.
                verified = False
                return Response({'verified': False, 'similarity': 0.0, 'error': 'Account Reference Photo Missing. Please contact Admin.'}, status=200) 
            
            if verified:
                # 3. Start Proctoring with NEW Reference
                # Important: Use the SESSION REF (Webcam) for proctoring
                proctor = ProctoringSystem()
                proctor.start_monitoring(student_id, image_path=session_ref_path)
                PROCTORING_INSTANCES[session.session_id] = proctor
                
                VERIFICATION_STATUS[session.session_id] = 'verified'
                
                return Response({'verified': True, 'similarity': sim})
            else:
                 VERIFICATION_STATUS[session.session_id] = 'failed'
                 return Response({'verified': False, 'similarity': sim, 'error': 'Face mismatch'}, status=200)

        except Exception as e:
            print(f"Verification Error: {e}")
            return Response({'error': str(e)}, status=500)

    @action(detail=True, methods=['get'])
    def proctoring_status(self, request, session_id=None):
        try:
            self.get_object()  # ensure exists
        except Exception:
            return Response({'error': 'Session not found'}, status=status.HTTP_404_NOT_FOUND)
        
        # Get live proctoring status from the ProctoringSystem instance
        proctor = PROCTORING_INSTANCES.get(session_id)
        if proctor:
            try:
                live_status = proctor.get_live_status()
                return Response(live_status)
            except Exception as e:
                return Response({'error': f'Failed to get proctoring status: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        else:
            # Return a default status for completed sessions
            return Response({
                'num_faces': 0,
                'head_pose': 'Unknown',
                'left_eye_dir': 'Unknown',
                'right_eye_dir': 'Unknown',
                'left_eye_img': '',
                'right_eye_img': '',
                'last_update': time.time(),
                'anomaly_count': 0,
                'face_confidence': 0.0,
                'is_completed': True,
                'monitoring': False,
                'status': 'completed'
            })

    @action(detail=True, methods=['post'])
    def event(self, request, session_id=None):
        """Proctoring/security events from frontend."""
        exam_session = self.get_object()
        payload = request.data or {}
        event_type = payload.get('type', 'unknown')
        description = payload.get('description', '')
        exam_session.add_security_event(event_type, description)
        # lightweight counters
        if event_type == 'tab_switch': exam_session.tab_switches += 1
        if event_type == 'copy_paste': exam_session.copy_paste_attempts += 1
        if event_type == 'right_click': exam_session.right_click_attempts += 1
        exam_session.save()
        return Response({'ok': True})

    @action(detail=True, methods=['post'])
    def receive_frame(self, request, session_id=None):
        """
        Receives a frame from the frontend (Client-Side Proctoring)
        and pushes it to the ML Engine.
        """
        session = self.get_object()
        frame_data = request.data.get('frame')
        
        if not frame_data:
            return Response({'error': 'No frame data'}, status=400)
            
        if session.session_id in PROCTORING_INSTANCES:
             proctor = PROCTORING_INSTANCES[session.session_id]
             # Check if it has the new method (MLAdapter)
             if hasattr(proctor, 'process_external_frame'):
                 # Run in background to not block HTTP response? 
                 # For now run sync to ensure order, it's fast enough on CPU
                 proctor.process_external_frame(frame_data)
                 # Update timestamp to prevent 'connection lost'
                 proctor.latest_status['last_update'] = time.time()
                 # Return the UPDATED status so the frontend can render boxes/risk immediately
                 return Response(proctor.latest_status)
        
        return Response({'status': 'ignored'}, status=200)
        
def index(request):
    return render(request, 'ind.html')

@method_decorator(csrf_exempt, name='dispatch')
class loginview(APIView):
    authentication_classes = [NoCSRFSessionAuthentication]
    permission_classes = [AllowAny]
    parser_classes = [JSONParser, FormParser, MultiPartParser]
    def post(self, request):
        try:
            email = (request.data.get('email') or '').strip()
            password = request.data.get('password')

            if not email or not password:
                return Response({'error': 'Email and password are required'}, status=status.HTTP_400_BAD_REQUEST)

            # Resolve to a User by email first; if not found, try as username
            user_obj = User.objects.filter(email__iexact=email).order_by('id').first()
            if not user_obj:
                user_obj = User.objects.filter(username__iexact=email).order_by('id').first()
            if not user_obj:
                return Response({'error': 'Invalid email or password'}, status=status.HTTP_401_UNAUTHORIZED)

            user = authenticate(request, username=user_obj.username, password=password)

            if user is not None:
                try:
                    is_verified = getattr(user, 'profile', None) and getattr(user.profile, 'is_verified', True)
                except Exception:
                    is_verified = True
                if is_verified:
                    login(request, user)
                    return Response({'message': 'Login successful', 'redirect': '/student_dashboard'}, status=status.HTTP_200_OK)
                else:
                    return Response({'error': 'Account not verified'}, status=status.HTTP_403_FORBIDDEN)
            else:
                return Response({'error': 'Invalid email or password'}, status=status.HTTP_401_UNAUTHORIZED)
        except Exception as e:
            return Response({'error': f'Unexpected error: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def get(self, request):
        return render(request, 'login.html')


@api_view(['GET'])
def test_verification(request):
    """Test endpoint to check verification system status"""
    try:
        # Check if required directories exist
        students_dir = os.path.join(settings.BASE_DIR, 'static', 'uploads', 'students')
        embeddings_dir = os.path.join(settings.BASE_DIR, 'static', 'uploads', 'embeddings')
        
        # Check if DeepFace is available
        try:
            from deepface import DeepFace
            deepface_available = True
        except ImportError:
            deepface_available = False
        
        # Check if OpenCV is available
        try:
            import cv2
            cv2_available = True
        except ImportError:
            cv2_available = False
        
        # Check if directories exist
        students_dir_exists = os.path.exists(students_dir)
        embeddings_dir_exists = os.path.exists(embeddings_dir)
        
        # List registered photos
        registered_photos = []
        if students_dir_exists:
            for file in os.listdir(students_dir):
                if file.endswith('.jpg') and not file.endswith('_reference.jpg'):
                    registered_photos.append(file)
        
        return Response({
            'status': 'ok',
            'deepface_available': deepface_available,
            'cv2_available': cv2_available,
            'students_dir_exists': students_dir_exists,
            'embeddings_dir_exists': embeddings_dir_exists,
            'registered_photos': registered_photos,
            'students_dir': students_dir,
            'embeddings_dir': embeddings_dir
        })
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@method_decorator(csrf_exempt, name='dispatch')   # disable CSRF for this view
class registerview(APIView):
    parser_classes = [MultiPartParser, FormParser]  # allow file upload
    authentication_classes = [NoCSRFSessionAuthentication]
    permission_classes = [AllowAny]

    def post(self, request):
        try:
            serializer = RegisterSerializer(data=request.data)
            if serializer.is_valid():
                serializer.save()
                return Response({
                    "message": "User registered successfully",
                    "redirect": "/verification_pending"
                }, status=status.HTTP_201_CREATED)
            # Return detailed validation errors
            error_message = "Registration failed. Please check the following:"
            if isinstance(serializer.errors, dict):
                error_details = []
                for field, errors in serializer.errors.items():
                    if isinstance(errors, list):
                        error_details.append(f"{field}: {', '.join(str(e) for e in errors)}")
                    else:
                        error_details.append(f"{field}: {errors}")
                error_message = "; ".join(error_details) if error_details else str(serializer.errors)
            return Response({
                "error": error_message,
                "details": serializer.errors
            }, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({
                "error": f"An unexpected error occurred: {str(e)}"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def get(self, request):
        return render(request, 'register.html')


@method_decorator(csrf_exempt, name='dispatch')
class faculty_loginview(APIView):
    authentication_classes = [NoCSRFSessionAuthentication]
    permission_classes = [AllowAny]
    parser_classes = [JSONParser, FormParser, MultiPartParser]
    
    def post(self, request):
        try:
            email = (request.data.get('email') or '').strip()
            password = request.data.get('password')

            if not email or not password:
                return Response({'error': 'Email and password are required'}, status=status.HTTP_400_BAD_REQUEST)

            # Resolve to a User by email first; if not found, try as username
            user_obj = User.objects.filter(email__iexact=email).order_by('id').first()
            if not user_obj:
                user_obj = User.objects.filter(username__iexact=email).order_by('id').first()
            if not user_obj:
                return Response({'error': 'Invalid email or password'}, status=status.HTTP_401_UNAUTHORIZED)

            user = authenticate(request, username=user_obj.username, password=password)

            if user is not None:
                try:
                    is_verified = getattr(user, 'profile', None) and getattr(user.profile, 'is_verified', True)
                except Exception:
                    is_verified = True
                if is_verified:
                    login(request, user)
                    # Faculty redirect - can be customized later
                    return Response({'message': 'Login successful', 'redirect': '/student_dashboard'}, status=status.HTTP_200_OK)
                else:
                    return Response({'error': 'Account not verified'}, status=status.HTTP_403_FORBIDDEN)
            else:
                return Response({'error': 'Invalid email or password'}, status=status.HTTP_401_UNAUTHORIZED)
        except Exception as e:
            return Response({'error': f'Unexpected error: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def get(self, request):
        return render(request, 'login.html', {'role': 'faculty'})


@method_decorator(csrf_exempt, name='dispatch')
class faculty_registerview(APIView):
    parser_classes = [MultiPartParser, FormParser]
    authentication_classes = [NoCSRFSessionAuthentication]
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = RegisterSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response({
                "message": "Faculty registered successfully",
                "redirect": "/verification_pending"
            }, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def get(self, request):
        return render(request, 'register.html', {'role': 'faculty'})


def frontend_view(request):
    return render(request, 'exams/index.html')
