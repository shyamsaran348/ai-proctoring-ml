from django.http import JsonResponse, StreamingHttpResponse
import base64
from django.core.files.base import ContentFile

from django.shortcuts import redirect, get_object_or_404
from django.contrib.auth.models import User
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.contrib.auth import authenticate, login
from django.contrib.auth.hashers import make_password
from django.contrib.auth.decorators import login_required
from django.views import View

from django.conf import settings
from rest_framework import status
from rest_framework import viewsets, status
from rest_framework.decorators import action, api_view, parser_classes
from rest_framework.parsers import JSONParser, MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework.permissions import AllowAny, IsAuthenticated, IsAdminUser
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

from .models import Problem, TestCase, ExamSession, Submission, TestResult, Language, ProctoringRecord, MCQQuestion, MCQSubmission, Contest


from .serializers import (
    ProblemSerializer, TestCaseSerializer, ExamSessionSerializer,
    SubmissionSerializer, CodeExecutionSerializer, CodeExecutionResponseSerializer,
    RegisterSerializer, MCQQuestionSerializer, ContestSerializer
)

from .services.capture_reference import capture_reference_image
from .services.generate_embedding import generate_and_store_embedding
from .services.verify_identity import verify_identity
# from .services.proctoring_system import ProctoringSystem <-- DEPRECATED
from .services.ml_adapter import MLProctoringAdapter as ProctoringSystem # Adapter Pattern

PROCTORING_INSTANCES = {}
VERIFICATION_STATUS = {}
LAST_CLEANUP = 0

def cleanup_stale_sessions():
    global LAST_CLEANUP
    now = time.time()
    if now - LAST_CLEANUP < 60: # Only run once per minute
        return
    
    LAST_CLEANUP = now
    stale_ids = []
    for sid, proctor in PROCTORING_INSTANCES.items():
        # Check if the proctor has a last_update timestamp
        status = getattr(proctor, 'latest_status', {})
        last_upd = status.get('last_update', 0)
        if now - last_upd > 600: # 10 minutes
            stale_ids.append(sid)
    
    for sid in stale_ids:
        print(f"[Cleanup] Removing stale proctoring session: {sid}")
        PROCTORING_INSTANCES.pop(sid, None)

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


# For SSE throttling
LAST_ALERT_TIME = {} # {session_id: timestamp}

@api_view(['GET'])
def proctoring_pulse(request):
    """
    SSE stream providing real-time risk alerts for faculty.
    Yields data when risk_score > 0.7 for an active session.
    """
    if not request.user.is_staff:
         return Response({'error': 'Unauthorized'}, status=403)

    def event_generator():
        while True:
            # Check all active sessions
            alerts = []
            now = time.time()
            
            for sid, adapter in PROCTORING_INSTANCES.items():
                status = adapter.latest_status
                risk = status.get('risk_score', 0)
                
                if risk > 0.7:
                    # Throttle: 1 alert per 5s per session (Phase 20)
                    last_time = LAST_ALERT_TIME.get(sid, 0)
                    if now - last_time > 5:
                        alerts.append({
                            'session_id': sid,
                            'student': status.get('student_id', 'Unknown'),
                            'risk': round(float(risk), 4),
                            'primary_violation': status.get('violation_type', 'Suspicious Behavior'),
                            'metrics': {
                                'sim': status.get('uc1_identity_sim', 0),
                                'presence': status.get('uc3_presence', 0.5),
                                'audio': status.get('uc6_audio', 0.5),
                                'drift': status.get('uc4_drift', 0),
                                'gaze': status.get('gam_gaze', 0.5),
                                'hgdm': status.get('hgdm_prob', 0.5)
                            }
                        })
                        LAST_ALERT_TIME[sid] = now
            
            if alerts:
                # Yield JSON SSE format
                yield f"event: risk_alert\ndata: {json.dumps(alerts)}\n\n"
            else:
                # Heartbeat to keep connection alive
                yield ": heartbeat\n\n"
                
            time.sleep(2) # Pulse every 2 seconds

    response = StreamingHttpResponse(event_generator(), content_type='text/event-stream')
    response['Cache-Control'] = 'no-cache'
    return response


class ProblemViewSet(viewsets.ModelViewSet):

    queryset = Problem.objects.all()
    serializer_class = ProblemSerializer
    permission_classes = [AllowAny]
    authentication_classes = [NoCSRFSessionAuthentication]

    def get_queryset(self):
        if self.request.user.is_staff:
            return Problem.objects.all()
        return Problem.objects.filter(is_active=True)

    
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

class MCQQuestionViewSet(viewsets.ModelViewSet):
    queryset = MCQQuestion.objects.all()
    serializer_class = MCQQuestionSerializer
    permission_classes = [AllowAny] # Enforce staff checks in create/update
    authentication_classes = [NoCSRFSessionAuthentication]

    def get_queryset(self):
        contest_id = self.request.query_params.get('contest_id')
        if contest_id:
            return MCQQuestion.objects.filter(contest_id=contest_id).order_by('order')
        return MCQQuestion.objects.all().order_by('-created_at')


class ContestViewSet(viewsets.ModelViewSet):
    queryset = Contest.objects.all()
    serializer_class = ContestSerializer
    permission_classes = [AllowAny] # In production, restrict to staff/creator
    authentication_classes = [NoCSRFSessionAuthentication]

    def get_queryset(self):
        user = self.request.user
        if not user or not user.is_authenticated:
            # Fallback to query param for development/testing
            student_id = self.request.query_params.get('student_id')
            if student_id:
                user = User.objects.filter(username=student_id).first() or User.objects.filter(email=student_id).first()
        
        if not user or not user.is_authenticated:
            return Contest.objects.none()
            
        if user.is_staff:
            return Contest.objects.all().order_by('-start_time')
            
        # Standard students: return active contests they are participant in OR public contests
        from django.db.models import Q
        return Contest.objects.filter(
            Q(participants__user=user) | Q(visibility='public'),
            status='active'
        ).distinct().order_by('-start_time')

    @action(detail=True, methods=['get'])
    def exam_designer(self, request, pk=None):
        """API for faculty to get all questions and which are currently in the contest"""
        if not request.user.is_staff:
             return Response({'error': 'Unauthorized'}, status=403)
             
        contest = self.get_object()
        from .models import Problem, MCQQuestion, ContestProblem
        from django.db.models import Q
        
        # 1. Fetch Global Bank
        all_problems = Problem.objects.all().values('id', 'title', 'difficulty')
        all_mcqs = MCQQuestion.objects.filter(Q(contest=contest) | Q(contest__isnull=True)).values('id', 'question_text', 'marks', 'order', 'contest_id')
        
        # 2. Fetch Current Assignments
        current_problems = ContestProblem.objects.filter(contest=contest).order_by('order').values('problem_id', 'order', 'time_limit_override')
        current_mcqs = MCQQuestion.objects.filter(contest=contest).order_by('order').values('id', 'order')

        return Response({
            'all_problems': list(all_problems),
            'all_mcqs': list(all_mcqs),
            'current_problems': list(current_problems),
            'current_mcqs': list(current_mcqs)
        })

    @action(detail=True, methods=['post'])
    def update_structure(self, request, pk=None):
        """Save new order and selection for a contest"""
        if not request.user.is_staff:
             return Response({'error': 'Unauthorized'}, status=403)
             
        contest = self.get_object()
        problems_data = request.data.get('problems', [])
        mcqs_data = request.data.get('mcqs', [])
        
        from .models import ContestProblem, MCQQuestion, Problem
        
        # 1. Update Problems
        ContestProblem.objects.filter(contest=contest).delete()
        for p in problems_data:
             prob = Problem.objects.get(id=p['id'])
             ContestProblem.objects.create(
                 contest=contest,
                 problem=prob,
                 order=p.get('order', 0),
                 time_limit_override=p.get('limit')
             )
             
        # 2. Update MCQs
        assigned_ids = [m['id'] for m in mcqs_data]
        MCQQuestion.objects.filter(contest=contest).exclude(id__in=assigned_ids).update(contest=None, order=0)
        
        for m in mcqs_data:
             MCQQuestion.objects.filter(id=m['id']).update(
                 contest=contest, 
                 order=m.get('order', 0)
             )
             
        return Response({'ok': True})

    @action(detail=True, methods=['post'])
    def start_exam(self, request, pk=None):
        """Start a secure proctored contest session for a student"""
        contest = self.get_object()
        student_id = request.data.get('student_id') or (request.user.username if request.user and request.user.is_authenticated else None)
        if not student_id:
            return Response({'error': 'student_id is required'}, status=status.HTTP_400_BAD_REQUEST)
            
        user = User.objects.filter(username=student_id).first() or User.objects.filter(email=student_id).first()
        if not user:
            return Response({'error': f'Student "{student_id}" not found'}, status=status.HTTP_404_NOT_FOUND)
            
        # Enrollment Gate
        if contest.visibility in ['invite_only', 'private']:
            is_enrolled = contest.participants.filter(user=user, is_active=True).exists()
            if not is_enrolled:
                return Response({'error': 'You are not assigned to this secure assessment.'}, status=status.HTTP_403_FORBIDDEN)
                
        # Check completed sessions
        completed = ExamSession.objects.filter(contest=contest, student=user, status='completed').exists()
        if completed:
            return Response({
                'error': 'You have already completed this secure exam.',
                'completed': True
            }, status=status.HTTP_400_BAD_REQUEST)
            
        # Get or create active session
        session, created = ExamSession.objects.get_or_create(
            contest=contest,
            student=user,
            status='active',
            defaults={
                'session_id': str(uuid.uuid4()),
                'time_remaining': contest.duration_minutes * 60,
                'problem': contest.problems.all().first()
            }
        )
        
        VERIFICATION_STATUS[session.session_id] = 'pending_frontend'
        
        return Response({
            'session_id': session.session_id,
            'contest_id': contest.id,
            'time_remaining': session.time_remaining,
            'problems': [
                {'problem_id': p.id, 'title': p.title} for p in contest.problems.all()
            ]
        }, status=status.HTTP_200_OK)

    @action(detail=True, methods=['post'])
    def enroll_students(self, request, pk=None):
        """Enroll specific candidates (students) to this private assessment"""
        if not request.user.is_staff:
             return Response({'error': 'Unauthorized'}, status=403)
             
        contest = self.get_object()
        student_ids = request.data.get('student_ids', [])
        
        from .models import ContestParticipant
        
        enrolled_count = 0
        for s_id in student_ids:
            user = User.objects.filter(username=s_id).first() or User.objects.filter(email=s_id).first()
            if user:
                ContestParticipant.objects.get_or_create(contest=contest, user=user)
                enrolled_count += 1
                
        return Response({'ok': True, 'enrolled_count': enrolled_count})

class ExamSessionViewSet(viewsets.ModelViewSet):
    queryset = ExamSession.objects.all()
    serializer_class = ExamSessionSerializer
    lookup_field = 'session_id'

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

        finalize = request.data.get('finalize', False)
        code = request.data.get('code')
        lang_name = request.data.get('language', 'javascript')
        
        if not code and not finalize:
            return Response({'error': 'Code is required'}, status=status.HTTP_400_BAD_REQUEST)
        
        submission_data = None
        
        if code:
            # Debug logging
            print(f"Submit request - Session: {session_id}, Language: {lang_name}, Code length: {len(code)}")

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
            
            # Resolve specific problem
            problem_id = request.data.get('problem_id')
            problem_obj = None
            if problem_id:
                try:
                    from .models import Problem
                    problem_obj = Problem.objects.get(id=problem_id)
                except Problem.DoesNotExist:
                    pass
            if not problem_obj:
                problem_obj = exam_session.problem
            
            try:
                submission = Submission.objects.create(
                    exam_session=exam_session,
                    user=user_obj,  # Link to user if available
                    problem=problem_obj,
                    contest=exam_session.contest,
                    code=code,
                    language=lang_obj,
                    status='running',
                    max_score=problem_obj.points if problem_obj else 0
                )
            except Exception as e:
                return Response({'error': f'Failed to create submission: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)

            # Build test data
            try:
                test_cases = problem_obj.test_cases.filter(is_active=True).order_by('order', 'id')
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
            
            submission_data = SubmissionSerializer(submission).data

        # Finalize the entire session if requested
        if finalize:
            exam_session.is_submitted = True
            exam_session.status = 'completed'
            from django.utils import timezone
            exam_session.end_time = timezone.now()
            exam_session.save()

            # Stop monitoring & Persist ML Metrics
            try:
                proctor = PROCTORING_INSTANCES.pop(exam_session.session_id, None)
                if proctor:
                    proctor.stop_monitoring()
                    # 1. Sync final risk to ProctoringSession
                    final_status = proctor.get_live_status()
                    
                    # 2. Update/Create ProctoringSession record
                    from .models import ProctoringSession
                    proc_data, _ = ProctoringSession.objects.get_or_create(
                        exam_session=exam_session,
                        defaults={'session_id': exam_session.session_id}
                    )
                    proc_data.risk_score = final_status.get('risk_score', 0.0) * 100 # 0-100 scale
                    proc_data.is_flagged = proc_data.risk_score > 75
                    proc_data.face_detection_failures = final_status.get('anomaly_count', 0)
                    proc_data.save()
                    
                    print(f"[ML Cleanup] Session {exam_session.session_id} finalized. Risk: {proc_data.risk_score:.2f}")

            except Exception as e:
                print(f"Warning: Failed to finalize proctoring for session {exam_session.session_id}: {str(e)}")

        if submission_data:
            return Response(submission_data, status=status.HTTP_200_OK)
        return Response({'ok': True, 'session_finalized': True}, status=status.HTTP_200_OK)


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
        student_id = request.data.get('student_id') or (request.user.username if request.user and request.user.is_authenticated else None)
        
        if not student_id:
             return Response({'error': 'student_id is required'}, status=400)
        
        # Link student to session if not already set
        if not session.student:
            student_obj = User.objects.filter(username=student_id).first()
            if student_obj:
                session.student = student_obj
                session.save(update_fields=['student'])

        
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
                # 3. Create Audit Record for Identity Verification
                from .models import ProctoringRecord
                ProctoringRecord.objects.create(
                    session=session,
                    risk_score=0.0,
                    violation_type='identity_verified',
                    frame_path=f"/media/sessions/{student_id}/{session.session_id}/session_ref.jpg",
                    meta_data={'similarity': float(sim), 'method': 'uc1_resnet'}
                )
                
                # 4. Start Proctoring with NEW Reference
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

    @action(detail=False, methods=['get'])
    def check_enrollment(self, request):
        """Check if a student has a registered reference photo."""
        student_id = request.query_params.get('student_id')
        if not student_id:
             return Response({'error': 'student_id required'}, status=400)
             
        student_ref_dir = os.path.join(settings.MEDIA_ROOT, 'students', student_id)
        reg_path = os.path.join(student_ref_dir, 'reference.jpg')
        
        # Also check lowercase to be robust
        exists = os.path.exists(reg_path) or os.path.exists(os.path.join(settings.MEDIA_ROOT, 'students', student_id.lower(), 'reference.jpg'))
        
        return Response({'enrolled': exists})

    @action(detail=False, methods=['post'], permission_classes=[AllowAny])
    def enroll_biometric(self, request):
        """
        Self-service biometric enrollment.
        Saves photo to media/students/{student_id}/reference.jpg
        """
        student_id = request.data.get('student_id')
        image_data = request.data.get('image')
        
        if not student_id or not image_data:
            return Response({'error': 'Missing student_id or image'}, status=400)
            
        try:
            if 'base64,' in image_data:
                image_data = image_data.split('base64,')[1]
            img_bytes = base64.b64decode(image_data)
            
            final_dir = os.path.join(settings.MEDIA_ROOT, 'students', student_id)
            os.makedirs(final_dir, exist_ok=True)
            final_path = os.path.join(final_dir, 'reference.jpg')
            
            with open(final_path, 'wb') as f:
                f.write(img_bytes)
                
            print(f"[Enrollment] Successful: {student_id}")
            return Response({'ok': True, 'message': 'Biometric identity enrolled.'})
        except Exception as e:
            return Response({'error': str(e)}, status=500)

    @action(detail=True, methods=['post'])
    def capture_id_card(self, request, session_id=None):
        """
        [DECOMMISSIONED] Step 2 of MFA: Capture Student ID Card.
        Now AUTO-PASSES for high-velocity proctoring.
        """
        print(f"ID Card Capture: AUTO-PASS triggered for session {session_id}")
        return Response({'verified': True, 'ok': True})

    @action(detail=True, methods=['post'])
    def warn_student(self, request, session_id=None):
        """
        Faculty intervention: send a warning message to the student.
        """
        if not request.user.is_staff:
            return Response({'error': 'Unauthorized'}, status=403)
            
        message = request.data.get('message', 'Please focus on your exam.')
        proctor = PROCTORING_INSTANCES.get(session_id)
        
        if proctor:
            proctor.latest_status['intervention_message'] = message
            proctor.latest_status['intervention_id'] = int(time.time())
            print(f"[Intervention] Warning sent to session {session_id}: {message}")
            return Response({'ok': True})
        else:
            return Response({'error': 'Session not found or inactive'}, status=404)

    @action(detail=True, methods=['post'])
    def submit_mcqs(self, request, session_id=None):
        """
        Handle MCQ submissions. 
        Payload: { "responses": [{"question_id": 1, "selected": 0}, ...] }
        """
        session = self.get_object()
        responses = request.data.get('responses', [])
        
        from .models import MCQQuestion, MCQSubmission
        total_score = 0
        questions_processed = 0
        
        for resp in responses:
            q_id = resp.get('question_id')
            selected = resp.get('selected')
            
            try:
                # If no contest, allow repository-wide lookup
                if session.contest:
                    question = MCQQuestion.objects.get(id=q_id, contest=session.contest)
                else:
                    question = MCQQuestion.objects.get(id=q_id)

                is_correct = (selected == question.correct_option)
                
                MCQSubmission.objects.update_or_create(
                    exam_session=session,
                    question=question,
                    student=session.student if session.student else (request.user if request.user.is_authenticated else None),
                    defaults={'selected_option': selected, 'is_correct': is_correct}
                )

                
                if is_correct: total_score += question.marks
                questions_processed += 1
            except MCQQuestion.DoesNotExist:
                continue

        session.marks_obtained = total_score
        session.save(update_fields=['marks_obtained'])

        # If MCQ-only, finalize session

        if session.contest and session.contest.contest_type == 'mcq':

            session.is_submitted = True
            session.status = 'completed'
            from django.utils import timezone
            session.end_time = timezone.now()
            session.save()
            
            # Stop ML
            proctor = PROCTORING_INSTANCES.pop(session.session_id, None)
            if proctor: proctor.stop_monitoring()

        return Response({'ok': True, 'score': total_score, 'processed': questions_processed})

    @action(detail=True, methods=['get'])
    def proctoring_status(self, request, session_id=None):
        cleanup_stale_sessions() # 🛡️ Industrial Hardening: Purge inactive sessions
        try:
            session = self.get_object()  # ensure exists
        except Exception:
            return Response({'error': 'Session not found'}, status=status.HTTP_404_NOT_FOUND)
        
        # Get live proctoring status
        proctor = PROCTORING_INSTANCES.get(session_id)
        if proctor:
            try:
                live_status = proctor.get_live_status()
                # Add contest type for frontend logic
                live_status['contest_type'] = session.contest.contest_type if session.contest else 'practice'
                
                # Fetch MCQs if it's MCQ/Hybrid (Sorted)
                if session.contest and session.contest.contest_type in ['mcq', 'hybrid']:

                    from .models import MCQQuestion
                    mcqs = MCQQuestion.objects.filter(contest=session.contest).order_by('order').values('id', 'question_text', 'options', 'marks')
                    live_status['mcqs'] = list(mcqs)
                
                # Fetch Problems if it's Coding/Hybrid (Sorted)
                if session.contest and session.contest.contest_type in ['coding', 'hybrid']:

                    from .models import ContestProblem
                    problems = ContestProblem.objects.filter(contest=session.contest).order_by('order').values('problem_id', 'problem__title', 'time_limit_override')
                    live_status['problems'] = list(problems)
                
                return Response(live_status)

            except Exception as e:
                return Response({'error': f'Failed: {str(e)}'}, status=500)
        else:
            return Response({
                'is_active': False,
                'contest_type': session.contest.contest_type if session.contest else 'practice',

                'head_pose': "Session Inactive",
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
        and pushes it to the ML Engine (Asynchronously).
        """
        session = self.get_object()
        frame_data = request.data.get('frame')
        audio_volume = request.data.get('audio_volume', None)
        
        # Touch session before cleanup
        proctor = PROCTORING_INSTANCES.get(session_id)
        if proctor:
            proctor.latest_status['last_update'] = time.time()
            
        cleanup_stale_sessions()
        
        if not frame_data:
            return Response({'error': 'No frame data'}, status=400)
            
        # --- BLACK BOX EVIDENCE LOGIC ---
        # If risk is extremely high, we capture this frame for the audit trail.
        proctor = PROCTORING_INSTANCES.get(session_id)
        if proctor and proctor.latest_status.get('risk_score', 0) > 0.8:
            try:
                # Throttling: only save one evidence frame every 30 seconds
                last_ev = getattr(proctor, '_last_evidence_ts', 0)
                if time.time() - last_ev > 30:
                    if 'base64,' in frame_data:
                        raw_frame = frame_data.split('base64,')[1]
                    else:
                        raw_frame = frame_data
                        
                    ev_bytes = base64.b64decode(raw_frame)
                    ev_name = f"violation_{int(time.time())}.jpg"
                    ev_dir = os.path.join(settings.MEDIA_ROOT, 'sessions', session.student.username if session.student else 'unknown', str(session.session_id))
                    os.makedirs(ev_dir, exist_ok=True)
                    ev_path = os.path.join(ev_dir, ev_name)
                    
                    with open(ev_path, 'wb') as f:
                        f.write(ev_bytes)
                    
                    # Log to ProctoringRecord
                    from .models import ProctoringRecord
                    ProctoringRecord.objects.create(
                        session=session,
                        risk_score=proctor.latest_status.get('risk_score', 0.8),
                        violation_type='high_risk_evidence',
                        frame_path=f"/media/sessions/{session.student.username if session.student else 'unknown'}/{session.session_id}/{ev_name}",
                        meta_data={'description': 'Automatic Black Box Capture due to High Risk'}
                    )
                    proctor._last_evidence_ts = time.time()
                    print(f"[Evidence] Captured high-risk frame for session {session_id}")
            except Exception as e:
                print(f"[Evidence] Failed to capture frame: {e}")
            
        if session.session_id in PROCTORING_INSTANCES:
            proctor = PROCTORING_INSTANCES[session.session_id]
            
            if hasattr(proctor, 'process_external_frame'):
                # ─── Thread Throttling (Logical Hardening) ───
                # Check if AI is busy with previous frame or still initializing
                if getattr(proctor, 'is_processing', False):
                    # Still busy, skip this frame to prevent overload
                    resp = {**proctor.latest_status}
                    resp['busy'] = True
                    return Response(resp)
                    
                # ─── Asynchronous ML Processing (Non-blocking) ───
                # Move heavy CV2/Inference to a background thread
                threading.Thread(
                    target=proctor.process_external_frame, 
                    args=(frame_data,), 
                    kwargs={'audio_volume': audio_volume},
                    daemon=True
                ).start()
                
                # Record update time to prevent cleanup
                proctor.latest_status['last_update'] = time.time()
                
                # ─── SENTINEL PRIME: Automated Enforcement (RELAXED FOR TESTING) ───
                # Support configurable strictness (low, medium, high) from request/params/headers
                strictness = request.data.get('strictness', request.query_params.get('strictness', 'medium')).lower()
                if strictness not in ['low', 'medium', 'high']:
                    strictness = 'medium'

                if strictness == 'low':
                    risk_threshold = 0.98
                    strike_limit = 50
                elif strictness == 'high':
                    risk_threshold = 0.90
                    strike_limit = 15
                else:  # medium
                    risk_threshold = 0.95
                    strike_limit = 30

                smoothed_risk = proctor.latest_status.get('risk_score', 0.0)
                uncertainty = proctor.latest_status.get('uncertainty', 0.0)
                
                # Uncertainty Gate: Suppress strikes if model uncertainty is high (>0.25)
                is_uncertain = uncertainty > 0.25

                if smoothed_risk > risk_threshold and not is_uncertain:
                    if not hasattr(proctor, '_critical_strike_count'):
                        proctor._critical_strike_count = 0
                    proctor._critical_strike_count += 1
                    
                    if proctor._critical_strike_count >= strike_limit:
                        session.last_command = 'TERMINATE'
                        session.save(update_fields=['last_command'])
                        print(f"[Sentinel Prime] AUTO-TERMINATING session {session_id} due to total sustained violation under {strictness} strictness.")
                else:
                    # Decay the strike count slowly to allow for more flexibility
                    if hasattr(proctor, '_critical_strike_count') and proctor._critical_strike_count > 0:
                        proctor._critical_strike_count -= 1

                # ─── Intervention Propagation ───
                # Bundle the faculty's last_command into the real-time response
                with proctor.lock:
                    resp_data = {**proctor.latest_status}
                resp_data['last_command'] = session.last_command
                
                # Clear command once dispatched so it only fires once on student side
                if session.last_command != 'NONE':
                    session.last_command = 'NONE' # Reset to neutral
                    session.save(update_fields=['last_command'])

                return Response(resp_data)
        
        return Response({'is_active': False, 'head_pose': 'Session Error'})

    @action(detail=True, methods=['get'])
    def export_report(self, request, session_id=None, **kwargs):


        """
        Generates a consolidated evidence report for faculty/admin review.
        Aggregates proctoring statistics, violations, and snapshot timeline.
        """
        session = self.get_object()
        records = ProctoringRecord.objects.filter(session=session).order_by('timestamp')
        
        # Aggregation Logic
        stats = {
            'total_violations': 0,
            'violation_counts': {},
            'avg_risk': 0.0,
            'max_risk': 0.0,
            'duration_mins': 0
        }
        
        total_risk = 0
        gallery_records = []
        
        for r in records:
            total_risk += r.risk_score
            stats['max_risk'] = max(stats['max_risk'], r.risk_score)
            
            if r.violation_type:
                stats['total_violations'] += 1
                vtype = r.violation_type
                stats['violation_counts'][vtype] = stats['violation_counts'].get(vtype, 0) + 1
                
                # Add to gallery if it's a violation with a frame
                if r.frame_path:
                    gallery_records.append(r)
            
            # Also add high risk frames to gallery
            elif r.risk_score > 0.6 and r.frame_path:
                 gallery_records.append(r)
        
        if records.count() > 0:
            stats['avg_risk'] = total_risk / records.count()
            
        context = {
            'session': session,
            'student': session.student,
            'contest': session.contest,
            'stats': stats,
            'records': records[:200], # Timeline
            'chart_data': {
                'labels': [r.timestamp.strftime('%H:%M:%S') for r in records],
                'scores': [r.risk_score for r in records]
            },
            'gallery': gallery_records[:24], # 6x4 grid
            'generated_at': time.ctime()
        }
        
        return render(request, 'exams/evidence_report.html', context)

    @action(detail=True, methods=['get'])

    def session_history(self, request, session_id=None):

        session = self.get_object()
        records = session.proctoring_records.all().order_by('timestamp')
        
        history = []
        for r in records:
            history.append({
                "timestamp": r.timestamp.isoformat(),
                "risk_score": r.risk_score,
                "violation": r.violation_type,
                "frame_url": r.frame_path if r.frame_path else None,
                "meta_data": r.meta_data
            })
            
        return Response({
            "session_id": session.session_id,
            "student_id": session.student_id,
            "history": history
        })

    @action(detail=True, methods=['get'])
    def session_history(self, request, session_id=None):
        """
        Returns the full per-frame risk trajectory for this session.
        Used by the investigative audit terminal for deep dives.
        """
        try:
            self.get_object()
        except Exception:
            return Response({'error': 'Session not found'}, status=status.HTTP_404_NOT_FOUND)

        proctor = PROCTORING_INSTANCES.get(session_id)
        if proctor and hasattr(proctor, 'get_risk_history'):
            history = proctor.get_risk_history()
            # Return last 150 points max to keep payloads small
            return Response({'history': history[-150:]})
        return Response({'history': []})

    @action(detail=True, methods=['post'], permission_classes=[IsAdminUser])
    def send_command(self, request, session_id=None):
        """
        Faculty action to push a remote command to the student's terminal.
        Available: PAUSE, RESUME, WARN, TERMINATE
        """
        session = self.get_object()
        command = request.data.get('command', 'NONE').upper()
        if command not in ['PAUSE', 'RESUME', 'WARN', 'TERMINATE', 'NONE']:
            return Response({'error': 'Invalid command'}, status=400)
            
        session.last_command = command
        session.save(update_fields=['last_command'])
        return Response({'status': 'Command queued', 'command': command})


@api_view(['GET'])
@login_required
def dashboard_stats(request):
    """Returns analytics for the faculty command center"""
    if not request.user.is_staff:
        return Response({'error': 'Unauthorized'}, status=403)
    
    from .models import Problem, MCQQuestion, Contest, ExamSession, ProctoringRecord
    from django.utils import timezone
    from datetime import timedelta
    
    now = timezone.now()
    last_24h = now - timedelta(hours=24)
    
    stats = {
        'total_problems': Problem.objects.count(),
        'total_mcqs': MCQQuestion.objects.count(),
        'active_sessions': ExamSession.objects.filter(status='active').count(),
        'verified_students': User.objects.filter(is_staff=False).count(), # Simplification
        'recent_anomalies': ProctoringRecord.objects.filter(timestamp__gte=last_24h, risk_score__gt=0.7).count(),
        'total_contests': Contest.objects.count()
    }
    return Response(stats)

@login_required
def faculty_dashboard(request):
    """View for faculty to manage questions and exams"""
    if not request.user.is_staff:
        from django.shortcuts import render
        return render(request, '403.html', status=403)
    
    from .models import Problem, MCQQuestion, Contest, ExamSession
    problems = Problem.objects.all()
    mcqs = MCQQuestion.objects.all()
    contests = Contest.objects.all().order_by('-start_time')
    
    # Get high-risk sessions
    recent_sessions = ExamSession.objects.filter(is_submitted=True).order_by('-end_time')[:10]
    
    context = {
        'problems': problems,
        'mcqs': mcqs,
        'contests': contests,
        'recent_sessions': recent_sessions,
    }
    return render(request, 'exams/faculty_dashboard.html', context)

@login_required
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
            user = serializer.save()
            # Elevate to staff to grant access to proctoring dashboards
            user.is_staff = True
            user.save()
            return Response({
                "message": "Faculty registered successfully",
                "redirect": "/login"
            }, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def get(self, request):
        return render(request, 'register.html', {'role': 'faculty'})


def frontend_view(request):
    return render(request, 'exams/index.html')


from rest_framework import permissions

class AuthSessionCheckView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    
    def get(self, request):
        """Professional session verification endpoint"""
        user = request.user
        # Check if user is staff (Faculty indicator)
        role = 'faculty' if user.is_staff else 'student'
        return Response({
            'user': {
                'username': user.username,
                'email': user.email,
                'role': role
            }
        })
