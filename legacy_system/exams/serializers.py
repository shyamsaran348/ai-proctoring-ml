from rest_framework import serializers
from django.contrib.auth.models import User
from django.contrib.auth.hashers import make_password
from .models import (
    Problem, TestCase, ExamSession, Submission, TestResult, Language, ProblemLanguage, UserProfile,
    MCQQuestion, MCQSubmission, Contest
)

from django.conf import settings
import os

class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, min_length=6)
    role = serializers.CharField(write_only=True, required=False)
    first_name = serializers.CharField(write_only=True, required=False, allow_blank=True)
    last_name = serializers.CharField(write_only=True, required=False, allow_blank=True)
    aadhar_number = serializers.CharField(write_only=True, required=False, allow_blank=True)
    photo = serializers.ImageField(write_only=True, required=False, allow_null=True)

    class Meta:
        model = User
        fields = ['username', 'email', 'password', 'role', 'first_name', 'last_name', 'aadhar_number', 'photo']
        extra_kwargs = {
            'username': {'required': True},
            'email': {'required': True},
        }

    def validate_username(self, value):
        if User.objects.filter(username__iexact=value).exists():
            raise serializers.ValidationError("A user with this username already exists.")
        return value

    def validate_email(self, value):
        if User.objects.filter(email__iexact=value).exists():
            raise serializers.ValidationError("A user with this email already exists.")
        return value

    def create(self, validated_data):
        role = validated_data.pop('role', None)
        first_name = validated_data.pop('first_name', '')
        last_name = validated_data.pop('last_name', '')
        aadhar_number = validated_data.pop('aadhar_number', '')
        photo = validated_data.pop('photo', None)

        user = User(
            username=validated_data['username'],
            email=validated_data['email'],
            password=make_password(validated_data['password']),
            first_name=first_name,
            last_name=last_name,
        )
        user.save()

        # Create UserProfile automatically
        profile = UserProfile.objects.create(user=user, aadhar_number=aadhar_number)
        if photo:
            # 1. Save to UserProfile (Standard Django)
            profile.avatar = photo
            profile.save()
            
            # 2. Save strict Reference Copy for Proctoring Verification
            # Path: media/students/{username}/reference.jpg
            try:
                import shutil
                reference_dir = os.path.join(settings.MEDIA_ROOT, 'students', user.username)
                os.makedirs(reference_dir, exist_ok=True)
                
                dest_path = os.path.join(reference_dir, 'reference.jpg')
                
                # Robustness: Copy the file that Django just saved to disk
                if profile.avatar and profile.avatar.path:
                    shutil.copy2(profile.avatar.path, dest_path)
                    print(f"[Register] Copied reference photo to: {dest_path}")
                
            except Exception as e:
                print(f"[RegisterError] Failed to save reference photo: {e}")
        
        # Optionally save role inside profile.bio for now
        if role:
            profile.bio = f"Role: {role}"
            profile.save()

        return user

class TestCaseSerializer(serializers.ModelSerializer):
    # expose a computed is_hidden for the frontend, based on test_type
    is_hidden = serializers.SerializerMethodField()

    class Meta:
        model = TestCase
        fields = ['id', 'name', 'test_type', 'input_data', 'expected_output', 'is_hidden', 'order']

    def get_is_hidden(self, obj):
        return obj.test_type in ('private', 'stress')


class ProblemSerializer(serializers.ModelSerializer):
    test_cases = TestCaseSerializer(many=True, required=False) # Changed from read_only=True
    # provide an initial_code fallback (JS default). You can expand to per-language later.
    initial_code = serializers.SerializerMethodField(read_only=True)
    language_specific_code = serializers.SerializerMethodField(read_only=True)
    time_limit = serializers.IntegerField(source='time_limit_seconds', required=False)

    class Meta:
        model = Problem
        fields = [
            'id', 'title', 'description', 'problem_statement', 'input_format', 'output_format',
            'constraints', 'sample_input', 'sample_output', 'explanation',
            'difficulty', 'points', 'time_limit', 'test_cases', 'initial_code', 'language_specific_code',
            'time_limit_seconds'
        ]
        extra_kwargs = {'time_limit_seconds': {'write_only': True, 'required': False}}

    def create(self, validated_data):
        test_cases_data = validated_data.pop('test_cases', [])
        problem = Problem.objects.create(**validated_data)
        for tc_data in test_cases_data:
            TestCase.objects.create(problem=problem, **tc_data)
        return problem

    def update(self, instance, validated_data):
        test_cases_data = validated_data.pop('test_cases', None)
        instance = super().update(instance, validated_data)
        
        if test_cases_data is not None:
            # Simple sync: delete old and create new (or match by ID if we want to be fancy)
            instance.test_cases.all().delete()
            for tc_data in test_cases_data:
                TestCase.objects.create(problem=instance, **tc_data)
        
        return instance


    def get_initial_code(self, obj):
        # Try to find a ProblemLanguage starter code for javascript, else fallback
        try:
            lang = Language.objects.get(name__iexact='javascript')
            pl = ProblemLanguage.objects.filter(problem=obj, language=lang).first()
            if pl and pl.starter_code:
                return pl.starter_code
        except Language.DoesNotExist:
            pass
        return """\
// Write your code below. Implement a function named `solve`.
function solve(...args) {
  // your logic
  return null;
}
"""

    def get_language_specific_code(self, obj):
        """Return a dictionary of language-specific starter code"""
        language_codes = {}
        
        # Get all ProblemLanguage entries for this problem
        problem_languages = ProblemLanguage.objects.filter(problem=obj)
        
        for pl in problem_languages:
            if pl.starter_code:
                language_codes[pl.language.name] = pl.starter_code
        
        # Add default codes for common languages if not present
        if 'javascript' not in language_codes:
            language_codes['javascript'] = """\
// Write your code below. Implement a function named `solve`.
function solve(...args) {
  // your logic
  return null;
}
"""
        
        if 'python' not in language_codes:
            language_codes['python'] = """\
# Write your code below. Implement a function named `solve`.
def solve(*args):
    # your logic
    return None
"""
        
        return language_codes


class ExamSessionSerializer(serializers.ModelSerializer):
    problem = ProblemSerializer(read_only=True)
    problem_id = serializers.IntegerField(write_only=True)
    contest_id = serializers.IntegerField(source='contest.id', read_only=True, required=False)
    problems = serializers.SerializerMethodField(read_only=True)

    class Meta:
        model = ExamSession
        fields = [
            'id', 'session_id', 'problem', 'problem_id', 'contest_id', 'problems', 'start_time', 'end_time',
            'time_remaining', 'status', 'is_submitted'
        ]
        read_only_fields = ['id', 'session_id', 'start_time', 'end_time', 'status', 'is_submitted']

    def get_problems(self, obj):
        if obj.contest:
            return [{'id': p.id, 'title': p.title} for p in obj.contest.problems.all()]
        return []


class TestResultSerializer(serializers.ModelSerializer):
    test_case_name = serializers.CharField(source='test_case.name', read_only=True)

    class Meta:
        model = TestResult
        fields = [
            'id', 'test_case_name', 'input_data', 'expected_output', 'actual_output',
            'is_passed', 'execution_time', 'error_message'
        ]


class SubmissionSerializer(serializers.ModelSerializer):
    test_results = TestResultSerializer(many=True, read_only=True)
    exam_session = ExamSessionSerializer(read_only=True)
    language = serializers.CharField()  # accept language "name" string from client

    class Meta:
        model = Submission
        fields = [
            'id', 'exam_session', 'problem', 'contest', 'code', 'language',
            'status', 'test_results', 'execution_time', 'memory_used',
            'error_message', 'submitted_at', 'score', 'max_score'
        ]
        read_only_fields = [
            'id', 'status', 'test_results', 'execution_time', 'memory_used',
            'error_message', 'submitted_at', 'score', 'max_score', 'problem', 'contest'
        ]


class CodeExecutionSerializer(serializers.Serializer):
    code = serializers.CharField()
    language = serializers.CharField(default='javascript')
    test_cases = serializers.ListField(child=serializers.DictField(), required=False)


class CodeExecutionResponseSerializer(serializers.Serializer):
    success = serializers.BooleanField()
    results = serializers.ListField(child=serializers.DictField())
    execution_time = serializers.FloatField()
    error_message = serializers.CharField(required=False)
class MCQQuestionSerializer(serializers.ModelSerializer):
    class Meta:
        model = MCQQuestion
        fields = ['id', 'contest', 'question_text', 'options', 'correct_option', 'marks', 'order', 'created_at']
        read_only_fields = ['created_at']

class ContestSerializer(serializers.ModelSerializer):
    class Meta:
        model = Contest
        fields = [
            'id', 'title', 'description', 'rules', 'start_time', 'end_time', 
            'duration_minutes', 'status', 'contest_type', 'visibility',
            'enable_proctoring', 'require_webcam', 'require_microphone', 'lock_browser',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['created_at', 'updated_at']
