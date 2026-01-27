from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator, MaxValueValidator
from django.utils import timezone
import json
import uuid
from datetime import timedelta


class Language(models.Model):
    """Programming language configuration"""
    name = models.CharField(max_length=50, unique=True)  # python3, java, cpp, javascript
    display_name = models.CharField(max_length=100)
    docker_image = models.CharField(max_length=200)
    compile_command = models.TextField(blank=True)
    execute_command = models.TextField()
    file_extension = models.CharField(max_length=10)
    default_code = models.TextField()
    is_active = models.BooleanField(default=True)
    
    # Resource limits
    max_execution_time = models.IntegerField(default=10)  # seconds
    max_memory_mb = models.IntegerField(default=512)  # MB
    max_cpu_usage = models.IntegerField(default=100)  # percentage
    
    def __str__(self):
        return self.display_name


class UserProfile(models.Model):
    """Enhanced user profile"""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    bio = models.TextField(blank=True, max_length=500)
    avatar = models.ImageField(upload_to='students/', blank=True, null=True)
    aadhar_number = models.CharField(max_length=20, blank=True, default="")
    
    # Performance metrics
    current_rating = models.IntegerField(default=1200)
    max_rating = models.IntegerField(default=1200)
    rank = models.CharField(max_length=20, default='Unrated')
    
    # Statistics
    total_submissions = models.IntegerField(default=0)
    accepted_submissions = models.IntegerField(default=0)
    problems_solved = models.IntegerField(default=0)
    contests_participated = models.IntegerField(default=0)
    
    # Preferences
    preferred_language = models.ForeignKey(
        Language, on_delete=models.SET_NULL, null=True, blank=True
    )
    
    # Account settings
    email_verified = models.BooleanField(default=False)
    is_premium = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.user.username}'s Profile"
    
    def update_rating(self, new_rating):
        """Update user rating and rank"""
        self.current_rating = new_rating
        if new_rating > self.max_rating:
            self.max_rating = new_rating
        
        # Update rank based on rating
        if self.current_rating < 1000:
            self.rank = 'Newbie'
        elif self.current_rating < 1200:
            self.rank = 'Pupil'
        elif self.current_rating < 1400:
            self.rank = 'Specialist'
        elif self.current_rating < 1600:
            self.rank = 'Expert'
        elif self.current_rating < 1900:
            self.rank = 'Candidate Master'
        elif self.current_rating < 2100:
            self.rank = 'Master'
        elif self.current_rating < 2300:
            self.rank = 'International Master'
        else:
            self.rank = 'Grandmaster'
        
        self.save()
    
    def get_acceptance_rate(self):
        """Calculate acceptance rate"""
        if self.total_submissions == 0:
            return 0.0
        return (self.accepted_submissions / self.total_submissions) * 100


class Contest(models.Model):
    """Enhanced contest model"""
    CONTEST_STATUS = [
        ('upcoming', 'Upcoming'),
        ('active', 'Active'),
        ('completed', 'Completed'),
        ('cancelled', 'Cancelled'),
    ]
    
    CONTEST_TYPE = [
        ('practice', 'Practice'),
        ('rated', 'Rated Contest'),
        ('unrated', 'Unrated Contest'),
        ('virtual', 'Virtual Contest'),
    ]
    
    VISIBILITY = [
        ('public', 'Public'),
        ('private', 'Private'),
        ('invite_only', 'Invite Only'),
    ]
    
    # Basic info
    title = models.CharField(max_length=200)
    description = models.TextField()
    rules = models.TextField(blank=True)
    
    # Timing
    start_time = models.DateTimeField()
    end_time = models.DateTimeField()
    duration_minutes = models.IntegerField()
    
    # Configuration
    status = models.CharField(max_length=20, choices=CONTEST_STATUS, default='upcoming')
    contest_type = models.CharField(max_length=20, choices=CONTEST_TYPE, default='rated')
    visibility = models.CharField(max_length=20, choices=VISIBILITY, default='public')
    
    # Limits
    max_participants = models.IntegerField(default=10000)
    registration_required = models.BooleanField(default=True)
    registration_deadline = models.DateTimeField(null=True, blank=True)
    
    # Languages
    allowed_languages = models.ManyToManyField(Language, blank=True)
    
    # Creator and timestamps
    created_by = models.ForeignKey(User, on_delete=models.CASCADE, related_name='contests_created')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Proctoring settings
    enable_proctoring = models.BooleanField(default=False)
    require_webcam = models.BooleanField(default=False)
    require_microphone = models.BooleanField(default=False)
    lock_browser = models.BooleanField(default=False)
    
    def __str__(self):
        return self.title
    
    @property
    def is_active(self):
        """Check if contest is currently active"""
        now = timezone.now()
        return self.start_time <= now <= self.end_time
    
    @property
    def current_participants_count(self):
        """Get current participant count"""
        return self.participants.filter(is_active=True).count()
    
    def can_register(self, user):
        """Check if user can register for contest"""
        now = timezone.now()
        
        # Check if registration is open
        if self.registration_deadline and now > self.registration_deadline:
            return False, "Registration deadline has passed"
        
        if now > self.start_time:
            return False, "Contest has already started"
        
        # Check if contest is full
        if self.current_participants_count >= self.max_participants:
            return False, "Contest is full"
        
        # Check if already registered
        if self.participants.filter(user=user).exists():
            return False, "Already registered"
        
        return True, "Can register"
    
    class Meta:
        ordering = ['-start_time']


class ProblemCategory(models.Model):
    """Problem categories"""
    name = models.CharField(max_length=50, unique=True)
    description = models.TextField(blank=True)
    color_hex = models.CharField(max_length=7, default='#3498db')
    
    def __str__(self):
        return self.name
    
    class Meta:
        verbose_name_plural = "Problem Categories"


class Problem(models.Model):
    DIFFICULTY_CHOICES = (
        ('easy', 'Easy'),
        ('medium', 'Medium'),
        ('hard', 'Hard'),
    )

    title = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    problem_statement = models.TextField(default="Problem statement not provided")
    input_format = models.TextField(default="Input description not provided")
    output_format = models.TextField(default="Output description not provided")
    constraints = models.TextField(default="No constraints specified")
    sample_input = models.TextField(blank=True)
    sample_output = models.TextField(blank=True)
    explanation = models.TextField(blank=True)
    difficulty = models.CharField(max_length=10, choices=DIFFICULTY_CHOICES, default='easy')
    points = models.IntegerField(default=0)
    time_limit_seconds = models.IntegerField(default=1)
    memory_limit_mb = models.IntegerField(default=256)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True, null=True, blank=True)   # ✅ allow null
    updated_at = models.DateTimeField(auto_now=True, null=True, blank=True)       # ✅ allow null

    def __str__(self):
        return self.title


class ProblemLanguage(models.Model):
    """Language-specific problem configurations"""
    problem = models.ForeignKey(Problem, on_delete=models.CASCADE)
    language = models.ForeignKey(Language, on_delete=models.CASCADE)
    starter_code = models.TextField(default='')
    solution_code = models.TextField(blank=True)
    
    # Language-specific limits (override problem defaults if set)
    time_limit_override = models.IntegerField(null=True, blank=True)
    memory_limit_override = models.IntegerField(null=True, blank=True)
    
    class Meta:
        unique_together = ['problem', 'language']


class TestCase(models.Model):
    TEST_TYPE_CHOICES = (
        ('sample', 'Sample'),
        ('public', 'Public'),
        ('private', 'Private'),
        ('stress', 'Stress'),
    )

    problem = models.ForeignKey(Problem, on_delete=models.CASCADE, related_name="test_cases")
    name = models.CharField(max_length=255)
    test_type = models.CharField(max_length=10, choices=TEST_TYPE_CHOICES, default='public')
    input_data = models.TextField(default="[]")
    expected_output = models.TextField(default="null")
    order = models.IntegerField(default=0)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True, null=True, blank=True)  # ✅ safe
    updated_at = models.DateTimeField(auto_now=True, null=True, blank=True)      # ✅ safe

    @property
    def is_hidden(self):
        return self.test_type in ('private', 'stress')

    def __str__(self):
        return f"{self.problem.title} - {self.name}"


class ContestParticipant(models.Model):
    """Contest participation with proctoring data"""
    contest = models.ForeignKey(Contest, on_delete=models.CASCADE, related_name='participants')
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='contest_participations')
    
    # Registration
    registered_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)
    
    # Performance
    total_score = models.IntegerField(default=0)
    problems_solved = models.IntegerField(default=0)
    penalty_time = models.IntegerField(default=0)  # in minutes
    last_submission_time = models.DateTimeField(null=True, blank=True)
    
    # Ranking
    rank = models.IntegerField(null=True, blank=True)
    
    # Proctoring data
    proctoring_session_id = models.CharField(max_length=100, null=True, blank=True)
    tab_switches = models.IntegerField(default=0)
    suspicious_activities = models.IntegerField(default=0)
    video_violations = models.IntegerField(default=0)
    
    class Meta:
        unique_together = ['contest', 'user']
        ordering = ['rank', '-total_score', 'penalty_time']


class ExamSession(models.Model):
    STATUS_CHOICES = (
        ('active', 'Active'),
        ('completed', 'Completed'),
        ('expired', 'Expired'),
    )

    session_id = models.CharField(max_length=64, unique=True)
    problem = models.ForeignKey(Problem, on_delete=models.CASCADE, related_name="exam_sessions", null=True, blank=True)
    contest = models.ForeignKey('Contest', on_delete=models.CASCADE, related_name="exam_sessions", null=True, blank=True)
    start_time = models.DateTimeField(auto_now_add=True, null=True, blank=True)
    end_time = models.DateTimeField(null=True, blank=True)
    time_remaining = models.IntegerField(default=0)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='active')
    is_submitted = models.BooleanField(default=False)
    suspicious_activities = models.JSONField(default=list, blank=True)
    tab_switches = models.IntegerField(default=0)
    copy_paste_attempts = models.IntegerField(default=0)
    right_click_attempts = models.IntegerField(default=0)

    def add_security_event(self, event_type, description=""):
        self.suspicious_activities.append({"type": event_type, "description": description})
        self.save(update_fields=['suspicious_activities'])

    def __str__(self):
        return f"Session {self.session_id} ({self.status})"


class Submission(models.Model):
    STATUS_CHOICES = (
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('accepted', 'Accepted'),
        ('wrong_answer', 'Wrong Answer'),
        ('runtime_error', 'Runtime Error'),
        ('time_limit_exceeded', 'Time Limit Exceeded'),
        ('compilation_error', 'Compilation Error'),
    )

    exam_session = models.ForeignKey('ExamSession', on_delete=models.CASCADE, related_name="submissions", null=True, blank=True)
    user = models.ForeignKey('auth.User', on_delete=models.CASCADE, null=True, blank=True)
    problem = models.ForeignKey('Problem', on_delete=models.CASCADE, related_name="submissions", null=True, blank=True)
    contest = models.ForeignKey('Contest', on_delete=models.CASCADE, related_name="submissions", null=True, blank=True)
    code = models.TextField()
    language = models.ForeignKey('Language', on_delete=models.CASCADE, null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')

    results_summary = models.JSONField(default=list, blank=True)   # ✅ renamed from test_results

    execution_time = models.FloatField(null=True, blank=True)
    memory_used = models.IntegerField(null=True, blank=True)
    error_message = models.TextField(blank=True)
    submitted_at = models.DateTimeField(auto_now_add=True, null=True, blank=True)
    score = models.IntegerField(default=0)
    max_score = models.IntegerField(default=0)

    def __str__(self):
        return f"Submission {self.id} - {self.status}"

class TestResult(models.Model):
    submission = models.ForeignKey('Submission', on_delete=models.CASCADE, related_name='test_results')
    test_case = models.ForeignKey('TestCase', on_delete=models.CASCADE)
    input_data = models.TextField(null=True, blank=True)
    expected_output = models.TextField(null=True, blank=True)
    actual_output = models.TextField(null=True, blank=True)
    is_passed = models.BooleanField(default=False)
    execution_time = models.FloatField(null=True, blank=True)
    error_message = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True, null=True, blank=True)

    class Meta:
        ordering = ['id']



class ProctoringSession(models.Model):
    """Proctoring session data"""
    session_id = models.CharField(max_length=100, unique=True)
    exam_session = models.OneToOneField(
        ExamSession, on_delete=models.CASCADE, related_name='proctoring_data'
    )
    
    # Recording files
    video_recording = models.FileField(upload_to='proctoring/videos/', null=True, blank=True)
    audio_recording = models.FileField(upload_to='proctoring/audio/', null=True, blank=True)
    screen_recording = models.FileField(upload_to='proctoring/screens/', null=True, blank=True)
    
    # Behavioral data
    eye_tracking_data = models.JSONField(default=list)
    keystroke_patterns = models.JSONField(default=list)
    mouse_movements = models.JSONField(default=list)
    
    # Violations
    face_detection_failures = models.IntegerField(default=0)
    multiple_faces_detected = models.IntegerField(default=0)
    no_face_detected_duration = models.IntegerField(default=0)  # in seconds
    suspicious_objects_detected = models.JSONField(default=list)
    
    # Analysis results
    risk_score = models.FloatField(default=0.0)  # 0-100
    is_flagged = models.BooleanField(default=False)
    review_required = models.BooleanField(default=False)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"Proctoring {self.session_id}"


class CodeSimilarity(models.Model):
    """Code similarity detection for plagiarism"""
    submission1 = models.ForeignKey(
        Submission, on_delete=models.CASCADE, related_name='similarity_checks_as_first'
    )
    submission2 = models.ForeignKey(
        Submission, on_delete=models.CASCADE, related_name='similarity_checks_as_second'
    )
    
    similarity_score = models.FloatField()  # 0-100
    similarity_type = models.CharField(max_length=50)  # structural, lexical, etc.
    
    # Detailed analysis
    common_patterns = models.JSONField(default=list)
    analysis_data = models.JSONField(default=dict)
    
    is_flagged = models.BooleanField(default=False)
    reviewed = models.BooleanField(default=False)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['submission1', 'submission2']



