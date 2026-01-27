from django.contrib import admin
from .models import Problem, TestCase, ExamSession, Submission, TestResult


@admin.register(Problem)
class ProblemAdmin(admin.ModelAdmin):
    list_display = (
        'id',
        'title',
        'difficulty',
        'time_limit_seconds',
        'points',
        'is_active',
    )
    list_filter = ('difficulty', 'is_active')


@admin.register(TestCase)
class TestCaseAdmin(admin.ModelAdmin):
    list_display = (
        'id',
        'name',
        'test_type',
        'problem',
        'order',
        'is_active',
        'is_hidden_flag',
    )
    list_filter = ('test_type', 'is_active')

    # Expose a computed column for admin
    def is_hidden_flag(self, obj):
        return obj.test_type in ('private', 'stress')
    is_hidden_flag.short_description = "Is Hidden"
    is_hidden_flag.boolean = True


@admin.register(ExamSession)
class ExamSessionAdmin(admin.ModelAdmin):
    list_display = (
        'id',
        'session_id',
        'problem',
        'status',
        'is_submitted',
        'time_remaining',
    )
    list_filter = ('status', 'is_submitted')


@admin.register(Submission)
class SubmissionAdmin(admin.ModelAdmin):
    list_display = (
        'id',
        'exam_session',
        'problem',
        'status',
        'score',
        'submitted_at',
    )
    list_filter = ('status',)


@admin.register(TestResult)
class TestResultAdmin(admin.ModelAdmin):
    list_display = (
        'id',
        'submission',
        'test_case',
        'is_passed',
        'execution_time',
    )
    list_filter = ('is_passed',)