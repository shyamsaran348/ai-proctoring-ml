#!/usr/bin/env python
import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'coding_exam_system.settings')
django.setup()

from exams.models import Problem, TestCase

# Check test case data
p = Problem.objects.first()
if p:
    print(f'Problem: {p.title}')
    print(f'Number of test cases: {p.test_cases.count()}')
    
    for i, tc in enumerate(p.test_cases.all()[:5]):
        print(f'\nTest Case {i+1}: {tc.name}')
        print(f'  Type: {tc.test_type}')
        print(f'  Input: {repr(tc.input_data)}')
        print(f'  Expected: {repr(tc.expected_output)}')
        print(f'  Order: {tc.order}')
else:
    print('No problems found')
