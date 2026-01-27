"""
WSGI config for coding_exam_system project.
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'coding_exam_system.settings')

application = get_wsgi_application() 