"""
Django settings for coding_exam_system project.
"""

from pathlib import Path
import os
from dotenv import load_dotenv
import mongoengine


# Load environment variables
load_dotenv()

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.getenv('SECRET_KEY', 'django-insecure-your-secret-key-here')

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'

ALLOWED_HOSTS = os.getenv('ALLOWED_HOSTS', 'localhost,127.0.0.1').split(',')

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'corsheaders',
    'exams',
    
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'exams.middleware.DisableCSRFForAPI',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

CORS_ALLOW_ALL_ORIGINS=True

ROOT_URLCONF = 'coding_exam_system.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'coding_exam_system.wsgi.application'

# Database - Using SQLite for now, will configure MongoDB separately
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# MongoDB Configuration for MongoEngine (Optional)
# Set ENABLE_MONGODB=true in environment to enable MongoDB connection
ENABLE_MONGODB = os.getenv('ENABLE_MONGODB', 'false').lower() == 'true'

if ENABLE_MONGODB:
    import mongoengine
    MONGODB_NAME = os.getenv('MONGODB_NAME', 'coding_exam_db')
    MONGODB_USERNAME = os.getenv('MONGODB_USERNAME', 'shyaamsundar2310422_db_user')
    MONGODB_PASSWORD = os.getenv('MONGODB_PASSWORD', 'IFP123$$')
    MONGODB_CLUSTER = os.getenv('MONGODB_CLUSTER', 'cluster0.8jdoj1f.mongodb.net')

    # Construct MongoDB connection string with database name
    # Format: mongodb+srv://username:password@cluster/database?retryWrites=true&w=majority
    MONGODB_URI = os.getenv('MONGODB_URI')
    if not MONGODB_URI:
        # URL encode the password to handle special characters like $
        from urllib.parse import quote_plus
        encoded_password = quote_plus(MONGODB_PASSWORD)
        MONGODB_URI = f"mongodb+srv://{MONGODB_USERNAME}:{encoded_password}@{MONGODB_CLUSTER}/{MONGODB_NAME}?retryWrites=true&w=majority&authSource=admin"

    # Connect to MongoDB (only attempt once, check if already connected)
    try:
        # Check if already connected to avoid duplicate connection attempts
        try:
            existing_conn = mongoengine.connection.get_connection()
            if existing_conn:
                print(f"✅ MongoDB already connected: {MONGODB_NAME}")
            else:
                raise AttributeError("No existing connection")
        except (AttributeError, Exception):
            # No existing connection, create new one
            mongoengine.connect(
                db=MONGODB_NAME,
                host=MONGODB_URI,
                retryWrites=True
            )
            print(f"✅ Connected to MongoDB: {MONGODB_NAME}")
    except Exception as e:
        # Only log as warning, not error, since SQLite fallback is available
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"MongoDB connection not available: {e}")
        logger.info("Using SQLite database (MongoDB is optional)")
else:
    # MongoDB disabled, using SQLite only
    pass

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Static files (CSS, JavaScript, Images)
STATIC_URL = 'static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'
STATICFILES_DIRS = [
    BASE_DIR / 'static',
]

# Media files
MEDIA_URL = 'media/'
MEDIA_ROOT = BASE_DIR / 'media'

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# REST Framework settings
REST_FRAMEWORK = {
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.AllowAny',
    ],
    'DEFAULT_RENDERER_CLASSES': [
        'rest_framework.renderers.JSONRenderer',
    ],
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.SessionAuthentication',
    ],
}

# CORS settings
CORS_ALLOW_ALL_ORIGINS = DEBUG
CORS_ALLOWED_ORIGINS = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

# Security settings
if not DEBUG:
    SECURE_BROWSER_XSS_FILTER = True
    SECURE_CONTENT_TYPE_NOSNIFF = True
    X_FRAME_OPTIONS = 'DENY'

# Debug logging
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'INFO',
    },
    'loggers': {
        'django': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False,
        },
    },
} 