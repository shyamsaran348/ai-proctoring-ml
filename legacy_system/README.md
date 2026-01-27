# Coding Exam System - Django Backend

A comprehensive coding exam system built with Django framework that provides a robust backend for managing coding problems, test cases, exam sessions, and code execution.

## Features

- **Problem Management**: Create and manage coding problems with test cases
- **Code Execution**: Execute JavaScript and Python code with real-time testing
- **Exam Sessions**: Track exam sessions with timers and submissions
- **Test Case Management**: Comprehensive test case system with hidden/public cases
- **Multi-language Support**: JavaScript and Python support out of the box
- **RESTful API**: Full API endpoints for integration
- **Admin Interface**: Django admin for managing all aspects of the system
- **Real-time Timer**: Built-in timer for exam sessions
- **Submission Tracking**: Track and evaluate code submissions

## Technology Stack

- **Backend**: Django 4.2.7
- **API**: Django REST Framework
- **Database**: SQLite (configurable for production)
- **Code Execution**: Subprocess-based execution with timeout protection
- **Frontend**: Modern HTML/CSS/JavaScript with responsive design

## Prerequisites

- Python 3.8+
- Node.js (for JavaScript code execution)
- pip (Python package manager)

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd coding_exam_system
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run database migrations**
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

5. **Create a superuser (optional)**
   ```bash
   python manage.py createsuperuser
   ```

6. **Populate with sample data**
   ```bash
   python manage.py populate_problems
   ```

7. **Run the development server**
   ```bash
   python manage.py runserver
   ```

8. **Access the application**
   - Frontend: http://localhost:8000/
   - Admin: http://localhost:8000/admin/
   - API: http://localhost:8000/api/

## Project Structure

```
coding_exam_system/
├── coding_exam_system/          # Django project settings
│   ├── __init__.py
│   ├── settings.py              # Django settings
│   ├── urls.py                  # Main URL configuration
│   ├── wsgi.py                  # WSGI configuration
│   └── asgi.py                  # ASGI configuration
├── exams/                       # Main Django app
│   ├── __init__.py
│   ├── models.py                # Database models
│   ├── views.py                 # Views and API endpoints
│   ├── serializers.py           # DRF serializers
│   ├── urls.py                  # App URL configuration
│   ├── admin.py                 # Admin interface
│   └── management/              # Management commands
│       └── commands/
│           └── populate_problems.py
├── templates/                   # HTML templates
│   └── exams/
│       └── index.html          # Main frontend
├── static/                      # Static files
├── manage.py                    # Django management script
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```

## API Endpoints

### Problems
- `GET /api/problems/` - List all problems
- `GET /api/problems/{id}/` - Get specific problem
- `POST /api/problems/{id}/start_exam/` - Start exam session

### Exam Sessions
- `GET /api/sessions/` - List all sessions
- `POST /api/sessions/` - Create new session
- `GET /api/sessions/{id}/` - Get specific session
- `POST /api/sessions/{id}/submit/` - Submit code

### Code Execution
- `POST /api/execute/` - Execute code with test cases

### Submissions
- `GET /api/submissions/` - List all submissions
- `GET /api/submissions/{id}/` - Get specific submission

## Models

### Problem
- Title, description, initial code
- Time limit for completion
- Active/inactive status

### TestCase
- Problem association
- Input data and expected output
- Hidden/public visibility
- Order for display

### ExamSession
- User session tracking
- Problem association
- Timer and completion status

### Submission
- Code content and language
- Test results and execution metrics
- Status tracking

### TestResult
- Individual test case results
- Execution time and memory usage
- Error messages

## Code Execution

The system supports two programming languages:

### JavaScript
- Executed using Node.js
- Function must be named `solve`
- Returns results via console.log

### Python
- Executed using Python interpreter
- Function must be named `solve`
- Returns results via print

## Security Features

- CSRF protection
- Input validation
- Execution timeout (10 seconds)
- Sandboxed code execution
- Rate limiting considerations

## Customization

### Adding New Problems
1. Use the Django admin interface
2. Or create via management commands
3. Or use the API endpoints

### Adding New Languages
1. Extend the `CodeExecutionView` class
2. Add language-specific execution logic
3. Update the frontend language selector

### Modifying Test Cases
- Use the admin interface for easy management
- Test cases support JSON data for complex inputs
- Hidden test cases for final evaluation

## Production Deployment

### Environment Variables
Create a `.env` file:
```env
DEBUG=False
SECRET_KEY=your-secure-secret-key
ALLOWED_HOSTS=yourdomain.com,www.yourdomain.com
DATABASE_URL=postgresql://user:password@localhost/dbname
```

### Database
For production, consider using PostgreSQL:
```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'your_db_name',
        'USER': 'your_db_user',
        'PASSWORD': 'your_db_password',
        'HOST': 'localhost',
        'PORT': '5432',
    }
}
```

### Static Files
```bash
python manage.py collectstatic
```

### Security
- Set `DEBUG=False`
- Use HTTPS
- Configure proper CORS settings
- Set secure secret key
- Enable security headers

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues and questions:
1. Check the documentation
2. Search existing issues
3. Create a new issue with detailed information

## Roadmap

- [ ] User authentication and authorization
- [ ] Multiple programming language support
- [ ] Real-time collaboration features
- [ ] Advanced test case types
- [ ] Performance analytics
- [ ] Mobile app support
- [ ] Integration with external IDEs 


current issues:
   CodeMirror editor done
   no dyanmic web cam feed update in frontend
   exam completion recording is incomplete
   mcq support