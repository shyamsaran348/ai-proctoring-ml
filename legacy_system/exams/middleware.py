from django.utils.deprecation import MiddlewareMixin
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator


class DisableCSRFForAPI(MiddlewareMixin):
    """
    Middleware to disable CSRF for API endpoints
    """
    
    def process_view(self, request, view_func, view_args, view_kwargs):
        # Check if the request is for an API endpoint
        if request.path.startswith('/api/'):
            # Mark request to skip CSRF enforcement without executing the view here
            setattr(request, '_dont_enforce_csrf', True)
        return None
