import axios from 'axios';

// Create a specialized Axios instance for Django backend
const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || '/api/',
  timeout: 10000,
  withCredentials: true, // IMPORTANT: Allows Django Session cookies to traverse ports
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  }
});

// Create a separate instance for FormData (file uploads like Reference webcam snapshots)
export const apiForm = axios.create({
  baseURL: import.meta.env.VITE_API_URL || '/api/',
  timeout: 15000,
  withCredentials: true,
});

// CSRF extraction helper for Django
function getCookie(name) {
  let cookieValue = null;
  if (document.cookie && document.cookie !== '') {
    const cookies = document.cookie.split(';');
    for (let i = 0; i < cookies.length; i++) {
      const cookie = cookies[i].trim();
      // Does this cookie string begin with the name we want?
      if (cookie.substring(0, name.length + 1) === (name + '=')) {
        cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
        break;
      }
    }
  }
  return cookieValue;
}

// Request Interceptor to inject CSRF tokens dynamically if they exist
api.interceptors.request.use(
  (config) => {
    const csrfToken = getCookie('csrftoken');
    if (csrfToken) {
      config.headers['X-CSRFToken'] = csrfToken;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

apiForm.interceptors.request.use(
  (config) => {
    const csrfToken = getCookie('csrftoken');
    if (csrfToken) {
      config.headers['X-CSRFToken'] = csrfToken;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response Interceptor to gracefully handle Unauthorized errors globally
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response && error.response.status === 401) {
      if (!window.location.pathname.includes('/login') && !window.location.pathname.includes('/register')) {
        // Only force redirect if we aren't already on an auth page, otherwise let the auth page handle the 401 logic
        window.location.href = '/login';
      }
    }
    return Promise.reject(error);
  }
);

export default api;
