import React, { createContext, useContext, useState, useEffect } from 'react';
import api from '../services/api';

const AuthContext = createContext();

export const useAuth = () => useContext(AuthContext);

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  // Re-hydrate session on mount
  useEffect(() => {
    const hydrator = async () => {
      try {
        // Strict check: ask backend if we have a valid session cookie
        const res = await api.get('/auth/session-check/'); 
        if (res.status === 200) {
           setUser(res.data.user);
           localStorage.setItem('userMeta', JSON.stringify(res.data.user));
        } else {
           throw new Error("Invalid session");
        }
      } catch (err) {
        // Forensic Clear: Wipe state if session is dead
        setUser(null);
        localStorage.removeItem('userMeta');
      } finally {
        setLoading(false);
      }
    };
    hydrator();
  }, []);

  const login = async (email, password, isFaculty = false) => {
    try {
      const endpoint = isFaculty ? '/faculty-login/' : '/login/';
      await api.post(endpoint, { email, password });
      
      // Fetch dynamic session check to get full backend user model attributes!
      const checkRes = await api.get('/auth/session-check/');
      const userData = checkRes.data.user;
      
      setUser(userData);
      localStorage.setItem('userMeta', JSON.stringify(userData));
      return { success: true };
    } catch (err) {
      return { 
        success: false, 
        error: err.response?.data?.error || 'Login failed due to network error.' 
      };
    }
  };

  const register = async (userData, isFaculty = false) => {
    try {
      const endpoint = isFaculty ? '/faculty-register/' : '/register/';
      
      // Handle Multi-part form data for photo uploads (AI Enrollment)
      const isFormData = userData instanceof FormData;
      await api.post(endpoint, userData, {
        headers: isFormData ? { 'Content-Type': 'multipart/form-data' } : {}
      });
      
      // Fetch dynamic session check to get full backend user model attributes!
      const checkRes = await api.get('/auth/session-check/');
      const savedUser = checkRes.data.user;

      setUser(savedUser);
      localStorage.setItem('userMeta', JSON.stringify(savedUser));
      return { success: true };
    } catch (err) {
      // Return Django constraint validation errors if they exist
      const errorMsg = err.response?.data?.error || 
                       (err.response?.data ? JSON.stringify(err.response.data) : 'Registration failed.');
      return { success: false, error: errorMsg };
    }
  };

  const logout = async () => {
    try {
      // Inform the Django backend to destroy the Session Cookie
      // (Assuming you map a /logout/ endpoint in Django, or simply rely on clearing local state 
      // and forcing a 401 on next request).
      // await api.post('/logout/');
    } catch (err) { }
    
    setUser(null);
    localStorage.removeItem('userMeta');
    window.location.href = '/login';
  };

  return (
    <AuthContext.Provider value={{ user, loading, login, register, logout }}>
      {!loading && children}
    </AuthContext.Provider>
  );
};
