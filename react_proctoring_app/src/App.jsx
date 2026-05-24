import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext';

// Pages
import Login from './components/auth/Login';
import Register from './components/auth/Register';
import Dashboard from './pages/student/Dashboard';
import ExamArena from './pages/student/ExamArena';
import CommandCenter from './pages/faculty/CommandCenter';

// Route Guards
const PrivateRoute = ({ children, requireFaculty }) => {
  const { user, loading } = useAuth();
  
  if (loading) {
    return <div className="flex items-center justify-center min-h-screen bg-gray-50">Loading Secure Environment...</div>;
  }
  
  if (!user) {
    return <Navigate to="/login" replace />;
  }

  if (requireFaculty && user.role !== 'faculty') {
    return <Navigate to="/dashboard" replace />;
  }

  if (!requireFaculty && user.role === 'faculty') {
    return <Navigate to="/faculty" replace />;
  }

  return children;
};

const RootRedirect = () => {
  const { user, loading } = useAuth();
  if (loading) {
    return <div className="flex items-center justify-center min-h-screen bg-gray-50 font-semibold text-slate-500">Loading Secure Environment...</div>;
  }
  if (!user) return <Navigate to="/login" replace />;
  return <Navigate to={user.role === 'faculty' ? '/faculty' : '/dashboard'} replace />;
};

export default function App() {
  return (
    <Router>
      <AuthProvider>
        <Routes>
          {/* Public Routes */}
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />

          {/* Student Protected Routes */}
          <Route 
            path="/dashboard" 
            element={
              <PrivateRoute>
                <Dashboard />
              </PrivateRoute>
            } 
          />
          <Route 
            path="/arena/:sessionId" 
            element={
              <PrivateRoute>
                <ExamArena />
              </PrivateRoute>
            } 
          />

          {/* Faculty Protected Routes */}
          <Route 
            path="/faculty" 
            element={
              <PrivateRoute requireFaculty={true}>
                <CommandCenter />
              </PrivateRoute>
            } 
          />

          {/* Base Redirect */}
          <Route path="/" element={<RootRedirect />} />
          <Route path="*" element={<RootRedirect />} />
        </Routes>
      </AuthProvider>
    </Router>
  );
}
