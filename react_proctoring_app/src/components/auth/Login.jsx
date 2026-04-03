import React, { useState } from 'react';
import { useAuth } from '../../context/AuthContext';
import { useNavigate, Link } from 'react-router-dom';
import { Shield, Mail, Lock, AlertCircle } from 'lucide-react';

export default function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isFaculty, setIsFaculty] = useState(false);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  
  const { login } = useAuth();
  const navigate = useNavigate();

  const handleLogin = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    
    // In Django, standard auth typically uses 'username' behind the scenes, 
    // but the custom view accepts 'email' parameter and handles the fallback.
    const res = await login(email, password, isFaculty);
    
    if (res.success) {
       navigate(isFaculty ? '/faculty' : '/dashboard');
    } else {
       setError(res.error);
       setIsLoading(false);
    }
  };

  return (
    <div className="flex justify-center items-center min-h-screen bg-gray-50 px-4">
      <div className="w-full max-w-md">
        <div className="mb-8 text-center">
          <div className="inline-flex justify-center items-center h-16 w-16 rounded-2xl bg-indigo-600 text-white shadow-xl mb-4">
            <Shield size={32} />
          </div>
          <h2 className="text-3xl font-extrabold text-gray-900 tracking-tight">
            DevProctor
          </h2>
          <p className="mt-2 text-sm text-gray-600 font-medium">
            AI-Powered Examination Environment
          </p>
        </div>

        <div className="bg-white py-8 px-8 shadow-2xl rounded-2xl border border-gray-100">
          <div className="flex bg-gray-100 p-1 rounded-xl mb-8">
            <button
              type="button"
              className={`flex-1 py-2.5 text-sm font-semibold rounded-lg transition-all ${
                !isFaculty ? 'bg-white text-indigo-700 shadow-sm' : 'text-gray-500 hover:text-gray-700'
              }`}
              onClick={() => setIsFaculty(false)}
            >
              Examinee
            </button>
            <button
              type="button"
              className={`flex-1 py-2.5 text-sm font-semibold rounded-lg transition-all ${
                isFaculty ? 'bg-white text-indigo-700 shadow-sm' : 'text-gray-500 hover:text-gray-700'
              }`}
              onClick={() => setIsFaculty(true)}
            >
              Proctor
            </button>
          </div>

          <form onSubmit={handleLogin} className="space-y-6">
            {error && (
              <div className="bg-red-50 p-4 rounded-xl flex items-start">
                <AlertCircle className="h-5 w-5 text-red-500 mt-0.5 mr-3 flex-shrink-0" />
                <p className="text-sm text-red-700 font-medium">{error}</p>
              </div>
            )}

            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                {isFaculty ? "Faculty ID / Email" : "Student ID / Email"}
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Mail className="h-5 w-5 text-gray-400" />
                </div>
                <input
                  type="text"
                  required
                  autoFocus
                  className="block w-full pl-10 pr-3 py-3 border border-gray-200 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 bg-gray-50 transition-colors"
                  placeholder="Enter your credential"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">Password</label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Lock className="h-5 w-5 text-gray-400" />
                </div>
                <input
                  type="password"
                  required
                  className="block w-full pl-10 pr-3 py-3 border border-gray-200 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 bg-gray-50 transition-colors"
                  placeholder="••••••••"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={isLoading}
              className="w-full flex justify-center py-3 px-4 border border-transparent rounded-xl shadow-md text-sm font-bold text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition-all disabled:opacity-70"
            >
              {isLoading ? 'Authenticating...' : 'Secure Login'}
            </button>
          </form>

          <div className="mt-8 text-center">
            <p className="text-sm text-gray-600">
              New to the platform?{' '}
              <Link to="/register" className="font-bold text-indigo-600 hover:text-indigo-500">
                Register an account
              </Link>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
