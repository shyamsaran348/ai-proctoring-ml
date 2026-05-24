import React, { useState, useRef, useEffect } from 'react';
import { useAuth } from '../../context/AuthContext';
import { useNavigate, Link } from 'react-router-dom';
import { Shield, UserPlus, Lock, Mail, AlertCircle, Camera, IdCard, CheckCircle2 } from 'lucide-react';

export default function Register() {
  const [formData, setFormData] = useState({
    username: '',
    email: '',
    password: '',
    confirmPassword: '',
    firstName: '',
    lastName: '',
    aadharNumber: ''
  });
  const [isFaculty, setIsFaculty] = useState(false);
  const [photo, setPhoto] = useState(null);
  const [photoPreview, setPhotoPreview] = useState(null);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  
  const fileInputRef = useRef(null);
  const { user, register } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    if (user) {
      navigate(user.role === 'faculty' ? '/faculty' : '/dashboard', { replace: true });
    }
  }, [user, navigate]);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handlePhotoChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setPhoto(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setPhotoPreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleRegister = async (e) => {
    e.preventDefault();
    setError(null);

    if (formData.password !== formData.confirmPassword) {
      setError("Passwords do not match");
      return;
    }

    if (!isFaculty && !photo) {
      setError("Student registration requires a reference photo for Biometric MFA.");
      return;
    }

    setIsLoading(true);
    
    // Construct FormData for multipart submission
    const data = new FormData();
    data.append('username', formData.username);
    data.append('email', formData.email);
    data.append('password', formData.password);
    data.append('first_name', formData.firstName);
    data.append('last_name', formData.lastName);
    data.append('role', isFaculty ? 'faculty' : 'student');
    
    if (!isFaculty) {
      data.append('aadhar_number', formData.aadharNumber);
      data.append('photo', photo);
    }

    const res = await register(data, isFaculty);
    
    if (res.success) {
       navigate(isFaculty ? '/faculty' : '/dashboard');
    } else {
       setError(res.error);
       setIsLoading(false);
    }
  };

  return (
    <div className="flex justify-center items-center min-h-screen bg-gray-50 px-4 py-12">
      <div className="w-full max-w-lg">
        <div className="mb-8 text-center">
          <div className="inline-flex justify-center items-center h-16 w-16 rounded-2xl bg-indigo-600 text-white shadow-xl mb-4">
            <UserPlus size={32} />
          </div>
          <h2 className="text-3xl font-extrabold text-gray-900 tracking-tight">
            Sentinel Enrollment
          </h2>
          <p className="mt-2 text-sm text-gray-600 font-medium uppercase tracking-wider">
            Identity Persistence Protocol
          </p>
        </div>

        <div className="bg-white py-8 px-8 shadow-2xl rounded-3xl border border-gray-100">
          <div className="flex bg-gray-100 p-1 rounded-2xl mb-8">
            <button
              type="button"
              className={`flex-1 py-3 text-sm font-black rounded-xl transition-all uppercase tracking-widest ${
                !isFaculty ? 'bg-white text-indigo-700 shadow-sm' : 'text-gray-500 hover:text-gray-700'
              }`}
              onClick={() => setIsFaculty(false)}
            >
              Examinee
            </button>
            <button
              type="button"
              className={`flex-1 py-3 text-sm font-black rounded-xl transition-all uppercase tracking-widest ${
                isFaculty ? 'bg-white text-indigo-700 shadow-sm' : 'text-gray-500 hover:text-gray-700'
              }`}
              onClick={() => setIsFaculty(true)}
            >
              Proctor
            </button>
          </div>

          <form onSubmit={handleRegister} className="space-y-5">
            {error && (
              <div className="bg-red-50 p-4 rounded-2xl flex items-start border border-red-100">
                <AlertCircle className="h-5 w-5 text-red-500 mt-0.5 mr-3 flex-shrink-0" />
                <p className="text-sm text-red-700 font-bold whitespace-pre-wrap">{error}</p>
              </div>
            )}

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-black text-slate-400 uppercase tracking-widest mb-2">First Name</label>
                <input
                  type="text"
                  name="firstName"
                  required
                  className="block w-full px-4 py-3 border border-gray-100 rounded-2xl focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 bg-gray-50 transition-all font-medium text-slate-700"
                  placeholder="John"
                  value={formData.firstName}
                  onChange={handleChange}
                />
              </div>
              <div>
                <label className="block text-xs font-black text-slate-400 uppercase tracking-widest mb-2">Last Name</label>
                <input
                  type="text"
                  name="lastName"
                  required
                  className="block w-full px-4 py-3 border border-gray-100 rounded-2xl focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 bg-gray-50 transition-all font-medium text-slate-700"
                  placeholder="Doe"
                  value={formData.lastName}
                  onChange={handleChange}
                />
              </div>
            </div>

            <div>
              <label className="block text-xs font-black text-slate-400 uppercase tracking-widest mb-2">
                {isFaculty ? "Faculty ID / Username" : "Student ID / Username"}
              </label>
              <input
                type="text"
                name="username"
                required
                className="block w-full px-4 py-3 border border-gray-100 rounded-2xl focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 bg-gray-50 transition-all font-medium text-slate-700"
                placeholder={isFaculty ? "prof_smith" : "stu_12345"}
                value={formData.username}
                onChange={handleChange}
              />
            </div>

            <div>
              <label className="block text-xs font-black text-slate-400 uppercase tracking-widest mb-2">Email Address</label>
              <input
                type="email"
                name="email"
                required
                className="block w-full px-4 py-3 border border-gray-100 rounded-2xl focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 bg-gray-50 transition-all font-medium text-slate-700"
                placeholder="name@university.edu"
                value={formData.email}
                onChange={handleChange}
              />
            </div>

            {!isFaculty && (
              <div>
                <label className="block text-xs font-black text-slate-400 uppercase tracking-widest mb-2">Aadhar / National ID</label>
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                    <IdCard className="h-5 w-5 text-gray-400" />
                  </div>
                  <input
                    type="text"
                    name="aadharNumber"
                    required
                    className="block w-full pl-12 pr-4 py-3 border border-gray-100 rounded-2xl focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 bg-gray-50 transition-all font-medium text-slate-700"
                    placeholder="0000 0000 0000"
                    value={formData.aadharNumber}
                    onChange={handleChange}
                  />
                </div>
              </div>
            )}

            {!isFaculty && (
              <div className="pt-2">
                <label className="block text-xs font-black text-slate-400 uppercase tracking-widest mb-4">Identity Capture (MFA Reference)</label>
                <div 
                  onClick={() => fileInputRef.current.click()}
                  className={`relative cursor-pointer border-2 border-dashed rounded-3xl p-8 transition-all flex flex-col items-center justify-center gap-3 ${
                    photoPreview ? 'border-emerald-200 bg-emerald-50' : 'border-slate-100 bg-slate-50 hover:bg-slate-100'
                  }`}
                >
                  <input 
                    type="file" 
                    ref={fileInputRef}
                    className="hidden" 
                    accept="image/*"
                    onChange={handlePhotoChange}
                  />
                  {photoPreview ? (
                    <>
                      <img src={photoPreview} alt="Reference" className="h-24 w-24 rounded-2xl object-cover shadow-lg border-2 border-white" />
                      <div className="flex items-center gap-1 text-emerald-600">
                        <CheckCircle2 size={16} />
                        <span className="text-[10px] font-black uppercase tracking-widest">Ground Truth Captured</span>
                      </div>
                    </>
                  ) : (
                    <>
                      <div className="p-4 bg-white rounded-2xl shadow-sm text-slate-400">
                        <Camera size={28} />
                      </div>
                      <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Upload Portrait</p>
                    </>
                  )}
                </div>
              </div>
            )}

            <div className="grid grid-cols-2 gap-4 pt-2">
              <div>
                <label className="block text-xs font-black text-slate-400 uppercase tracking-widest mb-2">Password</label>
                <input
                  type="password"
                  name="password"
                  required
                  className="block w-full px-4 py-3 border border-gray-100 rounded-2xl focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 bg-gray-50 transition-all font-medium text-slate-700"
                  placeholder="••••••••"
                  value={formData.password}
                  onChange={handleChange}
                />
              </div>
              <div>
                <label className="block text-xs font-black text-slate-400 uppercase tracking-widest mb-2">Confirm</label>
                <input
                  type="password"
                  name="confirmPassword"
                  required
                  className="block w-full px-4 py-3 border border-gray-100 rounded-2xl focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 bg-gray-50 transition-all font-medium text-slate-700"
                  placeholder="••••••••"
                  value={formData.confirmPassword}
                  onChange={handleChange}
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={isLoading}
              className="w-full flex justify-center py-4 px-4 border border-transparent rounded-2xl shadow-2xl text-sm font-black text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition-all disabled:opacity-70 mt-4 uppercase tracking-[0.2em]"
            >
              {isLoading ? 'Encrypting Account...' : 'Complete Enrollment'}
            </button>
          </form>

          <div className="mt-10 text-center">
            <p className="text-xs text-gray-500 font-medium">
              Already persists in registry?{' '}
              <Link to="/login" className="font-black text-indigo-600 hover:text-indigo-500 ml-1 uppercase tracking-tighter">
                Enter Interface
              </Link>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
