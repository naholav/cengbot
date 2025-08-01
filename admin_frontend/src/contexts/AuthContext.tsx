import React, { createContext, useContext, useState, useEffect } from 'react';
import axios from 'axios';

interface AuthContextType {
  isAuthenticated: boolean;
  login: (password: string) => Promise<boolean>;
  logout: () => void;
  authHeader: string;
}

const AuthContext = createContext<AuthContextType | null>(null);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return context;
};

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [authHeader, setAuthHeader] = useState('');

  useEffect(() => {
    // Check if we have saved credentials
    const savedAuth = localStorage.getItem('adminAuth');
    if (savedAuth) {
      setAuthHeader(savedAuth);
      setIsAuthenticated(true);
      // Set default auth header for axios
      axios.defaults.headers.common['Authorization'] = savedAuth;
    }
  }, []);

  const login = async (password: string): Promise<boolean> => {
    try {
      const authString = `Basic ${btoa(`admin:${password}`)}`;
      const response = await axios.post(
        'http://localhost:8001/auth/login',
        {},
        {
          headers: {
            'Authorization': authString
          }
        }
      );

      if (response.data.success) {
        setIsAuthenticated(true);
        setAuthHeader(authString);
        localStorage.setItem('adminAuth', authString);
        // Set default auth header for axios
        axios.defaults.headers.common['Authorization'] = authString;
        return true;
      }
      return false;
    } catch (error) {
      console.error('Login failed:', error);
      return false;
    }
  };

  const logout = () => {
    setIsAuthenticated(false);
    setAuthHeader('');
    localStorage.removeItem('adminAuth');
    delete axios.defaults.headers.common['Authorization'];
  };

  return (
    <AuthContext.Provider value={{ isAuthenticated, login, logout, authHeader }}>
      {children}
    </AuthContext.Provider>
  );
};