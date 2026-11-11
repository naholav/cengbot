import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import axios from 'axios';
import {
  apiService,
  clearAuth,
  encodeBasicAuth,
  getErrorMessage,
  restoreAuthFromStorage,
  setAuthHeader,
  setUnauthorizedHandler,
} from '../services/api';

export type LoginResult = { ok: true } | { ok: false; error: string };

interface AuthContextType {
  isAuthenticated: boolean;
  /** True after the API rejected the stored credentials (e.g. password changed). */
  sessionExpired: boolean;
  login: (username: string, password: string) => Promise<LoginResult>;
  logout: () => void;
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
  // Restore synchronously so the first data requests already carry the header
  // and the login screen does not flash on reload.
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(() => restoreAuthFromStorage() !== null);
  const [sessionExpired, setSessionExpired] = useState(false);

  useEffect(
    () =>
      setUnauthorizedHandler(() => {
        setIsAuthenticated(false);
        setSessionExpired(true);
      }),
    []
  );

  const login = useCallback(async (username: string, password: string): Promise<LoginResult> => {
    const header = encodeBasicAuth(username.trim(), password);
    try {
      const data = await apiService.login(header);
      if (data?.success) {
        setAuthHeader(header);
        setSessionExpired(false);
        setIsAuthenticated(true);
        return { ok: true };
      }
      return { ok: false, error: 'Login failed' };
    } catch (error) {
      // Never log the error object itself: its config carries the credentials.
      if (axios.isAxiosError(error) && error.response?.status === 401) {
        return { ok: false, error: 'Invalid username or password' };
      }
      return { ok: false, error: getErrorMessage(error, 'Login failed') };
    }
  }, []);

  const logout = useCallback(() => {
    clearAuth();
    setSessionExpired(false);
    setIsAuthenticated(false);
  }, []);

  const value = useMemo(
    () => ({ isAuthenticated, sessionExpired, login, logout }),
    [isAuthenticated, sessionExpired, login, logout]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};
