import React from 'react';
import { act, renderHook, waitFor } from '@testing-library/react';
import { AuthProvider, useAuth } from './AuthContext';
import {
  apiService,
  AUTH_STORAGE_KEY,
  clearAuth,
  encodeBasicAuth,
  getAuthHeader,
} from '../services/api';
import { authHeaderOf, failOnRealRequests, mockApi, restoreApi } from '../testUtils';

const wrapper = ({ children }: { children: React.ReactNode }) => <AuthProvider>{children}</AuthProvider>;

beforeEach(() => {
  failOnRealRequests();
});

afterEach(() => {
  restoreApi();
  clearAuth();
});

describe('AuthProvider', () => {
  it('starts logged out when nothing is stored', () => {
    const { result } = renderHook(() => useAuth(), { wrapper });
    expect(result.current.isAuthenticated).toBe(false);
    expect(result.current.sessionExpired).toBe(false);
  });

  it('logs in with valid credentials and keeps them out of localStorage', async () => {
    const calls = mockApi(() => ({ status: 200, data: { success: true, username: 'admin' } }));
    const { result } = renderHook(() => useAuth(), { wrapper });

    let outcome: Awaited<ReturnType<typeof result.current.login>> | undefined;
    await act(async () => {
      outcome = await result.current.login(' admin ', 'pässword');
    });

    const expected = encodeBasicAuth('admin', 'pässword');
    expect(outcome).toEqual({ ok: true });
    expect(result.current.isAuthenticated).toBe(true);
    expect(authHeaderOf(calls[0])).toBe(expected);
    expect(getAuthHeader()).toBe(expected);
    expect(window.sessionStorage.getItem(AUTH_STORAGE_KEY)).toBe(expected);
    expect(window.localStorage.getItem(AUTH_STORAGE_KEY)).toBeNull();
  });

  it('rejects invalid credentials without storing anything', async () => {
    mockApi(() => ({ status: 401, data: { detail: 'Incorrect username or password' } }));
    const { result } = renderHook(() => useAuth(), { wrapper });

    let outcome: Awaited<ReturnType<typeof result.current.login>> | undefined;
    await act(async () => {
      outcome = await result.current.login('admin', 'wrong');
    });

    expect(outcome).toEqual({ ok: false, error: 'Invalid username or password' });
    expect(result.current.isAuthenticated).toBe(false);
    expect(getAuthHeader()).toBeNull();
    expect(window.sessionStorage.getItem(AUTH_STORAGE_KEY)).toBeNull();
  });

  it('reports an unreachable server distinctly from bad credentials', async () => {
    vi.spyOn(apiService, 'login').mockRejectedValueOnce(
      Object.assign(new Error('Network Error'), { isAxiosError: true, code: 'ERR_NETWORK' })
    );
    const { result } = renderHook(() => useAuth(), { wrapper });

    let outcome: Awaited<ReturnType<typeof result.current.login>> | undefined;
    await act(async () => {
      outcome = await result.current.login('admin', 'pw');
    });

    expect(outcome).toEqual({ ok: false, error: expect.stringMatching(/Cannot reach the admin API/) });
    expect(result.current.isAuthenticated).toBe(false);
  });

  it('does not log credentials to the console on failure', async () => {
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    mockApi(() => ({ status: 401 }));
    const { result } = renderHook(() => useAuth(), { wrapper });
    await act(async () => {
      await result.current.login('admin', 'top-secret');
    });
    const logged = JSON.stringify(errorSpy.mock.calls);
    expect(logged).not.toContain(encodeBasicAuth('admin', 'top-secret'));
    errorSpy.mockRestore();
  });

  it('logout clears the stored credentials', async () => {
    mockApi(() => ({ status: 200, data: { success: true, username: 'admin' } }));
    const { result } = renderHook(() => useAuth(), { wrapper });
    await act(async () => {
      await result.current.login('admin', 'pw');
    });

    act(() => result.current.logout());

    expect(result.current.isAuthenticated).toBe(false);
    expect(getAuthHeader()).toBeNull();
    expect(window.sessionStorage.getItem(AUTH_STORAGE_KEY)).toBeNull();
  });

  it('restores a session from sessionStorage on reload', () => {
    const header = encodeBasicAuth('admin', 'pw');
    window.sessionStorage.setItem(AUTH_STORAGE_KEY, header);
    const { result } = renderHook(() => useAuth(), { wrapper });
    expect(result.current.isAuthenticated).toBe(true);
    expect(getAuthHeader()).toBe(header);
  });

  it('does not trust (and deletes) credentials left in localStorage by older builds', () => {
    window.localStorage.setItem(AUTH_STORAGE_KEY, encodeBasicAuth('admin', 'pw'));
    const { result } = renderHook(() => useAuth(), { wrapper });
    expect(result.current.isAuthenticated).toBe(false);
    expect(window.localStorage.getItem(AUTH_STORAGE_KEY)).toBeNull();
  });

  it('logs out and flags the session as expired when the API answers 401', async () => {
    window.sessionStorage.setItem(AUTH_STORAGE_KEY, encodeBasicAuth('admin', 'old-password'));
    const { result } = renderHook(() => useAuth(), { wrapper });
    expect(result.current.isAuthenticated).toBe(true);

    mockApi(() => ({ status: 401, data: { detail: 'Incorrect username or password' } }));
    await act(async () => {
      await apiService.getStats().catch(() => undefined);
    });

    await waitFor(() => expect(result.current.isAuthenticated).toBe(false));
    expect(result.current.sessionExpired).toBe(true);
    expect(window.sessionStorage.getItem(AUTH_STORAGE_KEY)).toBeNull();
  });

  it('useAuth outside the provider throws a helpful error', () => {
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    expect(() => renderHook(() => useAuth())).toThrow('useAuth must be used within AuthProvider');
    errorSpy.mockRestore();
  });
});
