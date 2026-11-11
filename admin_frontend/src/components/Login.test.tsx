import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent, { PointerEventsCheckLevel } from '@testing-library/user-event';
import Login from './Login';
import { AuthProvider } from '../contexts/AuthContext';
import { apiService, clearAuth, encodeBasicAuth, getAuthHeader } from '../services/api';
import { authHeaderOf, failOnRealRequests, mockApi, restoreApi } from '../testUtils';

const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
const loginButton = () => screen.getByText('Login').closest('button') as HTMLElement;

const renderLogin = () =>
  render(
    <AuthProvider>
      <Login />
    </AuthProvider>
  );

beforeEach(() => {
  failOnRealRequests();
});

afterEach(() => {
  restoreApi();
  clearAuth();
  vi.restoreAllMocks();
});

describe('Login', () => {
  it('pre-fills the default admin username and has no hardcoded password', () => {
    renderLogin();
    expect(screen.getByLabelText('Username')).toHaveValue('admin');
    expect(screen.getByLabelText('Password')).toHaveValue('');
  });

  it('requires a password before calling the API', async () => {
    const loginSpy = vi.spyOn(apiService, 'login');
    renderLogin();
    await user.click(loginButton());
    expect(await screen.findByText('Please enter the password!')).toBeInTheDocument();
    expect(loginSpy).not.toHaveBeenCalled();
  });

  it('submits the typed username and password as Basic credentials', async () => {
    const calls = mockApi(() => ({ status: 200, data: { success: true, username: 'operator' } }));
    renderLogin();

    const username = screen.getByLabelText('Username');
    await user.clear(username);
    await user.type(username, 'operator');
    await user.type(screen.getByLabelText('Password'), 'çok-gizli');
    await user.click(loginButton());

    const expected = encodeBasicAuth('operator', 'çok-gizli');
    await waitFor(() => expect(getAuthHeader()).toBe(expected));
    expect(calls).toHaveLength(1);
    expect(calls[0].url).toBe('/auth/login');
    expect(authHeaderOf(calls[0])).toBe(expected);
  });

  it('shows an error for wrong credentials', async () => {
    mockApi(() => ({ status: 401, data: { detail: 'Incorrect username or password' } }));
    renderLogin();

    await user.type(screen.getByLabelText('Password'), 'wrong');
    await user.click(loginButton());

    expect(await screen.findByText('Invalid username or password')).toBeInTheDocument();
    expect(getAuthHeader()).toBeNull();
  });

  it('shows a connection error when the API is down', async () => {
    vi.spyOn(apiService, 'login').mockRejectedValueOnce(
      Object.assign(new Error('Network Error'), { isAxiosError: true, code: 'ERR_NETWORK' })
    );
    renderLogin();

    await user.type(screen.getByLabelText('Password'), 'pw');
    await user.click(loginButton());

    expect(await screen.findByText(/Cannot reach the admin API/)).toBeInTheDocument();
  });
});
