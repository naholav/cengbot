import {
  api,
  apiService,
  AUTH_STORAGE_KEY,
  clearAuth,
  encodeBasicAuth,
  getAuthHeader,
  getErrorMessage,
  resolveApiBaseUrl,
  restoreAuthFromStorage,
  setAuthHeader,
  setUnauthorizedHandler,
} from './api';
import { authHeaderOf, httpError, mockApi, networkError, paginated, restoreApi } from '../testUtils';

const HEADER = encodeBasicAuth('admin', 'secret');

/** Decodes a Basic header back to its UTF-8 "user:password" string. */
const decodeBasic = (header: string): string => {
  const binary = atob(header.replace(/^Basic /, ''));
  const percentEncoded = Array.from(binary)
    .map((ch) => `%${ch.charCodeAt(0).toString(16).padStart(2, '0')}`)
    .join('');
  return decodeURIComponent(percentEncoded);
};

afterEach(() => {
  restoreApi();
  clearAuth();
  setUnauthorizedHandler(null);
});

describe('resolveApiBaseUrl', () => {
  it('defaults to localhost:8001', () => {
    expect(resolveApiBaseUrl(undefined)).toBe('http://localhost:8001');
    expect(resolveApiBaseUrl('   ')).toBe('http://localhost:8001');
  });

  it('uses REACT_APP_API_URL and strips trailing slashes', () => {
    expect(resolveApiBaseUrl('https://admin.example.edu/api/')).toBe('https://admin.example.edu/api');
    expect(resolveApiBaseUrl(' http://10.0.0.5:8001 ')).toBe('http://10.0.0.5:8001');
  });

  it('is the base URL of the shared axios instance', () => {
    expect(api.defaults.baseURL).toBe(resolveApiBaseUrl());
  });
});

describe('encodeBasicAuth', () => {
  it('builds a standard Basic header', () => {
    expect(encodeBasicAuth('admin', 'pw')).toBe(`Basic ${btoa('admin:pw')}`);
  });

  it('does not throw on non-Latin1 (Turkish) characters and encodes them as UTF-8', () => {
    const header = encodeBasicAuth('admin', 'şifreĞı€');
    expect(header.startsWith('Basic ')).toBe(true);
    expect(decodeBasic(header)).toBe('admin:şifreĞı€');
  });
});

describe('credential storage', () => {
  it('keeps credentials in memory and sessionStorage, never localStorage', () => {
    setAuthHeader(HEADER);
    expect(getAuthHeader()).toBe(HEADER);
    expect(window.sessionStorage.getItem(AUTH_STORAGE_KEY)).toBe(HEADER);
    expect(window.localStorage.getItem(AUTH_STORAGE_KEY)).toBeNull();
  });

  it('clearAuth removes them everywhere', () => {
    setAuthHeader(HEADER);
    clearAuth();
    expect(getAuthHeader()).toBeNull();
    expect(window.sessionStorage.getItem(AUTH_STORAGE_KEY)).toBeNull();
  });

  it('restore purges the legacy plaintext localStorage copy and does not use it', () => {
    window.localStorage.setItem(AUTH_STORAGE_KEY, HEADER);
    expect(restoreAuthFromStorage()).toBeNull();
    expect(getAuthHeader()).toBeNull();
    expect(window.localStorage.getItem(AUTH_STORAGE_KEY)).toBeNull();
  });

  it('restore reads this tab\'s sessionStorage', () => {
    window.sessionStorage.setItem(AUTH_STORAGE_KEY, HEADER);
    expect(restoreAuthFromStorage()).toBe(HEADER);
    expect(getAuthHeader()).toBe(HEADER);
  });

  it('restore ignores malformed values', () => {
    window.sessionStorage.setItem(AUTH_STORAGE_KEY, 'garbage');
    expect(restoreAuthFromStorage()).toBeNull();
  });
});

describe('request interceptor', () => {
  it('attaches the current credentials to API requests', async () => {
    setAuthHeader(HEADER);
    const calls = mockApi(() => ({ status: 200, data: { total_questions: 1 } }));
    await apiService.getStats();
    expect(calls).toHaveLength(1);
    expect(authHeaderOf(calls[0])).toBe(HEADER);
  });

  it('sends no Authorization header when logged out', async () => {
    const calls = mockApi(() => ({ status: 200, data: {} }));
    await apiService.getStats();
    expect(authHeaderOf(calls[0])).toBeUndefined();
  });

  it('login uses the credentials being verified, not stored ones', async () => {
    setAuthHeader(HEADER);
    const candidate = encodeBasicAuth('admin', 'new-password');
    const calls = mockApi(() => ({ status: 200, data: { success: true, username: 'admin' } }));
    await expect(apiService.login(candidate)).resolves.toEqual({ success: true, username: 'admin' });
    expect(calls[0].method).toBe('post');
    expect(calls[0].url).toBe('/auth/login');
    expect(authHeaderOf(calls[0])).toBe(candidate);
  });
});

describe('401 handling', () => {
  it('clears credentials and notifies the app when an authenticated request is rejected', async () => {
    setAuthHeader(HEADER);
    const onUnauthorized = jest.fn();
    setUnauthorizedHandler(onUnauthorized);
    mockApi(() => ({ status: 401, data: { detail: 'Incorrect username or password' } }));

    await expect(apiService.getStats()).rejects.toMatchObject({ response: { status: 401 } });
    expect(onUnauthorized).toHaveBeenCalledTimes(1);
    expect(getAuthHeader()).toBeNull();
    expect(window.sessionStorage.getItem(AUTH_STORAGE_KEY)).toBeNull();
  });

  it('does not treat a failed login attempt as a session expiry', async () => {
    setAuthHeader(HEADER);
    const onUnauthorized = jest.fn();
    setUnauthorizedHandler(onUnauthorized);
    mockApi(() => ({ status: 401, data: {} }));

    await expect(apiService.login(encodeBasicAuth('admin', 'wrong'))).rejects.toBeTruthy();
    expect(onUnauthorized).not.toHaveBeenCalled();
    expect(getAuthHeader()).toBe(HEADER);
  });

  it('ignores a 401 for credentials that are no longer current', async () => {
    const stale = encodeBasicAuth('admin', 'old');
    setAuthHeader(HEADER);
    const onUnauthorized = jest.fn();
    setUnauthorizedHandler(onUnauthorized);
    mockApi(() => ({ status: 401, data: {} }));

    await expect(api.get('/stats', { headers: { Authorization: stale } })).rejects.toBeTruthy();
    expect(onUnauthorized).not.toHaveBeenCalled();
    expect(getAuthHeader()).toBe(HEADER);
  });

  it('does not log out on server errors', async () => {
    setAuthHeader(HEADER);
    const onUnauthorized = jest.fn();
    setUnauthorizedHandler(onUnauthorized);
    mockApi(() => ({ status: 500, data: { detail: 'boom' } }));

    await expect(apiService.getStats()).rejects.toBeTruthy();
    expect(onUnauthorized).not.toHaveBeenCalled();
    expect(getAuthHeader()).toBe(HEADER);
  });

  it('the unregister function only removes its own handler', () => {
    const first = jest.fn();
    const second = jest.fn();
    const unregisterFirst = setUnauthorizedHandler(first);
    setUnauthorizedHandler(second);
    unregisterFirst();
    setAuthHeader(HEADER);
    mockApi(() => ({ status: 401 }));
    return apiService.getStats().catch(() => {
      expect(first).not.toHaveBeenCalled();
      expect(second).toHaveBeenCalledTimes(1);
    });
  });
});

describe('endpoints match the backend contract', () => {
  it('GET /raw-data with page, page_size and only_unapproved', async () => {
    const body = paginated([{ id: 1, question: 'q', admin_approved: 0 }], 2, 50, 51);
    const calls = mockApi(() => ({ status: 200, data: body }));
    await expect(apiService.getRawData(2, 50, true)).resolves.toEqual(body);
    expect(calls[0].method).toBe('get');
    expect(calls[0].url).toBe('/raw-data');
    expect(calls[0].params).toEqual({ page: 2, page_size: 50, only_unapproved: true });
  });

  it('GET /training-data with page and page_size', async () => {
    const calls = mockApi(() => ({ status: 200, data: paginated([]) }));
    await apiService.getTrainingData(3, 10);
    expect(calls[0].url).toBe('/training-data');
    expect(calls[0].params).toEqual({ page: 3, page_size: 10 });
  });

  it('PUT /raw-data/:id with {answer}', async () => {
    const calls = mockApi(() => ({ status: 200, data: { success: true, message: 'ok' } }));
    await apiService.updateAnswer(7, { answer: 'new answer' });
    expect(calls[0].method).toBe('put');
    expect(calls[0].url).toBe('/raw-data/7');
    expect(JSON.parse(calls[0].data)).toEqual({ answer: 'new answer' });
  });

  it('POST /approve/:id, DELETE /raw-data/:id and /training-data/:id', async () => {
    const calls = mockApi(() => ({ status: 200, data: { success: true, message: 'ok' } }));
    await apiService.approveToTraining(4);
    await apiService.deleteRawData(5);
    await apiService.deleteTrainingData(6);
    expect(calls.map((c) => `${c.method} ${c.url}`)).toEqual([
      'post /approve/4',
      'delete /raw-data/5',
      'delete /training-data/6',
    ]);
  });

  it('GET /docs/list and /docs/read?path=...', async () => {
    const calls = mockApi((config) =>
      config.url === '/docs/list'
        ? { status: 200, data: [{ path: 'README.md', name: 'README.md', type: 'documentation', size: 10 }] }
        : { status: 200, data: '# Title' }
    );
    await expect(apiService.listDocuments()).resolves.toHaveLength(1);
    await expect(apiService.readDocument('docs/API.md')).resolves.toBe('# Title');
    expect(calls[1].url).toBe('/docs/read');
    expect(calls[1].params).toEqual({ path: 'docs/API.md' });
  });

  it('readDocument returns raw text even when the file content is valid JSON', async () => {
    mockApi(() => ({ status: 200, data: '{"a": 1}' }));
    await expect(apiService.readDocument('x.json')).resolves.toBe('{"a": 1}');
  });
});

describe('getErrorMessage', () => {
  it('uses FastAPI string detail', () => {
    expect(getErrorMessage(httpError(400, { detail: 'Already in training data' }))).toBe('Already in training data');
  });

  it('joins FastAPI validation errors', () => {
    const err = httpError(422, { detail: [{ msg: 'field required' }, { msg: 'too long' }] });
    expect(getErrorMessage(err)).toBe('field required; too long');
  });

  it('falls back to the status code', () => {
    expect(getErrorMessage(httpError(502, '<html>bad gateway</html>'), 'Failed')).toBe('Failed (HTTP 502)');
  });

  it('explains unreachable servers', () => {
    expect(getErrorMessage(networkError())).toMatch(/Cannot reach the admin API/);
  });

  it('handles plain errors and unknown values', () => {
    expect(getErrorMessage(new Error('oops'))).toBe('oops');
    expect(getErrorMessage('???', 'fallback')).toBe('fallback');
  });
});
