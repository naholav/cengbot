import axios, { AxiosHeaders } from 'axios';
import {
  RawDataItem,
  TrainingDataItem,
  Stats,
  UpdateAnswerRequest,
  UpdateAnswerResponse,
  ApproveResponse,
  PaginatedResponse,
  MutationResponse,
  LoginResponse,
  DocumentInfo,
} from '../types/api';

const DEFAULT_API_URL = 'http://localhost:8001';

/** Base URL of the admin API: `REACT_APP_API_URL` (build time) or localhost:8001. */
export const resolveApiBaseUrl = (raw: string | undefined = import.meta.env.REACT_APP_API_URL): string => {
  const value = (raw ?? '').trim();
  return (value || DEFAULT_API_URL).replace(/\/+$/, '');
};

export const API_BASE_URL = resolveApiBaseUrl();

/** Username pre-filled on the login form (the backend's ADMIN_USERNAME defaults to "admin"). */
export const DEFAULT_ADMIN_USERNAME = (import.meta.env.REACT_APP_ADMIN_USERNAME ?? '').trim() || 'admin';

export const AUTH_STORAGE_KEY = 'adminAuth';
export const LOGIN_PATH = '/auth/login';
const REQUEST_TIMEOUT_MS = 60000;

// ---------------------------------------------------------------------------
// Credential handling.
//
// The API uses HTTP Basic auth, so the header *is* the password (base64 is not
// encryption). It lives in memory and, so that a page reload does not log the
// admin out, in sessionStorage (per tab, wiped when the tab closes). It is
// never written to localStorage; a copy left there by older builds is removed.
// ---------------------------------------------------------------------------

let authHeader: string | null = null;
let unauthorizedHandler: (() => void) | null = null;

const getSessionStorage = (): Storage | null => {
  try {
    return window.sessionStorage;
  } catch {
    return null;
  }
};

/** Builds a Basic auth header; UTF-8 safe (plain `btoa` throws on e.g. "ş", "ğ", "ı"). */
export const encodeBasicAuth = (username: string, password: string): string => {
  const utf8 = encodeURIComponent(`${username}:${password}`).replace(
    /%([0-9A-F]{2})/g,
    (_match, hex: string) => String.fromCharCode(parseInt(hex, 16))
  );
  return `Basic ${btoa(utf8)}`;
};

export const getAuthHeader = (): string | null => authHeader;

export const setAuthHeader = (header: string | null): void => {
  authHeader = header;
  const storage = getSessionStorage();
  try {
    if (header) {
      storage?.setItem(AUTH_STORAGE_KEY, header);
    } else {
      storage?.removeItem(AUTH_STORAGE_KEY);
    }
  } catch {
    // Storage may be unavailable (private mode, quota); memory still works.
  }
};

export const clearAuth = (): void => setAuthHeader(null);

/** Loads credentials saved in this tab's sessionStorage; purges legacy localStorage copies. */
export const restoreAuthFromStorage = (): string | null => {
  try {
    window.localStorage.removeItem(AUTH_STORAGE_KEY);
  } catch {
    // ignore
  }
  let stored: string | null = null;
  try {
    stored = getSessionStorage()?.getItem(AUTH_STORAGE_KEY) ?? null;
  } catch {
    stored = null;
  }
  authHeader = stored && stored.startsWith('Basic ') ? stored : null;
  return authHeader;
};

/**
 * Registers the callback run when an authenticated request gets a 401
 * (credentials changed on the server / session no longer valid).
 * Returns an unregister function.
 */
export const setUnauthorizedHandler = (handler: (() => void) | null): (() => void) => {
  unauthorizedHandler = handler;
  return () => {
    if (unauthorizedHandler === handler) {
      unauthorizedHandler = null;
    }
  };
};

// ---------------------------------------------------------------------------
// Axios instance
// ---------------------------------------------------------------------------

export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: REQUEST_TIMEOUT_MS,
});

api.interceptors.request.use((config) => {
  const headers = AxiosHeaders.from(config.headers);
  if (authHeader && !headers.has('Authorization')) {
    headers.set('Authorization', authHeader);
  }
  config.headers = headers;
  return config;
});

api.interceptors.response.use(
  (response) => response,
  (error: unknown) => {
    if (axios.isAxiosError(error) && error.response?.status === 401 && error.config) {
      const isLoginRequest = error.config.url === LOGIN_PATH;
      const sentHeader = AxiosHeaders.from(error.config.headers).get('Authorization');
      // Only log out if the rejected credentials are still the current ones;
      // a stale request from before a re-login must not log the admin out.
      if (!isLoginRequest && authHeader !== null && sentHeader === authHeader) {
        clearAuth();
        unauthorizedHandler?.();
      }
    }
    return Promise.reject(error);
  }
);

/** Human-readable message for a failed request (uses FastAPI's `detail` when present). */
export const getErrorMessage = (error: unknown, fallback = 'Request failed'): string => {
  if (axios.isAxiosError(error)) {
    if (!error.response) {
      if (error.code === 'ECONNABORTED' || error.code === 'ETIMEDOUT') {
        return 'The admin API did not respond in time';
      }
      return `Cannot reach the admin API at ${API_BASE_URL}`;
    }
    const detail = (error.response.data as { detail?: unknown } | undefined)?.detail;
    if (typeof detail === 'string' && detail) {
      return detail;
    }
    if (Array.isArray(detail)) {
      const msgs = detail
        .map((d) => (d && typeof d === 'object' ? (d as { msg?: unknown }).msg : undefined))
        .filter((m): m is string => typeof m === 'string' && m.length > 0);
      if (msgs.length) {
        return msgs.join('; ');
      }
    }
    return `${fallback} (HTTP ${error.response.status})`;
  }
  if (error instanceof Error && error.message) {
    return error.message;
  }
  return fallback;
};

export const apiService = {
  // Auth: verifies the given credentials without storing them.
  login: async (header: string) => {
    const response = await api.post<LoginResponse>(LOGIN_PATH, undefined, {
      headers: { Authorization: header },
    });
    return response.data;
  },

  // Raw Data
  getRawData: async (page: number = 1, pageSize: number = 20, onlyUnapproved: boolean = false) => {
    const response = await api.get<PaginatedResponse<RawDataItem>>('/raw-data', {
      params: { page, page_size: pageSize, only_unapproved: onlyUnapproved },
    });
    return response.data;
  },

  updateAnswer: async (id: number, data: UpdateAnswerRequest) => {
    const response = await api.put<UpdateAnswerResponse>(`/raw-data/${id}`, data);
    return response.data;
  },

  deleteRawData: async (id: number) => {
    const response = await api.delete<MutationResponse>(`/raw-data/${id}`);
    return response.data;
  },

  approveToTraining: async (id: number) => {
    const response = await api.post<ApproveResponse>(`/approve/${id}`);
    return response.data;
  },

  // Training Data
  getTrainingData: async (page: number = 1, pageSize: number = 20) => {
    const response = await api.get<PaginatedResponse<TrainingDataItem>>('/training-data', {
      params: { page, page_size: pageSize },
    });
    return response.data;
  },

  deleteTrainingData: async (id: number) => {
    const response = await api.delete<MutationResponse>(`/training-data/${id}`);
    return response.data;
  },

  // Stats
  getStats: async () => {
    const response = await api.get<Stats>('/stats');
    return response.data;
  },

  // Documentation
  listDocuments: async () => {
    const response = await api.get<DocumentInfo[]>('/docs/list');
    return response.data;
  },

  readDocument: async (path: string) => {
    // Plain text: never let axios JSON-parse it (a .json/.md file whose body is
    // valid JSON would otherwise become an object and crash rendering).
    const response = await api.get<string>('/docs/read', {
      params: { path },
      responseType: 'text',
      transformResponse: [(data: unknown) => data],
    });
    return typeof response.data === 'string' ? response.data : String(response.data ?? '');
  },
};
