// Test helpers: a fake axios adapter for the shared `api` instance so tests
// exercise the real request/response interceptors without any network I/O.
import { AxiosError, AxiosHeaders, AxiosResponse, InternalAxiosRequestConfig } from 'axios';
import { api } from './services/api';

export interface MockReply {
  status: number;
  data?: unknown;
}

export type MockHandler = (config: InternalAxiosRequestConfig) => MockReply | Promise<MockReply>;

export const buildResponse = (
  config: InternalAxiosRequestConfig,
  { status, data }: MockReply
): AxiosResponse => ({
  data,
  status,
  statusText: String(status),
  headers: {},
  config,
  request: {},
});

/** An AxiosError as the real adapter would raise it for an HTTP error status. */
export const httpError = (
  status: number,
  data: unknown = {},
  config: InternalAxiosRequestConfig = { headers: new AxiosHeaders() } as InternalAxiosRequestConfig
): AxiosError =>
  new AxiosError(
    `Request failed with status code ${status}`,
    status >= 500 ? AxiosError.ERR_BAD_RESPONSE : AxiosError.ERR_BAD_REQUEST,
    config,
    {},
    buildResponse(config, { status, data })
  );

/** An AxiosError for a request that never got a response (server down). */
export const networkError = (
  config: InternalAxiosRequestConfig = { headers: new AxiosHeaders() } as InternalAxiosRequestConfig
): AxiosError => new AxiosError('Network Error', AxiosError.ERR_NETWORK, config, {});

const originalAdapter = api.defaults.adapter;

/**
 * Routes every request of `api` through `handler`. Returns the list of request
 * configs seen (most useful: `url`, `method`, `params`, `headers`, `data`).
 */
export const mockApi = (handler: MockHandler): InternalAxiosRequestConfig[] => {
  const calls: InternalAxiosRequestConfig[] = [];
  api.defaults.adapter = async (config: InternalAxiosRequestConfig) => {
    calls.push(config);
    const reply = await handler(config);
    const response = buildResponse(config, reply);
    if (reply.status >= 200 && reply.status < 300) {
      return response;
    }
    throw new AxiosError(
      `Request failed with status code ${reply.status}`,
      reply.status >= 500 ? AxiosError.ERR_BAD_RESPONSE : AxiosError.ERR_BAD_REQUEST,
      config,
      {},
      response
    );
  };
  return calls;
};

export const restoreApi = (): void => {
  api.defaults.adapter = originalAdapter;
};

/** Requests must never leave the test process. */
export const failOnRealRequests = (): void => {
  mockApi((config) => {
    throw new Error(`Unexpected request in test: ${config.method?.toUpperCase()} ${config.url}`);
  });
};

export const authHeaderOf = (config: InternalAxiosRequestConfig): string | undefined => {
  const value = AxiosHeaders.from(config.headers).get('Authorization');
  return typeof value === 'string' ? value : undefined;
};

export const paginated = <T,>(data: T[], page = 1, pageSize = 20, total = data.length) => {
  const totalPages = Math.ceil(total / pageSize);
  return {
    data,
    total,
    page,
    page_size: pageSize,
    total_pages: totalPages,
    has_next: page < totalPages,
    has_prev: page > 1,
  };
};
