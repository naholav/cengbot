import React from 'react';
import { act, renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { message } from 'antd';
import type { MockInstance } from 'vitest';
import {
  POLL_INTERVAL_MS,
  queryKeys,
  useApproveToTraining,
  useDeleteRawData,
  useDeleteTrainingData,
  useRawData,
  useUpdateAnswer,
} from './useApiQueries';
import { failOnRealRequests, mockApi, paginated, restoreApi } from '../testUtils';

const makeClient = () =>
  new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });

const wrapperFor = (client: QueryClient) => ({ children }: { children: React.ReactNode }) => (
  <QueryClientProvider client={client}>{children}</QueryClientProvider>
);

let successSpy: MockInstance;
let errorSpy: MockInstance;

beforeEach(() => {
  failOnRealRequests();
  successSpy = vi.spyOn(message, 'success').mockImplementation(() => (() => {}) as never);
  errorSpy = vi.spyOn(message, 'error').mockImplementation(() => (() => {}) as never);
});

afterEach(() => {
  restoreApi();
  vi.restoreAllMocks();
  vi.useRealTimers();
});

const invalidatedKeys = (spy: MockInstance) =>
  spy.mock.calls.map(([filters]) => (filters as { queryKey: unknown[] }).queryKey[0]).sort();

describe('queryKeys', () => {
  it('include every parameter so pages and filters are cached separately', () => {
    expect(queryKeys.rawData(2, 50, true)).toEqual(['rawData', 2, 50, true]);
    expect(queryKeys.rawData(2, 50, false)).not.toEqual(queryKeys.rawData(2, 50, true));
    expect(queryKeys.trainingData(3, 10)).toEqual(['trainingData', 3, 10]);
    expect(queryKeys.documentContent('a.md')).toEqual(['documents', 'content', 'a.md']);
  });
});

describe('useRawData', () => {
  it('fetches the requested page', async () => {
    const body = paginated([{ id: 1 }], 2, 10, 11);
    const calls = mockApi(() => ({ status: 200, data: body }));
    const client = makeClient();
    const { result } = renderHook(() => useRawData(2, 10, true), { wrapper: wrapperFor(client) });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data).toEqual(body);
    expect(calls[0].params).toEqual({ page: 2, page_size: 10, only_unapproved: true });
  });

  it('polls every 30 s only when polling is enabled', async () => {
    vi.useFakeTimers();
    const calls = mockApi(() => ({ status: 200, data: paginated([]) }));
    const client = makeClient();
    const { result, rerender } = renderHook(({ poll }) => useRawData(1, 20, false, { poll }), {
      wrapper: wrapperFor(client),
      initialProps: { poll: false },
    });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(calls).toHaveLength(1);

    await act(async () => {
      vi.advanceTimersByTime(POLL_INTERVAL_MS * 3);
    });
    expect(calls).toHaveLength(1);

    rerender({ poll: true });
    await act(async () => {
      vi.advanceTimersByTime(POLL_INTERVAL_MS + 10);
    });
    await waitFor(() => expect(calls.length).toBeGreaterThanOrEqual(2));
    expect(calls.length).toBeLessThanOrEqual(3);
  });
});

describe('mutations', () => {
  it('approve invalidates raw data, training data and stats', async () => {
    const calls = mockApi(() => ({ status: 200, data: { success: true, message: 'ok', training_data_id: 3 } }));
    const client = makeClient();
    const invalidate = vi.spyOn(client, 'invalidateQueries');
    const { result } = renderHook(() => useApproveToTraining(), { wrapper: wrapperFor(client) });

    await act(async () => {
      await result.current.mutateAsync(5);
    });

    expect(calls[0].url).toBe('/approve/5');
    expect(invalidatedKeys(invalidate)).toEqual(['rawData', 'stats', 'trainingData']);
    expect(successSpy).toHaveBeenCalled();
  });

  it('deleting raw data also refreshes training data (its copy is deleted too)', async () => {
    mockApi(() => ({ status: 200, data: { success: true, message: 'ok' } }));
    const client = makeClient();
    const invalidate = vi.spyOn(client, 'invalidateQueries');
    const { result } = renderHook(() => useDeleteRawData(), { wrapper: wrapperFor(client) });

    await act(async () => {
      await result.current.mutateAsync(5);
    });

    expect(invalidatedKeys(invalidate)).toEqual(['rawData', 'stats', 'trainingData']);
  });

  it('removing training data refreshes all three views', async () => {
    mockApi(() => ({ status: 200, data: { success: true, message: 'ok' } }));
    const client = makeClient();
    const invalidate = vi.spyOn(client, 'invalidateQueries');
    const { result } = renderHook(() => useDeleteTrainingData(), { wrapper: wrapperFor(client) });

    await act(async () => {
      await result.current.mutateAsync(8);
    });

    expect(invalidatedKeys(invalidate)).toEqual(['rawData', 'stats', 'trainingData']);
  });

  it('update answer sends the new text and refreshes raw data', async () => {
    const calls = mockApi(() => ({ status: 200, data: { success: true, message: 'ok' } }));
    const client = makeClient();
    const invalidate = vi.spyOn(client, 'invalidateQueries');
    const { result } = renderHook(() => useUpdateAnswer(), { wrapper: wrapperFor(client) });

    await act(async () => {
      await result.current.mutateAsync({ id: 4, answer: 'fixed' });
    });

    expect(calls[0].url).toBe('/raw-data/4');
    expect(JSON.parse(calls[0].data)).toEqual({ answer: 'fixed' });
    expect(invalidatedKeys(invalidate)).toEqual(['rawData']);
  });

  it('update answer also refreshes training data when the backend synced the training copy', async () => {
    mockApi(() => ({ status: 200, data: { success: true, message: 'ok', training_data_updated: true } }));
    const client = makeClient();
    const invalidate = vi.spyOn(client, 'invalidateQueries');
    const { result } = renderHook(() => useUpdateAnswer(), { wrapper: wrapperFor(client) });

    await act(async () => {
      await result.current.mutateAsync({ id: 4, answer: 'fixed' });
    });

    expect(invalidatedKeys(invalidate)).toEqual(['rawData', 'trainingData']);
  });

  it('shows the backend error detail instead of a generic message', async () => {
    mockApi(() => ({ status: 400, data: { detail: 'Already in training data' } }));
    const client = makeClient();
    const invalidate = vi.spyOn(client, 'invalidateQueries');
    const { result } = renderHook(() => useApproveToTraining(), { wrapper: wrapperFor(client) });

    await act(async () => {
      await result.current.mutateAsync(5).catch(() => undefined);
    });

    expect(errorSpy).toHaveBeenCalledWith('Already in training data');
    expect(invalidate).not.toHaveBeenCalled();
  });
});
