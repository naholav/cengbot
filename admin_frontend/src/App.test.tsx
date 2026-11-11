import React from 'react';
import { render, screen, waitFor, within, fireEvent } from '@testing-library/react';
import userEvent, { PointerEventsCheckLevel } from '@testing-library/user-event';
import { InternalAxiosRequestConfig } from 'axios';
import App, { clampPage } from './App';
import { AUTH_STORAGE_KEY, clearAuth, encodeBasicAuth, getAuthHeader } from './services/api';
import { authHeaderOf, failOnRealRequests, MockReply, mockApi, paginated, restoreApi } from './testUtils';
import { RawDataItem } from './types/api';

const HEADER = encodeBasicAuth('admin', 'pw');

const STATS = {
  total_questions: 42,
  approved_questions: 7,
  liked_questions: 5,
  disliked_questions: 2,
  training_data_count: 7,
  duplicate_count: 3,
  approval_rate: '16.7%',
};

const rawRow = (overrides: Partial<RawDataItem>): RawDataItem => ({
  id: 1,
  telegram_id: 1001,
  telegram_message_id: 55,
  username: 'student1',
  question: 'Where is the CENG building?',
  answer: 'Next to the library.',
  language: 'en',
  like: null,
  admin_approved: 0,
  is_duplicate: false,
  duplicate_of_id: null,
  similarity_score: null,
  created_at: '2025-11-06T07:19:04.738283',
  answered_at: null,
  message_thread_id: null,
  likes: 0,
  dislikes: 0,
  total_votes: 0,
  vote_score: 0,
  ...overrides,
});

const RAW_ROWS = [
  rawRow({ id: 1 }),
  rawRow({ id: 2, question: 'When is the final exam?', answer: null, username: null }),
];

type Route = (config: InternalAxiosRequestConfig) => MockReply | undefined;

/** Fake backend: default happy-path responses, overridable per test. */
const mockBackend = (override?: Route) =>
  mockApi((config) => {
    const custom = override?.(config);
    if (custom) return custom;
    const key = `${config.method?.toUpperCase()} ${config.url}`;
    if (key === 'GET /stats') return { status: 200, data: STATS };
    if (key === 'GET /raw-data') return { status: 200, data: paginated(RAW_ROWS) };
    if (key === 'GET /training-data') return { status: 200, data: paginated([]) };
    if (key === 'GET /docs/list') return { status: 200, data: [] };
    if (key.startsWith('PUT /raw-data/')) return { status: 200, data: { success: true, message: 'ok' } };
    if (key.startsWith('POST /approve/')) {
      return { status: 200, data: { success: true, message: 'ok', training_data_id: 9 } };
    }
    return { status: 404, data: { detail: `No mock for ${key}` } };
  });

const loggedIn = () => window.sessionStorage.setItem(AUTH_STORAGE_KEY, HEADER);

const rawDataCalls = (calls: InternalAxiosRequestConfig[]) => calls.filter((c) => c.url === '/raw-data');

// antd tables are heavy in jsdom; give the integration tests some headroom.
jest.setTimeout(30000);
// Skipping the per-click pointer-events check avoids thousands of slow jsdom
// getComputedStyle() calls over antd's CSS-in-JS styles.
const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });

/** Clicks the <button> containing the given visible text (cheaper than *ByRole on a big table). */
const clickButtonWithText = async (text: string) => {
  const label = await screen.findByText(text);
  await user.click(label.closest('button') as HTMLElement);
};

beforeEach(() => {
  failOnRealRequests();
});

afterEach(() => {
  restoreApi();
  clearAuth();
});

describe('clampPage', () => {
  it('keeps the page when it exists and steps back when it does not', () => {
    expect(clampPage(2, 3)).toBe(2);
    expect(clampPage(4, 3)).toBe(3);
    expect(clampPage(3, 0)).toBe(1);
    expect(clampPage(5, undefined)).toBe(5);
  });
});

describe('App', () => {
  it('shows the login form and fetches no data when logged out', async () => {
    const calls = mockBackend();
    render(<App />);
    expect(screen.getByText('Admin Login')).toBeInTheDocument();
    await new Promise((r) => setTimeout(r, 50));
    expect(calls).toHaveLength(0);
  });

  it('renders stats and raw data for a restored session, sending credentials', async () => {
    loggedIn();
    const calls = mockBackend();
    render(<App />);

    expect(await screen.findByText('Where is the CENG building?')).toBeInTheDocument();
    expect(screen.getByText('When is the final exam?')).toBeInTheDocument();
    expect(screen.getByText('No answer')).toBeInTheDocument();
    expect(await screen.findByText('Total Questions')).toBeInTheDocument();
    expect(screen.getByText('42')).toBeInTheDocument();

    const first = rawDataCalls(calls)[0];
    expect(first.params).toEqual({ page: 1, page_size: 20, only_unapproved: false });
    expect(calls.every((c) => authHeaderOf(c) === HEADER)).toBe(true);
  });

  it('shows an error instead of an empty table when loading fails', async () => {
    loggedIn();
    mockBackend((config) =>
      config.url === '/raw-data' ? { status: 500, data: { detail: 'database is locked' } } : undefined
    );
    render(<App />);
    expect(await screen.findByText('Failed to load raw data', {}, { timeout: 4000 })).toBeInTheDocument();
    expect(screen.getByText('database is locked')).toBeInTheDocument();
  });

  it('logout returns to the login form and forgets the credentials', async () => {
    loggedIn();
    mockBackend();
    render(<App />);
    await screen.findByText('Where is the CENG building?');

    await clickButtonWithText('Logout');

    expect(await screen.findByText('Admin Login')).toBeInTheDocument();
    expect(window.sessionStorage.getItem(AUTH_STORAGE_KEY)).toBeNull();
    expect(getAuthHeader()).toBeNull();
    expect(screen.queryByText('Where is the CENG building?')).not.toBeInTheDocument();
  });

  it('a 401 from the API sends the admin back to the login form', async () => {
    loggedIn();
    mockBackend(() => ({ status: 401, data: { detail: 'Incorrect username or password' } }));
    render(<App />);

    expect(await screen.findByText('Admin Login')).toBeInTheDocument();
    expect(screen.getByText(/session is no longer valid/i)).toBeInTheDocument();
    expect(window.sessionStorage.getItem(AUTH_STORAGE_KEY)).toBeNull();
  });

  it('lets the admin write an answer for an unanswered question', async () => {
    loggedIn();
    const calls = mockBackend();
    render(<App />);
    const row = (await screen.findByText('When is the final exam?')).closest('tr') as HTMLElement;

    await user.click(within(row).getByLabelText('Write answer'));
    const saveButton = within(row).getByText('Save').closest('button') as HTMLElement;
    expect(saveButton).toBeDisabled(); // empty answers are not saved

    fireEvent.change(within(row).getByLabelText('Answer text'), { target: { value: 'On January 10th.' } });
    await user.click(within(row).getByText('Save').closest('button') as HTMLElement);

    await waitFor(() => {
      const put = calls.find((c) => c.method === 'put');
      expect(put?.url).toBe('/raw-data/2');
      expect(JSON.parse(put?.data)).toEqual({ answer: 'On January 10th.' });
    });
    await waitFor(() => expect(screen.queryByLabelText('Answer text')).not.toBeInTheDocument());
  });

  it('keeps the editor open when saving fails', async () => {
    loggedIn();
    mockBackend((config) =>
      config.method === 'put' ? { status: 404, data: { detail: 'Item not found' } } : undefined
    );
    render(<App />);
    const row = (await screen.findByText('Where is the CENG building?')).closest('tr') as HTMLElement;

    await user.click(within(row).getByLabelText('Edit answer'));
    await user.click(within(row).getByText('Save').closest('button') as HTMLElement);

    expect(await screen.findByText('Item not found')).toBeInTheDocument();
    expect(within(row).getByLabelText('Answer text')).toHaveValue('Next to the library.');
  });

  it('approves a row through POST /approve/:id', async () => {
    loggedIn();
    const calls = mockBackend();
    render(<App />);
    const row = (await screen.findByText('Where is the CENG building?')).closest('tr') as HTMLElement;

    await user.click(within(row).getByLabelText('Approve to training data'));
    await clickButtonWithText('Yes');

    await waitFor(() => expect(calls.some((c) => c.method === 'post' && c.url === '/approve/1')).toBe(true));
    // Data is refreshed after approving.
    await waitFor(() => expect(calls.filter((c) => c.url === '/stats').length).toBeGreaterThan(1));
  });

  it('the "Pending only" status filter is applied by the server', async () => {
    loggedIn();
    const calls = mockBackend();
    render(<App />);
    await screen.findByText('Where is the CENG building?');

    const statusHeader = screen.getByText('Status').closest('th') as HTMLElement;
    fireEvent.click(statusHeader.querySelector('.ant-table-filter-trigger') as HTMLElement);
    await user.click(await screen.findByText('Pending only'));
    await clickButtonWithText('OK');

    await waitFor(() =>
      expect(rawDataCalls(calls).some((c) => c.params?.only_unapproved === true && c.params?.page === 1)).toBe(true)
    );
  });

  it('steps back to the last existing page when the current page disappears', async () => {
    loggedIn();
    const calls = mockBackend((config) => {
      if (config.url !== '/raw-data') return undefined;
      if (config.params?.page === 2) {
        // Rows were deleted elsewhere: only one page left.
        return { status: 200, data: paginated([], 2, 20, 20) };
      }
      return { status: 200, data: paginated(RAW_ROWS, 1, 20, 25) };
    });
    render(<App />);
    await screen.findByText('Where is the CENG building?');

    await user.click(screen.getByTitle('2'));

    await waitFor(() => expect(rawDataCalls(calls).some((c) => c.params?.page === 2)).toBe(true));
    // Page 2 came back empty with total_pages=1: the table returns to page 1
    // (served from cache) instead of showing an empty page 2.
    await waitFor(() =>
      expect(document.querySelector('.ant-pagination-item-active')).toHaveAttribute('title', '1')
    );
    expect(await screen.findByText('Where is the CENG building?')).toBeInTheDocument();
  });
});
