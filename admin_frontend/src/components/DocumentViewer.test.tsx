import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent, { PointerEventsCheckLevel } from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import DocumentViewer, { PARTIAL_VIEW_BYTES } from './DocumentViewer';
import { DocumentInfo } from '../types/api';
import { failOnRealRequests, mockApi, restoreApi } from '../testUtils';

const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });

const DOCS: DocumentInfo[] = [
  { path: 'docs/README.md', name: 'README.md', type: 'documentation', size: 2048 },
  { path: 'scripts/start_system.sh', name: 'start_system.sh', type: 'script', size: 512 },
  { path: 'logs/worker.log', name: 'worker.log', type: 'log', size: 86 * 1024 * 1024 * 1024 },
];

const renderViewer = () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <DocumentViewer />
    </QueryClientProvider>
  );
};

const viewButtonFor = (name: string) => {
  const item = screen.getByText(name).closest('li') as HTMLElement;
  return item.querySelector('button') as HTMLButtonElement;
};

beforeEach(() => {
  failOnRealRequests();
});

afterEach(() => {
  restoreApi();
});

describe('DocumentViewer', () => {
  it('lists documents grouped by type', async () => {
    mockApi(() => ({ status: 200, data: DOCS }));
    renderViewer();
    expect(await screen.findByText('README.md')).toBeInTheDocument();
    expect(screen.getByText('documentations (1)')).toBeInTheDocument();
    expect(screen.getByText('scripts (1)')).toBeInTheDocument();
    expect(screen.getByText('logs (1)')).toBeInTheDocument();
    expect(screen.getByText('86.0 GB')).toBeInTheDocument();
  });

  it('opens huge logs (the API returns only their tail) and says so', async () => {
    const calls = mockApi((config) =>
      config.url === '/docs/list'
        ? { status: 200, data: DOCS }
        : { status: 200, data: '[... file is 86.0 GiB; showing last 1.0 MiB ...]\nlast line' }
    );
    renderViewer();
    await screen.findByText('worker.log');

    expect(DOCS[2].size).toBeGreaterThan(PARTIAL_VIEW_BYTES);
    expect(screen.getByText('large file: only the last 1 MB is shown')).toBeInTheDocument();
    await user.click(viewButtonFor('worker.log'));

    expect(await screen.findByText(/showing last 1.0 MiB/)).toBeInTheDocument();
    expect(calls.find((c) => c.url === '/docs/read')?.params).toEqual({ path: 'logs/worker.log' });
  });

  it('shows file content as plain text (no HTML injection)', async () => {
    const calls = mockApi((config) =>
      config.url === '/docs/list'
        ? { status: 200, data: DOCS }
        : { status: 200, data: '<img src=x onerror="alert(1)"><b>bold</b>' }
    );
    renderViewer();
    await screen.findByText('README.md');

    await user.click(viewButtonFor('README.md'));

    const pre = await screen.findByText('<img src=x onerror="alert(1)"><b>bold</b>');
    expect(pre.tagName).toBe('PRE');
    expect(pre.querySelector('img, b')).toBeNull();
    const read = calls.find((c) => c.url === '/docs/read');
    expect(read?.params).toEqual({ path: 'docs/README.md' });
  });

  it('filters documents by name or path', async () => {
    mockApi(() => ({ status: 200, data: DOCS }));
    renderViewer();
    await screen.findByText('README.md');

    await user.type(screen.getByPlaceholderText('Search documents...'), 'scripts/');

    await waitFor(() => expect(screen.queryByText('README.md')).not.toBeInTheDocument());
    expect(screen.getByText('start_system.sh')).toBeInTheDocument();
  });

  it('shows an error with the backend detail when listing fails', async () => {
    mockApi(() => ({ status: 500, data: { detail: 'disk error' } }));
    renderViewer();
    expect(await screen.findByText('Failed to load documents')).toBeInTheDocument();
    expect(screen.getByText('disk error')).toBeInTheDocument();
  });

  it('shows an empty state when there are no documents', async () => {
    mockApi(() => ({ status: 200, data: [] }));
    renderViewer();
    expect(await screen.findByText('No documents found')).toBeInTheDocument();
  });

  it('reports a failed read inside the modal', async () => {
    mockApi((config) =>
      config.url === '/docs/list'
        ? { status: 200, data: DOCS }
        : { status: 404, data: { detail: 'File not found' } }
    );
    renderViewer();
    await screen.findByText('README.md');
    await user.click(viewButtonFor('README.md'));
    expect(await screen.findByText('Failed to load document content')).toBeInTheDocument();
    expect(screen.getByText('File not found')).toBeInTheDocument();
  });
});
