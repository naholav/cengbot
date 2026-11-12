import React, { lazy } from 'react';
import { act, render, screen } from '@testing-library/react';
import userEvent, { PointerEventsCheckLevel } from '@testing-library/user-event';
import LazyScreen from './LazyScreen';

const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });

type Loaded = { default: React.ComponentType };

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('LazyScreen', () => {
  it('shows a spinner until the lazy screen has loaded', async () => {
    let finishLoading: (module: Loaded) => void = () => undefined;
    const Screen = lazy(() => new Promise<Loaded>((resolve) => { finishLoading = resolve; }));
    render(
      <LazyScreen>
        <Screen />
      </LazyScreen>
    );

    expect(screen.getByLabelText('Loading')).toBeInTheDocument();
    expect(screen.queryByText('Screen content')).not.toBeInTheDocument();

    await act(async () => finishLoading({ default: () => <p>Screen content</p> }));

    expect(await screen.findByText('Screen content')).toBeInTheDocument();
    expect(screen.queryByLabelText('Loading')).not.toBeInTheDocument();
  });

  it('offers a reload instead of a blank page when the chunk cannot be loaded', async () => {
    // React reports the caught error on the console; keep the test output clean.
    vi.spyOn(console, 'error').mockImplementation(() => undefined);
    const reload = vi.fn();
    vi.stubGlobal('location', { ...window.location, reload });
    // What a stale tab gets after a deployment removed the old chunk files.
    const Screen = lazy(() => Promise.reject(new TypeError('Failed to fetch dynamically imported module')));
    render(
      <LazyScreen>
        <Screen />
      </LazyScreen>
    );

    expect(await screen.findByText('Failed to load the admin panel')).toBeInTheDocument();
    await user.click(screen.getByText('Reload').closest('button') as HTMLElement);
    expect(reload).toHaveBeenCalledTimes(1);
  });
});
