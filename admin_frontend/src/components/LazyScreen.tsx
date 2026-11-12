import React, { Suspense } from 'react';
import { Alert, Button, Spin } from 'antd';

/** Centered spinner shown while a lazily loaded screen's code is downloading. */
export const ScreenSpinner: React.FC<{ fullHeight?: boolean }> = ({ fullHeight = true }) => (
  <div
    style={{
      minHeight: fullHeight ? '100vh' : 200,
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
    }}
  >
    <Spin size="large" aria-label="Loading" />
  </div>
);

interface BoundaryState {
  error: Error | null;
}

/**
 * Shows a reload prompt instead of a blank page when a screen fails to load or
 * render. The usual cause is a stale tab: a deployment replaced the build, so
 * the old chunk file names no longer exist on the server.
 */
export class LoadErrorBoundary extends React.Component<{ children: React.ReactNode }, BoundaryState> {
  state: BoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): BoundaryState {
    return { error };
  }

  render() {
    if (this.state.error) {
      return (
        <Alert
          type="error"
          showIcon
          message="Failed to load the admin panel"
          description="It may have been updated since this page was opened. Reload the page to continue."
          action={<Button onClick={() => window.location.reload()}>Reload</Button>}
          style={{ margin: 24 }}
        />
      );
    }
    return this.props.children;
  }
}

/** Renders a React.lazy() screen with a spinner while it loads and a reload prompt if it fails. */
const LazyScreen: React.FC<{ children: React.ReactNode; fullHeight?: boolean }> = ({
  children,
  fullHeight = true,
}) => (
  <LoadErrorBoundary>
    <Suspense fallback={<ScreenSpinner fullHeight={fullHeight} />}>{children}</Suspense>
  </LoadErrorBoundary>
);

export default LazyScreen;
