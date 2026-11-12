import React, { lazy, useEffect, useState } from 'react';
import { QueryClient, QueryClientProvider, useQueryClient } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import axios from 'axios';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import Login from './components/Login';
import LazyScreen from './components/LazyScreen';
import './App.css';

// The dashboard (data tables, modals, tabs) is a separate chunk, so the login
// page, the first screen of every new tab, does not have to download it.
const loadDashboard = () => import('./components/Dashboard');
const Dashboard = lazy(loadDashboard);

/** Creates the app's QueryClient (exported so tests can build an isolated one). */
export const createQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: {
        // Retry once on network/5xx errors; a 4xx (e.g. 401) will not fix itself.
        retry: (failureCount, error) => {
          const status = axios.isAxiosError(error) ? error.response?.status : undefined;
          if (status !== undefined && status >= 400 && status < 500) {
            return false;
          }
          return failureCount < 1;
        },
        refetchOnWindowFocus: false,
      },
    },
  });

function AppContent() {
  const { isAuthenticated } = useAuth();
  const queryClient = useQueryClient();

  // Drop all cached admin data once logged out (runs after the dashboard,
  // and with it every query observer, has unmounted).
  useEffect(() => {
    if (!isAuthenticated) {
      queryClient.clear();
      // Download the dashboard while the admin types the password, so it shows
      // right after login. A failure here is reported when it is rendered.
      loadDashboard().catch(() => undefined);
    }
  }, [isAuthenticated, queryClient]);

  // Data queries live in <Dashboard/>, so nothing is fetched before login.
  return <LazyScreen>{isAuthenticated ? <Dashboard /> : <Login />}</LazyScreen>;
}

function App() {
  const [queryClient] = useState(createQueryClient);
  return (
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <AppContent />
        <ReactQueryDevtools initialIsOpen={false} />
      </AuthProvider>
    </QueryClientProvider>
  );
}

export default App;
