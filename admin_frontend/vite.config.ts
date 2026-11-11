/// <reference types="vitest/config" />
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Dev server and preview stay on localhost:3000: that origin is in the admin
// API's default CORS_ORIGINS, and scripts/_common.sh starts the panel with
// `npm start`.
const LOCAL_SERVER = { host: 'localhost', port: 3000, strictPort: true };

export default defineConfig({
  plugins: [react()],
  // REACT_APP_* (the Create React App prefix) is still accepted, so existing
  // shells, .env files and scripts/deploy_production.sh keep working. Only these
  // prefixes reach the bundle; never put secrets in them.
  envPrefix: ['VITE_', 'REACT_APP_'],
  server: LOCAL_SERVER,
  preview: LOCAL_SERVER,
  build: {
    // scripts/deploy_production.sh copies admin_frontend/build/. to nginx.
    outDir: 'build',
    // Source maps would expose the full TypeScript source to every visitor.
    sourcemap: false,
    // The minifier drops license comments; ship the bundled packages'
    // licenses next to the bundle instead (Create React App's *.LICENSE.txt).
    license: { fileName: 'THIRD_PARTY_LICENSES.md' },
  },
  test: {
    environment: 'jsdom',
    // describe/it/expect/vi as globals; also lets @testing-library/react
    // unmount rendered trees after each test automatically.
    globals: true,
    include: ['src/**/*.test.{ts,tsx}'],
    setupFiles: ['./src/setupTests.ts'],
    // Spies never leak from one test into the next.
    restoreMocks: true,
    // Tests see the built-in defaults, whatever the shell or a local
    // admin_frontend/.env.local sets for the dev server and builds.
    env: { REACT_APP_API_URL: '', REACT_APP_ADMIN_USERNAME: '' },
  },
});
