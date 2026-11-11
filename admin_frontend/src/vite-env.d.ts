/// <reference types="vite/client" />

// Build-time settings (see envPrefix in vite.config.ts); all optional.
interface ImportMetaEnv {
  /** Base URL of the admin API (default http://localhost:8001). */
  readonly REACT_APP_API_URL?: string;
  /** Username pre-filled on the login form (default "admin"). */
  readonly REACT_APP_ADMIN_USERNAME?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
