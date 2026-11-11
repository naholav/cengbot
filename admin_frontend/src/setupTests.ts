// Test setup, run before every test file (setupFiles in vite.config.ts).
import '@testing-library/jest-dom/vitest';

// @testing-library/dom's waitFor() only advances fake timers when it sees a
// global `jest` (it checks `typeof jest` plus `setTimeout.clock`). Under
// vi.useFakeTimers() it would otherwise wait forever on frozen timers, so give
// it the one call it makes. With real timers this changes nothing.
Object.defineProperty(globalThis, 'jest', {
  configurable: true,
  value: { advanceTimersByTime: (ms: number) => vi.advanceTimersByTime(ms) },
});

// jsdom reports exceptions thrown inside event listeners (React 18 re-throws
// render errors that way in development) straight to the worker's original
// console, bypassing vi.spyOn(console, 'error'). Route them through the test's
// console, as Jest's jsdom environment did, so tests that expect an error and
// silence console.error stay quiet.
type JsdomError = Error & { cause?: { stack?: string } };
interface JsdomVirtualConsole {
  removeAllListeners(event: 'jsdomError'): void;
  on(event: 'jsdomError', listener: (error: JsdomError) => void): void;
}
const { virtualConsole } = (globalThis as unknown as { jsdom: { virtualConsole: JsdomVirtualConsole } }).jsdom;
virtualConsole.removeAllListeners('jsdomError');
virtualConsole.on('jsdomError', (error) => console.error(error.cause?.stack ?? error.message));

// antd's responsive helpers (Grid, Table) need matchMedia, which jsdom lacks.
if (!window.matchMedia) {
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    configurable: true,
    value: (query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: () => {},
      removeListener: () => {},
      addEventListener: () => {},
      removeEventListener: () => {},
      dispatchEvent: () => false,
    }),
  });
}

// jsdom does not implement getComputedStyle(el, pseudoElt) and logs an error
// each time antd measures the scrollbar; ignore the pseudo-element argument.
const originalGetComputedStyle = window.getComputedStyle.bind(window);
window.getComputedStyle = (elt: Element) => originalGetComputedStyle(elt);

beforeEach(() => {
  window.sessionStorage.clear();
  window.localStorage.clear();
});
