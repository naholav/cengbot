// Jest setup (picked up automatically by react-scripts).
import '@testing-library/jest-dom';

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
