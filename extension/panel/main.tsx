import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { App } from './app';

// Entry of the side panel bundle (panel.js), loaded by panel.html as a module
// script, so it runs after the document is parsed and #root exists.
const root = document.getElementById('root');
if (root) {
  createRoot(root).render(
    <StrictMode>
      <App />
    </StrictMode>,
  );
}
