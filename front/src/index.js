// src/index.js or src/main.jsx

import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import { Provider } from 'react-redux'; // 1. Import Provider
import { store } from './app/store';    // 2. Import your Redux store (adjust path if needed)
import App from './App';
import { rehydrateAuth } from './features/auth/authSlice'; // 3. Import the rehydrate action (adjust path if needed)
import './index.css'; // Assuming you have global styles

// 4. Dispatch the rehydration action *before* the initial render
// This attempts to load saved auth state from localStorage
store.dispatch(rehydrateAuth());

const root = ReactDOM.createRoot(document.getElementById('root'));

root.render(
  // 5. Wrap BrowserRouter (and thus App) with the Provider
  // Pass the imported store instance to the Provider
  <React.StrictMode> {/* Optional: Keep StrictMode if you use it */}
    <Provider store={store}>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </Provider>
  </React.StrictMode>
);