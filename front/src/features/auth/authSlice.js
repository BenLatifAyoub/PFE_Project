// src/features/auth/authSlice.js
import { createSlice } from '@reduxjs/toolkit';

// Define a maximum number of recent IDs to keep (optional)
const MAX_RECENT_ANALYSES = 10;

const initialState = {
  // User object now includes the list of recent IDs
  user: null, // e.g., { username: 'test', email: '...', recentlyAnalyzedIds: [46, 42] }
  token: null,
  isAuthenticated: false,
  status: 'idle', // 'idle' | 'loading' | 'succeeded' | 'failed'
  error: null,
};

// Helper to safely update localStorage
const updateUserInLocalStorage = (user) => {
    if (user) {
        localStorage.setItem('user', JSON.stringify(user));
    } else {
        localStorage.removeItem('user');
    }
};

const authSlice = createSlice({
  name: 'auth',
  initialState,
  reducers: {
    loginStart: (state) => {
      state.status = 'loading';
      state.error = null;
      state.isAuthenticated = false;
    },
    loginSuccess: (state, action) => {
      state.status = 'succeeded';
      // Expect payload to be the user data object, potentially including recentlyAnalyzedIds
      state.user = action.payload.user || action.payload;
      state.token = action.payload.token || null;
      state.isAuthenticated = true;
      state.error = null;

      // --- Initialize recentlyAnalyzedIds if not present ---
      if (state.user && !Array.isArray(state.user.recentlyAnalyzedIds)) {
          console.log("Initializing recentlyAnalyzedIds for user:", state.user.username);
          state.user.recentlyAnalyzedIds = [];
      }
      // --- End initialization ---

      // Persist to localStorage
      updateUserInLocalStorage(state.user); // Save the whole user object
      if (state.token) {
          localStorage.setItem('token', state.token);
      }
    },
    loginFailure: (state, action) => {
      state.status = 'failed';
      state.error = action.payload;
      state.user = null; // Clear user
      state.token = null;
      state.isAuthenticated = false;
      // Clear localStorage
      localStorage.removeItem('user');
      localStorage.removeItem('token');
    },
    logout: (state) => {
      state.user = null; // Clear user
      state.token = null;
      state.isAuthenticated = false;
      state.status = 'idle';
      state.error = null;
      // Clear localStorage
      localStorage.removeItem('user');
      localStorage.removeItem('token');
    },
    rehydrateAuth: (state) => {
        try {
            const storedUserString = localStorage.getItem('user');
            const token = localStorage.getItem('token');
            if (storedUserString) {
                const storedUser = JSON.parse(storedUserString);
                state.user = storedUser;
                // --- Ensure recentlyAnalyzedIds exists after rehydration ---
                if (state.user && !Array.isArray(state.user.recentlyAnalyzedIds)) {
                    state.user.recentlyAnalyzedIds = [];
                }
                // --- End check ---
                state.isAuthenticated = true;
                state.status = 'succeeded'; // Assume valid if stored
            }
            if (token) {
                state.token = token;
            }
        } catch (e) {
            console.error("Could not rehydrate auth state:", e);
            // Reset state and clear storage on error
            Object.assign(state, initialState);
            localStorage.removeItem('user');
            localStorage.removeItem('token');
        }
     },

    // --- New Reducer to add a recently analyzed ID ---
    addRecentAnalysis: (state, action) => {
        const analysisId = action.payload;

        // Ensure user exists and payload is a valid number
        if (state.user && typeof analysisId === 'number' && analysisId !== null) {
            // Ensure the recentlyAnalyzedIds array exists
             if (!Array.isArray(state.user.recentlyAnalyzedIds)) {
                state.user.recentlyAnalyzedIds = [];
            }

            // Remove existing instance of the ID (to move it to the front/top)
            const existingIndex = state.user.recentlyAnalyzedIds.indexOf(analysisId);
            if (existingIndex > -1) {
                state.user.recentlyAnalyzedIds.splice(existingIndex, 1);
            }

            // Add the new ID to the beginning of the array (most recent first)
            state.user.recentlyAnalyzedIds.unshift(analysisId);

            // Optional: Limit the array size
            if (state.user.recentlyAnalyzedIds.length > MAX_RECENT_ANALYSES) {
                state.user.recentlyAnalyzedIds = state.user.recentlyAnalyzedIds.slice(0, MAX_RECENT_ANALYSES);
            }

            console.log(`Added/Updated analysis ID ${analysisId} to recent list. New list:`, state.user.recentlyAnalyzedIds);

            // --- Update localStorage with the modified user object ---
            updateUserInLocalStorage(state.user);

        } else {
             if (!state.user) {
                console.warn("addRecentAnalysis called but no user is logged in.");
             } else {
                 console.warn("addRecentAnalysis called with invalid ID:", analysisId);
             }
        }
    }
    // --- End new reducer ---
  },
});

// Export actions, including the new one
export const {
    loginStart,
    loginSuccess,
    loginFailure,
    logout,
    rehydrateAuth,
    addRecentAnalysis // <-- Export the new action
} = authSlice.actions;

// Export selectors
export const selectCurrentUser = (state) => state.auth.user;
export const selectIsAuthenticated = (state) => state.auth.isAuthenticated;
export const selectAuthToken = (state) => state.auth.token;
export const selectAuthStatus = (state) => state.auth.status;
export const selectAuthError = (state) => state.auth.error;
// Optional: Selector specifically for the recent IDs
export const selectUserRecentlyAnalyzedIds = (state) => state.auth.user?.recentlyAnalyzedIds || [];


// Export the reducer
export default authSlice.reducer;