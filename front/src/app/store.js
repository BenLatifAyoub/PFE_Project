// src/app/store.js (or wherever your store is configured)
import { configureStore } from '@reduxjs/toolkit';
import authReducer from '../features/auth/authSlice';
import documentsReducer from '../features/documents/documentsSlice';
import analysisReducer from '../features/analysis/analysisSlice'; // <-- Import
// ... other reducers

export const store = configureStore({
    reducer: {
        auth: authReducer,
        documents: documentsReducer,
        analysis: analysisReducer, // <-- Add reducer
        // ... other reducers
    },
   // Optional: Configure middleware (e.g., disable serializableCheck if storing complex objects, though recommended to keep data serializable)
   middleware: (getDefaultMiddleware) => getDefaultMiddleware({
        serializableCheck: {
            // Ignore these paths in the state if needed, but try to keep data serializable
            // ignoredPaths: ['documents.entities'],
        }
   }),
});