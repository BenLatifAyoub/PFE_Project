// src/features/documents/documentsSlice.js
import { createSlice, createEntityAdapter } from '@reduxjs/toolkit';

// Using createEntityAdapter for potentially more efficient lookups/updates by ID
// It provides { ids: [], entities: {} } structure
const documentsAdapter = createEntityAdapter({
  // Use the 'id' field from your analysis results as the unique identifier
  selectId: (document) => document.id,
  // Optional: Keep the documents sorted by ID or another field
  // sortComparer: (a, b) => a.title.localeCompare(b.title),
});

const initialState = documentsAdapter.getInitialState({
  status: 'idle', // 'idle' | 'loading' | 'succeeded' | 'failed'
  error: null,
});

const documentsSlice = createSlice({
  name: 'documents',
  initialState,
  reducers: {
    // Action to add a single successfully analyzed document
    addDocumentSuccess: (state, action) => {
      // action.payload should be the full document analysis object
      // { id, pdfName, title, sections, figures, tables }
      const documentData = action.payload;
      // Ensure payload has a valid ID before adding
      if (documentData && typeof documentData.id === 'number' && documentData.id !== null) {
         // addOne efficiently adds or updates an entity
        documentsAdapter.addOne(state, documentData);
        state.status = 'succeeded'; // Mark status as succeeded for the add operation
        state.error = null;
      } else {
        console.warn("Attempted to add document without valid ID to Redux:", documentData);
        // Optionally set an error state or just ignore
        state.status = 'failed';
        state.error = 'Invalid document data received (missing or null ID).';
      }
    },
    // Action to remove a document (optional)
    removeDocument: documentsAdapter.removeOne,
    // Action to update a document (optional)
    updateDocument: documentsAdapter.updateOne,
    // Action to set multiple documents (e.g., loading from an API initially)
    setAllDocuments: documentsAdapter.setAll,
    // Reducers for loading status (optional, if you fetch documents separately)
    setDocumentsLoading: (state) => {
      state.status = 'loading';
      state.error = null;
    },
    setDocumentsError: (state, action) => {
      state.status = 'failed';
      state.error = action.payload;
    },
    resetDocumentsStatus: (state) => {
        state.status = 'idle';
        state.error = null;
    }
  },
  // Optional: Add extraReducers for handling async thunks if you fetch documents
});

// Export actions
export const {
  addDocumentSuccess,
  removeDocument,
  updateDocument,
  setAllDocuments,
  setDocumentsLoading,
  setDocumentsError,
  resetDocumentsStatus,
} = documentsSlice.actions;

// Export the adapter's selectors and custom selectors
export const {
  selectAll: selectAllDocuments, // Returns an array of all documents
  selectById: selectDocumentById, // Returns a single document by ID
  selectIds: selectDocumentIds,     // Returns an array of IDs
} = documentsAdapter.getSelectors((state) => state.documents); // Pass the correct slice state path

// Custom selectors
export const selectDocumentsStatus = (state) => state.documents.status;
export const selectDocumentsError = (state) => state.documents.error;

// Export the reducer
export default documentsSlice.reducer;