// src/features/analysis/analysisSlice.js
import { createSlice } from '@reduxjs/toolkit';

const initialState = {
    currentAnalysisData: null, // Will store { results, pdfName, targetLanguage, translatedSections }
};

const analysisSlice = createSlice({
    name: 'analysis',
    initialState,
    reducers: {
        setPersistedAnalysis: (state, action) => {
            // Expects payload: { results, pdfName, targetLanguage, translatedSections }
            state.currentAnalysisData = action.payload;
        },
        updatePersistedTranslations: (state, action) => {
            // Expects payload: { targetLanguage, translatedSections }
            if (state.currentAnalysisData) {
                state.currentAnalysisData.targetLanguage = action.payload.targetLanguage;
                state.currentAnalysisData.translatedSections = action.payload.translatedSections;
            }
        },
        clearPersistedAnalysis: (state) => {
            state.currentAnalysisData = null;
        },
    },
});

export const {
    setPersistedAnalysis,
    updatePersistedTranslations,
    clearPersistedAnalysis,
} = analysisSlice.actions;

// Selectors
export const selectPersistedAnalysisData = (state) => state.analysis.currentAnalysisData;

export default analysisSlice.reducer;