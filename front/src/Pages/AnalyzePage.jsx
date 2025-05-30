import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { useDispatch, useSelector } from 'react-redux';
import jsPDF from 'jspdf';

// Redux Hooks and Actions/Selectors
import { addDocumentSuccess } from '../features/documents/documentsSlice';
import { addRecentAnalysis, selectCurrentUser } from '../features/auth/authSlice';
import {
    setPersistedAnalysis,
    updatePersistedTranslations,
    clearPersistedAnalysis,
    selectPersistedAnalysisData,
} from '../features/analysis/analysisSlice';

// Child Components & Styling
import ExpandableText from '../components/ExpandableText';
import './analyze.css';
import { FaVolumeUp, FaStopCircle, FaEdit, FaSave, FaTimesCircle } from 'react-icons/fa';

// Constants
const SUPPORTED_LANGUAGES = [
    { code: '', name: 'Select Language...' }, { code: 'en', name: 'English' },
    { code: 'es', name: 'Spanish' }, { code: 'fr', name: 'French' },
    { code: 'de', name: 'German' }, { code: 'ja', name: 'Japanese' },
    { code: 'ar', name: 'Arabic' }, { code: 'zh-cn', name: 'Chinese (Simplified)' },
];
const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:5000';

const AnalyzePage = () => {
    const navigate = useNavigate();
    const dispatch = useDispatch();
    const currentUser = useSelector(selectCurrentUser);
    const persistedAnalysisData = useSelector(selectPersistedAnalysisData);

    const [loading, setLoading] = useState(false);
    const [message, setMessage] = useState('');
    const [messageType, setMessageType] = useState('');
    const [pdfFile, setPdfFile] = useState(null);
    const [results, setResults] = useState(null);
    const [targetLanguage, setTargetLanguage] = useState('');
    const [translatedSections, setTranslatedSections] = useState({});
    const [translationLoading, setTranslationLoading] = useState(false);
    const [translationError, setTranslationError] = useState('');
    const [ttsLoading, setTtsLoading] = useState({});
    const [ttsError, setTtsError] = useState({});
    const [currentAudio, setCurrentAudio] = useState(null);
    const audioUrlRef = useRef(null);
    const [pdfNameForDisplay, setPdfNameForDisplay] = useState('');
    const [isDownloadingPdf, setIsDownloadingPdf] = useState(false);

    const [editingSection, setEditingSection] = useState(null);
    const [editText, setEditText] = useState('');
    const [editLoading, setEditLoading] = useState(false);
    const [editError, setEditError] = useState('');

    useEffect(() => {
        if (!pdfFile && persistedAnalysisData) {
            // Ensure sections from persisted data are also an array
            let persistedResults = persistedAnalysisData.results;
            if (persistedResults && persistedResults.sections && !Array.isArray(persistedResults.sections)) {
                console.warn("Persisted sections data was not an array. Converting.");
                 // This conversion logic might need adjustment if persisted format is object
                persistedResults.sections = Object.entries(persistedResults.sections).map(([h, c]) => ({
                    heading: h,
                    summary: (c && c.summary) || null,
                    full_text: (c && c.full_text) || ""
                }));
            }
            setResults(persistedResults);
            setTargetLanguage(persistedAnalysisData.targetLanguage || '');
            setTranslatedSections(persistedAnalysisData.translatedSections || {});
            setPdfNameForDisplay(persistedAnalysisData.pdfName || '');
        }
    }, [persistedAnalysisData, pdfFile, dispatch]);

    const updateRecentAnalysisInDB = async (username, analysisId) => {
        if (!username || typeof analysisId !== 'number') return;
        try {
            await axios.patch(`${BACKEND_URL}/api/users/${username}/recent_analyses`, { analysis_id: analysisId });
        } catch (error) {
            console.error(`Failed to update recent analysis list in DB for user ${username}:`, error.response?.data || error.message);
        }
    };

    useEffect(() => () => stopCurrentAudio(), []);

    const handleFileChange = (event) => {
        const file = event.target.files[0];
        stopCurrentAudio();
        setEditingSection(null); setEditText(''); setEditError('');
        if (file && file.type === "application/pdf") {
            setPdfFile(file); setPdfNameForDisplay(file.name);
            setMessage(''); setMessageType(''); setResults(null); setTargetLanguage('');
            setTranslatedSections({}); setTranslationLoading(false); setTranslationError('');
            setTtsLoading({}); setTtsError({}); dispatch(clearPersistedAnalysis());
        } else {
            setPdfFile(null); setPdfNameForDisplay('');
            if (event.target.value) { setMessage('Please select a valid PDF file.'); setMessageType('error'); }
            else { setMessage(''); setMessageType(''); }
            setResults(null);
        }
    };

    const handleAnalyzeRFP = async () => {
        if (!pdfFile) { setMessage('Please upload a PDF first.'); setMessageType('error'); return; }
        setLoading(true); setMessage('Processing PDF...'); setMessageType(''); setResults(null);
        setTargetLanguage(''); setTranslatedSections({}); setTranslationLoading(false);
        setTranslationError(''); stopCurrentAudio(); setTtsLoading({}); setTtsError({});
        dispatch(clearPersistedAnalysis());
        setEditingSection(null); setEditText(''); setEditError('');

        try {
            const formData = new FormData(); formData.append('pdf', pdfFile);
            const backendResponse = await axios.post(`${BACKEND_URL}/api/process-pdf`, formData, {
                headers: { 'Content-Type': 'multipart/form-data' }, timeout: 500000
            });
            const { id, title, sections: rawSectionsFromServer, figures, tables, message: backendMessage, chroma_status } = backendResponse.data;

            if (typeof id !== 'number' && id !== null) throw new Error("Invalid analysis ID received from backend.");
            if (!title || typeof title !== 'string') throw new Error("Invalid analysis title received.");
            if (!Array.isArray(figures)) throw new Error("Invalid figures data.");
            if (!Array.isArray(tables)) throw new Error("Invalid tables data.");

            // Backend should now always send sections as an array.
            // Perform a check and handle if it's not an array for robustness.
            let processedSections;
            if (Array.isArray(rawSectionsFromServer)) {
                processedSections = rawSectionsFromServer;
            } else {
                console.error("Backend did not return sections as an array:", rawSectionsFromServer);
                // Fallback: if it's an object, convert (order might be alphabetical)
                if (rawSectionsFromServer && typeof rawSectionsFromServer === 'object') {
                    processedSections = Object.entries(rawSectionsFromServer).map(([h, c_val]) => ({
                        heading: h,
                        summary: (c_val && c_val.summary) || null,
                        full_text: (c_val && c_val.full_text) || ""
                    }));
                    console.warn("Converted object sections to array on frontend. Backend should ideally send an array in document order.");
                } else {
                    // If not an object or array, or if null/undefined
                    console.error("Invalid sections data received. Expected array or object, got:", typeof rawSectionsFromServer);
                    processedSections = []; // Default to empty array
                }
            }
            
            const processedData = { 
                id, pdfName: pdfFile.name, title, sections: processedSections, 
                figures: figures || [], tables: tables || [], 
                analysisDate: new Date().toISOString() 
            };
            setResults(processedData); 
            setPdfNameForDisplay(pdfFile.name);
            dispatch(setPersistedAnalysis({ 
                results: processedData, pdfName: pdfFile.name, 
                targetLanguage: '', translatedSections: {} 
            }));

            let successMsg = backendMessage || 'PDF processed successfully.';
            const isSaved = typeof id === 'number' && id !== null;
            let finalMsgType = 'success';

            if (isSaved) {
                if (chroma_status) successMsg += ` Vector data status: ${chroma_status}.`;
                successMsg += ` Analysis saved (ID: ${id}).`;
                try {
                    dispatch(addDocumentSuccess(processedData)); dispatch(addRecentAnalysis(id));
                    if (currentUser?.username) updateRecentAnalysisInDB(currentUser.username, id);
                } catch (reduxError) { successMsg += " (Note: Failed to update global document state)"; }
            } else {
                successMsg = `PDF processed, but failed to save. Results shown. Chroma: ${chroma_status || 'Skipped'}.`;
                finalMsgType = 'warning';
            }
            setMessage(successMsg); setMessageType(finalMsgType);
        } catch (error) {
            const errorDetail = error.response?.data?.details || error.response?.data?.error || error.message;
            setMessage(`Failed to analyze PDF: ${errorDetail}`); setMessageType('error');
            setResults(null); setPdfNameForDisplay(''); dispatch(clearPersistedAnalysis());
        } finally { setLoading(false); }
    };

const handleChat = () => {
    if (results?.id !== null && typeof results?.id === 'number') {
        stopCurrentAudio(); navigate('/chat', { state: { contextId: results.id, pdfName: results.pdfName || currentPdfName || "Analyzed Document" } });
    } else { setMessage("Analyze and save a document first."); setMessageType('error'); }
};
const handleCourse = () => {
    if (results?.id !== null && typeof results?.id === 'number') {
        stopCurrentAudio(); navigate('/course', { state: { contextId: results.id, pdfName: results.pdfName || currentPdfName || "Analyzed Document" } });
    } else { setMessage("Analyze and save a document first."); setMessageType('error'); }
};
const handleAnalyzeNewRFP = () => {
    setPdfFile(null); setPdfNameForDisplay(''); setMessage(''); setMessageType(''); setResults(null);
    setTargetLanguage(''); setTranslatedSections({}); setTranslationLoading(false); setTranslationError('');
    stopCurrentAudio(); setTtsLoading({}); setTtsError({}); dispatch(clearPersistedAnalysis());
    setEditingSection(null); setEditText(''); setEditError(''); // Clear editing state
    const fileInput = document.querySelector('input[type="file"]'); if (fileInput) fileInput.value = '';
};
const handleReturnHome = () => { stopCurrentAudio(); navigate('/home'); };

const createDataUrl = (base64String, mimeType = 'image/png') => {
    if (!base64String || typeof base64String !== 'string') return null;
    if (base64String.startsWith('data:')) return base64String;
    if (/^[A-Za-z0-9+/=]+$/.test(base64String.replace(/\s/g, ''))) {
        try { return `data:${mimeType};base64,${base64String}`; } catch (e) { return null; }
    }
    return null;
};

const handleLanguageChange = async (event) => {
    const newLang = event.target.value;
    setTargetLanguage(newLang); setTranslationError(''); stopCurrentAudio();
    if (!newLang || !results?.sections?.length) {
        setTranslationLoading(false); setTranslatedSections({});
        if (persistedAnalysisData) dispatch(updatePersistedTranslations({ targetLanguage: newLang, translatedSections: {} }));
        return;
    }
    setTranslationLoading(true);
    const translate = async (txt, lang, head, type) => {
        if (!txt?.trim()) return { heading: head, type, text: null };
        try {
            const resp = await axios.post(`${BACKEND_URL}/api/translate`, { text: txt, target_language: lang });
            return { heading: head, type, text: resp.data?.translated_text || `[Translation Error: ${resp.data?.error || 'Invalid Resp'}]` };
        } catch (err) { return { heading: head, type, text: `[Translation Error: ${err.response?.data?.error || 'API Fail'}]` }; }
    };
    const promises = results.sections.filter(s => s.summary?.trim()).map((s, i) => translate(s.summary, newLang, s.heading || `Section_${i}`, 'summary'));
    if (!promises.length) {
        setTranslationLoading(false); setTranslationError("No text to translate."); setTranslatedSections({});
        if (persistedAnalysisData) dispatch(updatePersistedTranslations({ targetLanguage: newLang, translatedSections: {} }));
        return;
    }
    try {
        const settled = await Promise.allSettled(promises);
        const newTrans = {}; let hasErr = false;
        settled.forEach(res => {
            if (res.status === 'fulfilled' && res.value) {
                const { heading, type, text } = res.value; if (text === null) return;
                if (!newTrans[heading]) newTrans[heading] = {};
                if (type === 'summary') newTrans[heading].translated_summary = text;
                if (typeof text === 'string' && text.startsWith('[Translation Error:')) hasErr = true;
            } else if (res.status === 'rejected') hasErr = true;
        });
        setTranslatedSections(newTrans);
        if (persistedAnalysisData) dispatch(updatePersistedTranslations({ targetLanguage: newLang, translatedSections: newTrans }));
        if (hasErr) setTranslationError("Some sections could not be translated.");
    } catch (err) { setTranslationError("Error processing translations."); }
    finally { setTranslationLoading(false); }
};

const stopCurrentAudio = () => {
    if (currentAudio?.audio) { currentAudio.audio.pause(); currentAudio.audio.currentTime = 0; currentAudio.audio.onended = null; currentAudio.audio.onpause = null; currentAudio.audio.onerror = null; }
    if (audioUrlRef.current) { URL.revokeObjectURL(audioUrlRef.current); audioUrlRef.current = null; }
    setCurrentAudio(null);
};

const handlePlayTTS = async (text, id, language = 'en') => {
    if (!text?.trim() || text.startsWith('[Translation Error:')) {
        setTtsError(p => ({ ...p, [id]: "Cannot play empty/error text." })); if (ttsLoading[id]) setTtsLoading(p => ({ ...p, [id]: false })); return;
    }
    stopCurrentAudio(); setTtsLoading(p => ({ ...p, [id]: true })); setTtsError(p => ({ ...p, [id]: null }));
    try {
        const resp = await axios.post(`${BACKEND_URL}/api/tts`, { text, language }, { responseType: 'json', timeout: 180000 });
        if (resp.data?.audio_base64) {
            const { audio_base64, format = 'wav' } = resp.data;
            const binStr = window.atob(audio_base64);
            const bytes = new Uint8Array(binStr.length);
            for (let i = 0; i < binStr.length; i++) bytes[i] = binStr.charCodeAt(i);
            const blob = new Blob([bytes], { type: `audio/${format}` });
            const url = URL.createObjectURL(blob); audioUrlRef.current = url;
            const audio = new Audio(url); setCurrentAudio({ id, audio });
            audio.play().catch(() => { setTtsError(p => ({ ...p, [id]: "Browser blocked audio." })); stopCurrentAudio(); });
            audio.onended = () => { if (currentAudio?.id === id) stopCurrentAudio(); };
            audio.onpause = () => { if (currentAudio?.id === id && !audio.ended) stopCurrentAudio(); };
            audio.onerror = () => { setTtsError(p => ({ ...p, [id]: "Audio playback failed." })); if (currentAudio?.id === id) stopCurrentAudio(); };
        } else { setTtsError(p => ({ ...p, [id]: `TTS failed: ${resp.data?.error || "Unknown TTS issue."}` })); }
    } catch (err) {
        const detail = err.response?.data?.error || err.message;
        if (axios.isCancel(err) || err.code === 'ECONNABORTED') setTtsError(p => ({ ...p, [id]: `TTS timed out.` }));
        else setTtsError(p => ({ ...p, [id]: `TTS failed: ${detail}` }));
        stopCurrentAudio();
    } finally { setTtsLoading(p => ({ ...p, [id]: false })); }
};

// --- PDF Generation Helpers (defined in component scope) ---
const _addTextToPdf = (doc, text, fontSize, yPos, options) => {
    const { isBold = false, indent = 0, customYStep, margin, maxWidth, pageHeight } = options;
    let currentY = yPos;
    const textStr = String(text || '').trim();

    if (textStr === "" && customYStep === undefined) return currentY;
    if (textStr === "" && customYStep !== undefined) { currentY += customYStep; return currentY; }
    
    doc.setFontSize(fontSize);
    doc.setFont('helvetica', isBold ? 'bold' : 'normal');
    const lines = doc.splitTextToSize(textStr, maxWidth - indent);
    const lineHeight = (doc.getLineHeight(textStr) / doc.internal.scaleFactor) * 0.85;

    lines.forEach(line => {
        if (currentY + lineHeight > pageHeight - margin) {
            doc.addPage(); currentY = margin;
        }
        doc.text(line, margin + indent, currentY);
        currentY += lineHeight;
    });
    currentY += customYStep !== undefined ? customYStep : 3;
    return currentY;
};

const _addImageToPdf = (doc, base64Img, descriptionText, yPos, options) => {
    const { imageType = "Image", margin, maxWidth, pageHeight } = options;
    let currentY = yPos;

    if (!base64Img) return currentY;

    try {
        const imgDataUrl = createDataUrl(base64Img);
        if (!imgDataUrl) {
            return _addTextToPdf(doc, `[${imageType} for "${descriptionText || 'untitled'}" data invalid]`, 9, currentY, { ...options, isBold: false, indent: 5, customYStep: 2 });
        }
        const props = doc.getImageProperties(imgDataUrl);
        const aspectRatio = props.width / props.height;
        let imgWidthMm = maxWidth;
        let imgHeightMm = imgWidthMm / aspectRatio;

        const descText = descriptionText ? `${imageType} Description: ${descriptionText}` : "";
        const descLines = descText ? doc.splitTextToSize(descText, maxWidth) : [];
        const descLineHeight = descText ? (doc.getLineHeight(descText) / doc.internal.scaleFactor) * 0.85 : 0;
        const spaceForDesc = descText ? (descLines.length * descLineHeight + 2) : 0;
        const maxImgHeightOnPage = pageHeight - (2 * margin) - spaceForDesc - 5;

        if (imgHeightMm > maxImgHeightOnPage && maxImgHeightOnPage > 0) {
            imgHeightMm = maxImgHeightOnPage;
            imgWidthMm = imgHeightMm * aspectRatio;
        }
        if (imgWidthMm > maxWidth) {
            imgWidthMm = maxWidth;
            imgHeightMm = imgWidthMm / aspectRatio;
        }
        const totalSpaceNeeded = imgHeightMm + spaceForDesc + 7; 

        if (currentY + totalSpaceNeeded > pageHeight - margin) {
            doc.addPage(); currentY = margin;
        }
        if (descText) {
            currentY = _addTextToPdf(doc, descText, 9, currentY, { ...options, isBold: false, indent: 0, customYStep: 2 });
        }
        doc.addImage(imgDataUrl, props.fileType.toUpperCase(), margin + (maxWidth - imgWidthMm) / 2, currentY, imgWidthMm, imgHeightMm);
        currentY += imgHeightMm + 7;
        return currentY;
    } catch (e) {
        return _addTextToPdf(doc, `[Error rendering ${imageType.toLowerCase()}: "${descriptionText || 'untitled'}"]`, 9, currentY, { ...options, isBold: false, indent: 5, customYStep: 2 });
    }
};

const handleDownloadAllPDF = async () => {
    if (!results || !results.sections) {
        setMessage("No analysis results to download."); setMessageType('error'); return;
    }
    setIsDownloadingPdf(true);
    setMessage("Generating PDF, please wait..."); setMessageType('info');

    await new Promise(resolve => setTimeout(resolve, 100)); 

    const doc = new jsPDF({ orientation: 'p', unit: 'mm', format: 'a4' });
    let yPos = 15;
    const pageHeight = doc.internal.pageSize.getHeight();
    const pageWidth = doc.internal.pageSize.getWidth();
    const margin = 15;
    const maxWidth = pageWidth - 2 * margin;
    const pdfOptions = { margin, maxWidth, pageHeight };

    const docTitle = results.title && results.title !== "No title found" ? results.title : (pdfNameForDisplay || "Analyzed Document");
    yPos = _addTextToPdf(doc, `Summary Report for: ${docTitle}`, 18, yPos, { ...pdfOptions, isBold: true, customYStep: 8 });
    if (results.id !== null && results.id !== undefined) {
        yPos = _addTextToPdf(doc, `Analysis ID: ${results.id}`, 10, yPos, { ...pdfOptions, customYStep: 5 });
    }
    
    results.sections.forEach((section, index) => {
        if (index > 0) {
            if (yPos + 10 > pageHeight - margin) { doc.addPage(); yPos = margin; } 
            else { yPos += 5; }
            doc.setLineWidth(0.3); doc.line(margin, yPos, pageWidth - margin, yPos); yPos += 5;
        }
        if (yPos > pageHeight - margin - 20) { doc.addPage(); yPos = margin; }

        yPos = _addTextToPdf(doc, `Section: ${section.heading || `Section ${index + 1}`}`, 14, yPos, { ...pdfOptions, isBold: true, customYStep: 4 });
        
        const currentSummary = section.summary || "Not available.";
        yPos = _addTextToPdf(doc, "Original Summary:", 11, yPos, { ...pdfOptions, isBold: true, customYStep: 2 });
        yPos = _addTextToPdf(doc, currentSummary, 10, yPos, { ...pdfOptions, indent: 0, customYStep: 3});

        const translated = translatedSections[section.heading || `Section ${index + 1}`] || {};
        const translatedSummaryText = translated.translated_summary;
        const targetLangInfo = SUPPORTED_LANGUAGES.find(l => l.code === targetLanguage);
        const langDisplayName = targetLangInfo?.name || (targetLanguage && targetLanguage.toUpperCase());

        if (langDisplayName && translatedSummaryText) {
             if (!String(translatedSummaryText).startsWith("[Translation Error:")) {
                yPos = _addTextToPdf(doc, `Translated Summary (to ${langDisplayName}):`, 11, yPos, { ...pdfOptions, isBold: true, customYStep: 2 });
                yPos = _addTextToPdf(doc, translatedSummaryText, 10, yPos, { ...pdfOptions, indent: 0, customYStep: 3 });
             } else {
                yPos = _addTextToPdf(doc, `Translated Summary (to ${langDisplayName}): Translation failed or not available.`, 10, yPos, { ...pdfOptions, indent: 0, customYStep: 3 });
             }
        }
        
        if (section.full_text?.trim()) {
            yPos = _addTextToPdf(doc, "Original Full Text:", 11, yPos, { ...pdfOptions, isBold: true, customYStep: 2 });
            yPos = _addTextToPdf(doc, section.full_text, 10, yPos, { ...pdfOptions, indent: 0, customYStep: 5 });
        } else {
            yPos = _addTextToPdf(doc, "Original Full Text: Not available.", 10, yPos, { ...pdfOptions, indent: 0, customYStep: 5 });
        }
    });

    if (results.figures?.length > 0) {
        if (yPos + 25 > pageHeight - margin) { doc.addPage(); yPos = margin; } else { yPos += 10; }
        yPos = _addTextToPdf(doc, "Figures from the Document", 16, yPos, { ...pdfOptions, isBold: true, customYStep: 6 });
        results.figures.forEach(([desc, img], i) => {
            yPos = _addImageToPdf(doc, img, desc || `Figure ${i + 1}`, yPos, { ...pdfOptions, imageType: "Figure" });
        });
    }
    if (results.tables?.length > 0) {
        if (yPos + 25 > pageHeight - margin) { doc.addPage(); yPos = margin; } else { yPos += 10; }
        yPos = _addTextToPdf(doc, "Tables from the Document", 16, yPos, { ...pdfOptions, isBold: true, customYStep: 6 });
        results.tables.forEach(([desc, img], i) => {
            yPos = _addImageToPdf(doc, img, desc || `Table ${i + 1}`, yPos, { ...pdfOptions, imageType: "Table" });
        });
    }
    const safeDocTitle = docTitle.replace(/[^a-z0-9_ .-]/gi, '').replace(/\s+/g, '_').substring(0, 50);
    doc.save(`${safeDocTitle}_Full_Report.pdf`);
    
    setMessage("Full report PDF generated and download initiated."); setMessageType('success');
    setIsDownloadingPdf(false);
};

    const handleEditSummary = (index, currentSummary) => {
        if (!results?.id) {
            setMessage("Document must be saved (have an ID) to edit summaries.");
            setMessageType('error'); return;
        }
        stopCurrentAudio();
        setEditingSection({ index, type: 'summary' }); // Store index
        setEditText(currentSummary);
        setEditError(''); setMessage('');
    };

    const handleCancelEditSummary = () => {
        setEditingSection(null); setEditText(''); setEditError('');
    };

    const handleEditTextChange = (event) => {
        setEditText(event.target.value);
    };

    const handleSaveEditedSummary = async () => {
        if (!editingSection || editingSection.type !== 'summary' || !results?.id) {
            setEditError("Cannot save: No active edit session or missing analysis ID."); return;
        }
        setEditLoading(true); setEditError(''); setMessage(''); setMessageType('');

        const { index } = editingSection; // Use index to get the section from local state
        const analysisId = results.id;
        
        if (!results.sections || index < 0 || index >= results.sections.length) {
            setEditError("Cannot save: Section not found locally.");
            setEditLoading(false); return;
        }

        const sectionToUpdate = results.sections[index];
        if (!sectionToUpdate || !sectionToUpdate.heading) {
            setEditError("Cannot save: Section heading is missing for the target section.");
            setEditLoading(false); return;
        }
        const sectionHeading = sectionToUpdate.heading; // Get the heading to send to backend

        try {
            const response = await axios.patch(
                `${BACKEND_URL}/api/analyses/${analysisId}/sections/summary`, // UPDATED URL
                { 
                    heading: sectionHeading, // Send heading
                    summary: editText 
                },
                { headers: { ...(currentUser?.token && {'Authorization': `Bearer ${currentUser.token}`}) } }
            );

            if (response.data && (response.status === 200 || response.status === 201)) {
                // Local state update still uses the index
                const updatedSections = results.sections.map((sec, i) =>
                    i === index ? { ...sec, summary: editText } : sec
                );
                const updatedResults = { ...results, sections: updatedSections };
                setResults(updatedResults);
                dispatch(setPersistedAnalysis({
                    results: updatedResults,
                    pdfName: pdfNameForDisplay,
                    targetLanguage: targetLanguage,
                    translatedSections: translatedSections,
                }));
                setMessage('Summary updated successfully.'); setMessageType('success');
                handleCancelEditSummary(); 
            } else {
                throw new Error(response.data?.error || "Failed to update summary: Invalid server response.");
            }
        } catch (error) {
            console.error("Error saving summary:", error);
            const errorMsg = error.response?.data?.error || error.message || "An unknown error occurred while saving.";
            setEditError(`Failed to save summary: ${errorMsg}`);
            setMessage(`Failed to save summary: ${errorMsg}`);
            setMessageType('error');
        } finally {
            setEditLoading(false);
        }
    };

    const currentPdfName = pdfFile ? pdfFile.name : pdfNameForDisplay;

    return (
        <div className="analyze-page-container">
            <div className="analyze-main-content">
                <div className="analyze-control-panel">
                    {/* ... (Control panel JSX - no changes from previous full code) ... */}
                    <h2 className="control-panel-title">Upload and Analyze PDF</h2>
                    <div className="file-input-container">
                        <input type="file" accept=".pdf" onChange={handleFileChange} disabled={loading || isDownloadingPdf || editLoading} className="file-input-element" aria-label="Select PDF file" />
                        <div className="action-buttons">
                            <button onClick={handleReturnHome} className="button button-home" disabled={isDownloadingPdf || editLoading}>Home</button>
                            <button onClick={handleAnalyzeNewRFP} className="button button-clear" disabled={isDownloadingPdf || editLoading}>Clear / New PDF</button>
                            <button onClick={handleAnalyzeRFP} disabled={!pdfFile || loading || isDownloadingPdf || editLoading} className="button button-analyze">
                                {loading ? 'Analyzing...' : 'Analyze PDF'}
                            </button>
                            <button
                                onClick={handleDownloadAllPDF}
                                className="button button-download-all-pdf"
                                disabled={!results || loading || isDownloadingPdf || editLoading}
                                title="Download all summaries, texts, and media as a single PDF"
                            >
                                {isDownloadingPdf ? 'Generating PDF...' : 'Download Full Report PDF'}
                            </button>
                        </div>
                    </div>
                    <div className="status-area">
                        {(loading || isDownloadingPdf || editLoading) && (
                            <div className="loading-indicator" role="status">
                                <div className="loading-spinner" aria-hidden="true"></div>
                                <p>{message || (loading ? 'Analyzing PDF...' : (isDownloadingPdf ? 'Generating PDF...' : (editLoading ? 'Saving edit...' : 'Processing...')))}</p>
                            </div>
                        )}
                        {!loading && !isDownloadingPdf && !editLoading && message && (
                            <p role="alert" className={`status-message ${messageType}`}>{message}</p>
                        )}
                         {editError && !editLoading && <p role="alert" className="status-message error" style={{marginTop: '10px'}}>{editError}</p>}
                    </div>
                </div>

                {results && (
                    <div className="analyze-results-area">
                        <h2 className="results-title">
                            Analysis Results
                            {currentPdfName && <span className="document-name">for: {currentPdfName}</span>}
                            {results.id !== null && results.id !== undefined && <span className="analysis-id">(ID: {results.id})</span>}
                        </h2>

                        {results.title && results.title !== "No title found" && (
                            <div className="document-title-section"><h3>Title</h3><p>{results.title}</p></div>
                        )}
                        <div className="sections-area">
                            {results.sections?.length > 0 && (
                                <div className="translation-controls">
                                    <label htmlFor="language-select">Translate Summaries To:</label>
                                    <select id="language-select" value={targetLanguage} onChange={handleLanguageChange} disabled={translationLoading || !results.sections?.length || isDownloadingPdf || editLoading || !!editingSection}>
                                        {SUPPORTED_LANGUAGES.map(l => (<option key={l.code} value={l.code}>{l.name}</option>))}
                                    </select>
                                    {translationLoading && <div className="loading-spinner" aria-hidden="true"></div>}
                                    {translationError && <p role="alert" className="translation-error">{translationError}</p>}
                                </div>
                            )}
                            <h3>Sections</h3>
                            {/* Ensure results.sections IS an array before mapping */}
                            {Array.isArray(results.sections) && results.sections.length > 0 ? (
                                <div className="sections-list">
                                    {results.sections.map((section, index) => { // section is now an object {heading, summary, full_text}
                                        const heading = section.heading || `Section ${index + 1}`;
                                        const originalSummary = section.summary || "";
                                        const originalFullText = section.full_text || "";
                                        
                                        const translated = translatedSections[heading] || {}; // Use actual heading for translation lookup
                                        const translatedSummary = translated.translated_summary;

                                        const safeHeading = heading.replace(/[^a-zA-Z0-9-_]/g, '_').substring(0, 50);
                                        const originalSummaryId = `tts_orig_summary_${safeHeading}_${index}`;
                                        const originalFullTextId = `tts_orig_full_${safeHeading}_${index}`;
                                        const translatedSummaryId = `tts_trans_summary_${safeHeading}_${index}`;
                                        
                                        const isOriginalSummaryValid = !!originalSummary?.trim() && !originalSummary.toLowerCase().includes("not available");
                                        const isOriginalFullTextValid = !!originalFullText?.trim() && !originalFullText.toLowerCase().includes("not available");
                                        const isTranslatedSummaryValid = !!translatedSummary && typeof translatedSummary === 'string' && !translatedSummary.startsWith("[Translation Error:");

                                        const isOriginalSummaryPlaying = currentAudio?.id === originalSummaryId;
                                        const isOriginalFullTextPlaying = currentAudio?.id === originalFullTextId;
                                        const isTranslatedSummaryPlaying = currentAudio?.id === translatedSummaryId;

                                        const isEditingThisSummary = editingSection && editingSection.index === index && editingSection.type === 'summary';

                                        if (!heading && !isOriginalFullTextValid && !isOriginalSummaryValid) return null;

                                        return (
                                            <div key={`section-${index}-${safeHeading}`} className="result-section">
                                                <h4 className="section-heading">{heading}</h4>
                                                {isOriginalSummaryValid && (
                                                    <div className="summary-block">
                                                        <div className="text-block-header">
                                                            <p>Summary (Original):</p>
                                                            <div className="action-icons">
                                                                {!isEditingThisSummary && (
                                                                    <button onClick={() => isOriginalSummaryPlaying ? stopCurrentAudio() : handlePlayTTS(originalSummary, originalSummaryId, 'en')}
                                                                            disabled={ttsLoading[originalSummaryId] || isDownloadingPdf || editLoading || !!editingSection}
                                                                            title={isOriginalSummaryPlaying ? "Stop" : "Read Original Summary"}
                                                                            className={`icon-button tts-button ${isOriginalSummaryPlaying ? 'playing' : ''}`}>
                                                                        {ttsLoading[originalSummaryId] ? <div className="loading-spinner-small"></div> : (isOriginalSummaryPlaying ? <FaStopCircle/> : <FaVolumeUp/>)}
                                                                    </button>
                                                                )}
                                                                {!isEditingThisSummary && results?.id && (
                                                                    <button onClick={() => handleEditSummary(index, originalSummary)}
                                                                            disabled={editLoading || isDownloadingPdf || !!editingSection}
                                                                            title="Edit Summary"
                                                                            className="icon-button edit-button">
                                                                        <FaEdit />
                                                                    </button>
                                                                )}
                                                            </div>
                                                        </div>

                                                        {isEditingThisSummary ? (
                                                            <div className="edit-summary-container">
                                                                <textarea
                                                                    value={editText}
                                                                    onChange={handleEditTextChange}
                                                                    rows="6"
                                                                    className="edit-summary-textarea"
                                                                    disabled={editLoading}
                                                                    aria-label={`Edit summary for ${heading}`}
                                                                />
                                                                <div className="edit-summary-actions">
                                                                    <button onClick={handleSaveEditedSummary} disabled={editLoading || editText === originalSummary} className="button button-save">
                                                                        {editLoading ? <div className="loading-spinner-small"></div> : <FaSave />} Save
                                                                    </button>
                                                                    <button onClick={handleCancelEditSummary} disabled={editLoading} className="button button-cancel">
                                                                        <FaTimesCircle /> Cancel
                                                                    </button>
                                                                </div>
                                                            </div>
                                                        ) : (
                                                            <>
                                                                {ttsError[originalSummaryId] && <p role="alert" className="tts-error">{ttsError[originalSummaryId]}</p>}
                                                                <p className="text-content">{originalSummary}</p>
                                                            </>
                                                        )}

                                                        {!isEditingThisSummary && targetLanguage && translatedSummary && (
                                                            <div className="translated-block">
                                                                <div className="text-block-header">
                                                                    <p>Translated Summary ({SUPPORTED_LANGUAGES.find(l => l.code === targetLanguage)?.name || targetLanguage.toUpperCase()}):</p>
                                                                    {isTranslatedSummaryValid && (
                                                                        <div className="tts-controls">
                                                                            <button onClick={() => isTranslatedSummaryPlaying ? stopCurrentAudio() : handlePlayTTS(translatedSummary, translatedSummaryId, targetLanguage)} disabled={ttsLoading[translatedSummaryId] || isDownloadingPdf || editLoading || !!editingSection} title={isTranslatedSummaryPlaying ? "Stop" : "Read Translated Summary"} className={`icon-button tts-button ${isTranslatedSummaryPlaying ? 'playing' : ''}`}>
                                                                                {ttsLoading[translatedSummaryId] ? <div className="loading-spinner-small"></div> : (isTranslatedSummaryPlaying ? <FaStopCircle/> : <FaVolumeUp/>)}
                                                                            </button>
                                                                        </div>
                                                                    )}
                                                                </div>
                                                                {ttsError[translatedSummaryId] && <p role="alert" className="tts-error">{ttsError[translatedSummaryId]}</p>}
                                                                <p className={`text-content ${!isTranslatedSummaryValid ? 'error' : ''}`}>{translatedSummary}</p>
                                                            </div>
                                                        )}
                                                    </div>
                                                )}
                                                {isOriginalFullTextValid && (
                                                    <ExpandableText fullText={originalFullText} label={
                                                        <div style={{display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
                                                            <span>Original Text</span>
                                                            <div className="tts-controls">
                                                                <button onClick={(e) => { e.stopPropagation(); isOriginalFullTextPlaying ? stopCurrentAudio() : handlePlayTTS(originalFullText, originalFullTextId, 'en'); }} disabled={ttsLoading[originalFullTextId] || isDownloadingPdf || editLoading || !!editingSection} title={isOriginalFullTextPlaying ? "Stop" : "Read Original Text"} className={`icon-button tts-button ${isOriginalFullTextPlaying ? 'playing' : ''}`}>
                                                                    {ttsLoading[originalFullTextId] ? <div className="loading-spinner-small"></div> : (isOriginalFullTextPlaying ? <FaStopCircle/> : <FaVolumeUp/>)}
                                                                </button>
                                                            </div>
                                                        </div>
                                                    }>
                                                        <>
                                                            {ttsError[originalFullTextId] && <p role="alert" className="tts-error">{ttsError[originalFullTextId]}</p>}
                                                            <p className="text-content">{originalFullText}</p>
                                                        </>
                                                    </ExpandableText>
                                                )}
                                                {!isOriginalSummaryValid && !isOriginalFullTextValid && <p className="no-content-message">No text content extracted for this section.</p>}
                                            </div>
                                        );
                                    })}
                                </div>
                            ) : <p className="no-content-message">No text sections found or extracted.</p>}
                        </div>

                        {/* Figures, Tables, Chat/Course Buttons JSX - no changes from previous full code */}
                        <div className="figures-area">
                            <h3>Figures</h3>
                            {results.figures?.length > 0 ? results.figures.map(([desc, img], i) => {
                                const imgUrl = createDataUrl(img); const alt = desc || `Figure ${i + 1}`; const errId = `fig-err-${i}`;
                                return (<div key={`figure-${i}`} className="media-item">
                                    <p className="media-item-header">Figure:</p><p className="media-item-description">{alt}</p>
                                    {imgUrl && <img src={imgUrl} alt={alt} className="media-item-image" onError={(e) => { const el = document.getElementById(errId); if (el) el.style.display = 'block'; e.target.style.display = 'none'; }}/>}
                                    <p id={errId} className="media-item-error" style={{ display: imgUrl ? 'none' : 'block' }}>[Image data invalid or failed to load]</p>
                                </div>);
                            }) : <p className="no-content-message">No figures found.</p>}
                        </div>
                        <div className="tables-area">
                            <h3>Tables</h3>
                            {results.tables?.length > 0 ? results.tables.map(([desc, img], i) => {
                                const imgUrl = createDataUrl(img); const alt = desc || `Table ${i + 1}`; const errId = `tbl-err-${i}`;
                                return (<div key={`table-${i}`} className="media-item">
                                    <p className="media-item-header">Table:</p><p className="media-item-description">{alt}</p>
                                    {imgUrl && <img src={imgUrl} alt={alt} className="media-item-image" onError={(e) => { const el = document.getElementById(errId); if (el) el.style.display = 'block'; e.target.style.display = 'none'; }}/>}
                                    <p id={errId} className="media-item-error" style={{ display: imgUrl ? 'none' : 'block' }}>[Image data invalid or failed to load]</p>
                                </div>);
                            }) : <p className="no-content-message">No tables found.</p>}
                        </div>
                        <div className="chat-button-area">
                            <button className="button button-chat" onClick={handleChat} disabled={!results?.id || isDownloadingPdf || editLoading || !!editingSection} title={(!results?.id) ? "Analyze and save a document first" : "Ask AI"}>
                                Ask AI about this Document
                            </button>
                        </div>
                        <div className="chat-button-area">
                            <button className="button button-course" onClick={handleCourse} disabled={!results?.id || isDownloadingPdf || editLoading || !!editingSection} title={(!results?.id) ? "Analyze and save a document first" : "Generate course/quiz"}>
                                Generate course and Quiz
                            </button>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default AnalyzePage;