// MultiArticleCourseGenerator.js
import React, { useState, useRef } from 'react';
import jsPDF from 'jspdf'; // <<<<<<<<<<<< IMPORT jspdf
import './generate.css'; // Assuming your CSS file is named generate.css

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:5000';

// Helper function to parse a theme block string into displayable parts (remains the same)
const parseThemeBlockForDisplay = (blockString, index) => {
  const lines = blockString.split('\n').map(line => line.trim());
  let displayTitle = `Theme ${index + 1}`;
  let displayDescription = "No description available.";

  if (lines.length > 0) {
    const firstLine = lines[0];
    const titleMatch = firstLine.match(/^(?:\d+\.\s*)?\*\*Theme Title:\*\*\s*(.*)/i);
    if (titleMatch && titleMatch[1]) {
      displayTitle = titleMatch[1];
    } else {
      displayTitle = firstLine.replace(/^\d+\.\s*/, '').trim();
    }
  }

  if (lines.length > 1) {
    const secondLine = lines[1];
    const descriptionMatch = secondLine.match(/^\*\*Description:\*\*\s*(.*)/i);
    if (descriptionMatch && descriptionMatch[1]) {
      displayDescription = descriptionMatch[1];
    } else {
      displayDescription = secondLine;
    }
  }
  return { displayTitle, displayDescription, originalBlock: blockString };
};

// --- Helper for inline formatting like bold ---
const processInlineFormatting = (text) => {
  const parts = text.split(/(\*\*.*?\*\*)/g);
  return parts.map((part, index) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      return <strong key={index}>{part.substring(2, part.length - 2)}</strong>;
    }
    return part;
  });
};

// --- Component to render parsed markdown-like text ---
const RenderMarkdownContent = ({ content }) => {
  if (!content) return null;

  const lines = content.split('\n');
  const elements = [];
  let currentListType = null; // 'ul' or 'ol'
  let currentListItems = [];

  const flushList = () => {
    if (currentListItems.length > 0) {
      if (currentListType === 'ul') {
        elements.push(<ul key={`list-${elements.length}`} className="md-ul">{currentListItems}</ul>);
      } else if (currentListType === 'ol') {
        elements.push(<ol key={`list-${elements.length}`} className="md-ol">{currentListItems}</ol>);
      }
      currentListItems = [];
      currentListType = null;
    }
  };

  for (let i = 0; i < lines.length; i++) {
    let line = lines[i];

    if (line.startsWith('## ')) {
      flushList();
      elements.push(<h2 key={i} className="md-h2">{processInlineFormatting(line.substring(3))}</h2>);
      continue;
    }
    if (line.startsWith('### ')) {
      flushList();
      elements.push(<h3 key={i} className="md-h3">{processInlineFormatting(line.substring(4))}</h3>);
      continue;
    }
    if (line.startsWith('#### ')) {
      flushList();
      elements.push(<h4 key={i} className="md-h4">{processInlineFormatting(line.substring(5))}</h4>);
      continue;
    }

    const unorderedMatch = line.match(/^(\s*)-\s+(.*)/);
    const orderedMatch = line.match(/^(\s*)(\d+)\.\s+(.*)/);

    if (unorderedMatch) {
      if (currentListType !== 'ul') {
        flushList();
        currentListType = 'ul';
      }
      currentListItems.push(<li key={i}>{processInlineFormatting(unorderedMatch[2])}</li>);
    } else if (orderedMatch) {
      if (currentListType !== 'ol') {
        flushList();
        currentListType = 'ol';
      }
      currentListItems.push(<li key={i}>{processInlineFormatting(orderedMatch[3])}</li>);
    } else {
      flushList();
      if (line.trim() === '') {
        elements.push(<br key={i} />);
      } else {
        elements.push(<p key={i} className="md-p">{processInlineFormatting(line)}</p>);
      }
    }
  }
  flushList(); // Ensure any trailing list is flushed

  return <div className="markdown-render-area">{elements}</div>;
};


const MultiArticleCourseGenerator = () => {
  const [selectedFiles, setSelectedFiles] = useState(null);
  const [extractedArticlesContent, setExtractedArticlesContent] = useState([]);
  const [themeOptions, setThemeOptions] = useState([]);
  const [selectedThemeForCourse, setSelectedThemeForCourse] = useState('');
  const [generatedCourseOutline, setGeneratedCourseOutline] = useState('');
  const [isLoadingThemes, setIsLoadingThemes] = useState(false);
  const [isLoadingCourse, setIsLoadingCourse] = useState(false);
  const [isDownloadingPdf, setIsDownloadingPdf] = useState(false);
  const [error, setError] = useState(null);
  const [statusMessage, setStatusMessage] = useState('');
  const fileInputRef = useRef(null);

  const handleFileChange = (event) => {
    setSelectedFiles(event.target.files);
    setExtractedArticlesContent([]);
    setThemeOptions([]);
    setSelectedThemeForCourse('');
    setGeneratedCourseOutline('');
    setError(null);
    setStatusMessage(event.target.files?.length > 0 ? 'Files selected. Ready to identify themes.' : '');
  };

  const handleIdentifyThemes = async () => {
    if (!selectedFiles || selectedFiles.length === 0) {
      setError('Please select one or more PDF files.');
      setStatusMessage('Operation failed.');
      return;
    }
    setIsLoadingThemes(true);
    setError(null);
    setStatusMessage(`Processing ${selectedFiles.length} file(s) to identify themes...`);
    setThemeOptions([]);
    setGeneratedCourseOutline('');
    try {
      const formData = new FormData();
      for (let i = 0; i < selectedFiles.length; i++) formData.append('pdf_files', selectedFiles[i]);
      const response = await fetch(`${BACKEND_URL}/api/multi-article/identify-themes`, { method: 'POST', body: formData });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || data.details || `HTTP error ${response.status}`);
      if (!data.hasOwnProperty('identified_themes')) {
        setError("Themes identified, but the 'identified_themes' key was missing. Check backend.");
        setExtractedArticlesContent(data.extracted_articles_content || []);
        setStatusMessage('Themes processed, but no theme data received.');
        return;
      }
      const themesString = data.identified_themes || "";
      const rawThemeBlocks = themesString.split(/\n\s*\n+/).map(b => b.trim()).filter(b => b.length > 0);
      const processedThemeOptions = rawThemeBlocks.map((block, idx) => parseThemeBlockForDisplay(block, idx));
      setThemeOptions(processedThemeOptions);
      setExtractedArticlesContent(data.extracted_articles_content || []);
      setStatusMessage(processedThemeOptions.length > 0 ? 'Themes identified. Please select a theme.' : 'No themes identified.');
      setError(processedThemeOptions.length === 0 && !data.error && data.hasOwnProperty('identified_themes') ? "No distinct themes." : null);
    } catch (err) {
      setError(`Error identifying themes: ${err.message}`);
      setStatusMessage('Failed to identify themes.');
    } finally {
      setIsLoadingThemes(false);
    }
  };

  const handleGenerateCourse = async (themeObject) => {
    const originalThemeBlockText = themeObject.originalBlock;
    if (!originalThemeBlockText) { setError('No theme selected.'); setStatusMessage('Operation failed.'); return; }
    if (!extractedArticlesContent || extractedArticlesContent.length === 0) { setError('No article content.'); setStatusMessage('Operation failed.'); return; }
    setIsLoadingCourse(true);
    setSelectedThemeForCourse(originalThemeBlockText);
    setError(null);
    setStatusMessage(`Generating course for theme: "${themeObject.displayTitle}"...`);
    setGeneratedCourseOutline('');
    try {
      const payload = { article_contents: extractedArticlesContent, selected_themes_text: originalThemeBlockText };
      const response = await fetch(`${BACKEND_URL}/api/multi-article/generate-course`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || data.details || `HTTP error ${response.status}`);
      setGeneratedCourseOutline(data.generated_course_outline);
      setStatusMessage('Course generated successfully!');
    } catch (err) {
      setError(`Error generating course: ${err.message}`);
      setStatusMessage('Failed to generate course.');
    } finally {
      setIsLoadingCourse(false);
    }
  };

  // --- PDF Generation Helper ---
  const _addTextToPdfDoc = (doc, text, fontSize, yPos, options = {}) => {
    const { isBold = false, indent = 0, customYStep, margin = 15, maxWidth, pageHeight, color = '#000000' } = options;
    let currentY = yPos;
    const textStr = String(text || '').trim();

    if (textStr === "" && customYStep === undefined) return currentY;
    if (textStr === "" && customYStep !== undefined) { currentY += customYStep; return currentY; }
    
    doc.setFontSize(fontSize);
    doc.setFont('helvetica', isBold ? 'bold' : 'normal');
    doc.setTextColor(color);

    const lines = doc.splitTextToSize(textStr, maxWidth - indent);
    const lineHeight = (doc.getTextDimensions('M').h / doc.internal.scaleFactor) * 0.9; // More reliable line height

    lines.forEach(line => {
        if (currentY + lineHeight > pageHeight - margin) {
            doc.addPage(); currentY = margin;
        }
        doc.text(line, margin + indent, currentY);
        currentY += lineHeight;
    });
    currentY += customYStep !== undefined ? customYStep : (lineHeight * 0.3); // Smaller gap after text
    return currentY;
  };


  const handleDownloadCoursePdf = async () => {
    if (!generatedCourseOutline) {
      setError("No course outline to download.");
      setStatusMessage("PDF generation failed.");
      return;
    }
    setIsDownloadingPdf(true);
    setStatusMessage("Generating PDF, please wait...");

    await new Promise(resolve => setTimeout(resolve, 100)); // Allow UI to update

    const doc = new jsPDF({ orientation: 'p', unit: 'mm', format: 'a4' });
    let yPos = 15;
    const pageHeight = doc.internal.pageSize.getHeight();
    const pageWidth = doc.internal.pageSize.getWidth();
    const margin = 15;
    const maxWidth = pageWidth - (2 * margin);
    
    const courseTitle = selectedThemeForCourse 
        ? parseThemeBlockForDisplay(selectedThemeForCourse, 0).displayTitle 
        : "Generated Course";
    
    yPos = _addTextToPdfDoc(doc, `Course Outline: ${courseTitle}`, 18, yPos, { margin, maxWidth, pageHeight, isBold: true, customYStep: 8 });

    const lines = generatedCourseOutline.split('\n');

    for (const line of lines) {
        let fontSize = 10;
        let isBold = false;
        let indent = 0;
        let customYStep = 2; // Default small step after a line

        if (line.startsWith('## ')) {
            fontSize = 16; isBold = true; customYStep = 6;
            yPos = _addTextToPdfDoc(doc, line.substring(3), fontSize, yPos, { margin, maxWidth, pageHeight, isBold, customYStep });
        } else if (line.startsWith('### ')) {
            fontSize = 14; isBold = true; customYStep = 5;
            yPos = _addTextToPdfDoc(doc, line.substring(4), fontSize, yPos, { margin, maxWidth, pageHeight, isBold, customYStep });
        } else if (line.startsWith('#### ')) {
            fontSize = 12; isBold = true; customYStep = 4;
            yPos = _addTextToPdfDoc(doc, line.substring(5), fontSize, yPos, { margin, maxWidth, pageHeight, isBold, customYStep });
        } else if (line.match(/^(\s*)-\s+(.*)/)) {
            const match = line.match(/^(\s*)-\s+(.*)/);
            indent = match[1].length * 2; // Basic indent for lists
            const itemText = `• ${match[2]}`; // Add bullet point
            isBold = match[2].startsWith('**') && match[2].endsWith('**');
            const cleanText = isBold ? itemText.replace(/\*\*/g, '') : itemText;
            yPos = _addTextToPdfDoc(doc, cleanText, fontSize, yPos, { margin, maxWidth, pageHeight, isBold, indent, customYStep:1 });
        } else if (line.match(/^(\s*)(\d+)\.\s+(.*)/)) {
            const match = line.match(/^(\s*)(\d+)\.\s+(.*)/);
            indent = match[1].length * 2;
            const itemText = `${match[2]}. ${match[3]}`;
            isBold = match[3].startsWith('**') && match[3].endsWith('**');
            const cleanText = isBold ? itemText.replace(/\*\*/g, '') : itemText;
            yPos = _addTextToPdfDoc(doc, cleanText, fontSize, yPos, { margin, maxWidth, pageHeight, isBold, indent, customYStep:1 });
        } else if (line.trim() === '') {
            yPos += 3; // Add a bit of space for blank lines
             if (yPos > pageHeight - margin) { doc.addPage(); yPos = margin; }
        } else {
            // Handle bold within paragraphs for PDF
            const parts = line.split(/(\*\*.*?\*\*)/g);
            let currentX = margin + indent;
            let lineProcessed = false;

            for (const part of parts) {
                if (part.startsWith('**') && part.endsWith('**')) {
                    const boldText = part.substring(2, part.length - 2);
                    yPos = _addTextToPdfDoc(doc, boldText, fontSize, yPos, { margin, maxWidth, pageHeight, isBold: true, indent: currentX - margin, customYStep: 0});
                    // Estimate width of bold text to adjust currentX (very approximate)
                    // currentX += doc.getTextWidth(boldText); // This would require currentX to be passed and returned by _addTextToPdfDoc
                    // For simplicity, we assume it fits or _addTextToPdfDoc handles wrapping the bold part itself on a new line if needed.
                    // This part is tricky for precise inline bold placement without a more complex text renderer.
                    // A simpler approach for PDF: if a line contains **, make the whole line bold or parse more carefully.
                    // For now, we'll just render it. _addTextToPdfDoc will handle its placement.
                } else if (part.trim() !== ''){
                     yPos = _addTextToPdfDoc(doc, part, fontSize, yPos, { margin, maxWidth, pageHeight, isBold: false, indent: currentX - margin, customYStep: 0 });
                }
                lineProcessed = true;
            }
             if (lineProcessed) yPos += customYStep; // Add step after the full paragraph line is processed.
        }
         if (yPos > pageHeight - margin - 10) { // Check before potential next element
            doc.addPage(); yPos = margin;
        }
    }

    const safeDocTitle = courseTitle.replace(/[^a-z0-9_ .-]/gi, '').replace(/\s+/g, '_').substring(0, 50);
    doc.save(`${safeDocTitle}_Course_Outline.pdf`);
    
    setStatusMessage("Course Outline PDF generated and download initiated.");
    setIsDownloadingPdf(false);
  };


  return (
    <div className="multi-article-page-container">
      <div className="multi-article-main-content">
        <h1 className="page-main-title">Multi-Article Course Generator</h1>

        <div className="control-panel">
          <h2 className="panel-title">Step 1: Upload PDFs & Identify Themes</h2>
          <div className="file-input-container-macg">
            <input
              type="file" multiple accept=".pdf" onChange={handleFileChange} ref={fileInputRef}
              className="file-input-element" aria-label="Select PDF files"
              disabled={isLoadingThemes || isLoadingCourse || isDownloadingPdf}
            />
          </div>
          <div className="action-buttons-macg">
            <button
              onClick={handleIdentifyThemes}
              disabled={!selectedFiles || selectedFiles.length === 0 || isLoadingThemes || isLoadingCourse || isDownloadingPdf}
              className="button button-primary"
            >
              {isLoadingThemes ? (<><div className="loading-spinner-small"></div> Identifying...</>) : 'Identify Themes'}
            </button>
          </div>
        </div>

        <div className="status-area">
          {(isLoadingThemes || isLoadingCourse || isDownloadingPdf) && 
           !statusMessage.includes('Generating course for theme') && 
           !statusMessage.includes('Processing') &&
           !statusMessage.includes('Generating PDF') && (
            <div className="loading-indicator"><div className="loading-spinner"></div><p>Processing...</p></div>
          )}
          {statusMessage && (
            <p className={`status-message ${error ? 'error' : statusMessage.includes('successfully') || statusMessage.includes('initiated') ? 'success' : 'info'}`}>
              <em>{statusMessage}</em>
            </p>
          )}
          {error && !statusMessage.includes(error) && (
             <p className="status-message error">Error: {error}</p>
          )}
        </div>

        {themeOptions.length > 0 && !isLoadingThemes && (
          <div className="results-panel">
            <h2 className="panel-title">Step 2: Select a Theme to Generate Course</h2>
            <p className="panel-subtitle">Click a theme button below.</p>
            <div className="theme-options-list">
              {themeOptions.map((themeObj, index) => (
                <div key={index} className={`theme-item ${selectedThemeForCourse === themeObj.originalBlock ? 'selected' : ''}`}>
                  <button
                    onClick={() => handleGenerateCourse(themeObj)}
                    disabled={isLoadingCourse || isDownloadingPdf}
                    className="button button-theme-select"
                  >
                    {isLoadingCourse && selectedThemeForCourse === themeObj.originalBlock ? (<><div className="loading-spinner-small"></div> Generating...</>) : `${index + 1}. ${themeObj.displayTitle}`}
                  </button>
                  <p className="theme-description">{themeObj.displayDescription}</p>
                </div>
              ))}
            </div>
            {isLoadingCourse && statusMessage.includes('Generating course for theme') && (
                 <div className="loading-indicator" style={{marginTop: '15px'}}><div className="loading-spinner"></div><p>{statusMessage}</p></div>
            )}
          </div>
        )}

        {generatedCourseOutline && !isLoadingCourse && (
          <div className="results-panel">
            <div style={{ justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px',justifyItems: 'center' }}>
              <h2 className="panel-title" style={{ marginBottom: 0, borderBottom: 'none' }}>Step 3: Generated Course Outline</h2>
              <button
                onClick={handleDownloadCoursePdf}
                className="button button-secondary" // You might want a new class like button-download
                disabled={isDownloadingPdf}
              >
                {isDownloadingPdf ? (<><div className="loading-spinner-small"></div> Downloading...</>) : 'Download PDF'}
              </button>
            </div>
            <h3 className="course-theme-title">
              Based on theme: "{selectedThemeForCourse ? parseThemeBlockForDisplay(selectedThemeForCourse, 0).displayTitle : 'Selected Theme'}"
            </h3>
            {/* Using the new RenderMarkdownContent component instead of <pre> */}
            <RenderMarkdownContent content={generatedCourseOutline} />
          </div>
        )}
      </div>
    </div>
  );
};

export default MultiArticleCourseGenerator;