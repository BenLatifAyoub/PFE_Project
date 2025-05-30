import React from 'react';
import './modal.css'; // Ensure you have this CSS file

function Modal({ isOpen, onClose, article, isLoading, error }) {

    if (!isOpen) {
        return null;
    }

    // Helper function to render images from Base64 strings (same as before)
    const renderBase64Image = (base64String, altText, key) => {
        if (typeof base64String !== 'string' || base64String.length < 100) {
            console.warn("Invalid Base64 data provided for image:", key);
            return <p key={key} className="error-message">Invalid image data</p>;
        }
        const imageUrl = `data:image/png;base64,${base64String}`;
        return (
            <img
                key={key}
                src={imageUrl}
                alt={altText}
                className="modal-image"
            />
        );
    };

    // Helper function to render complex data like JSON (can be removed if not needed elsewhere)
    // const renderComplexData = (data) => { ... };

    let content;
    if (isLoading) {
        content = <p className='loading-message'>Loading article details...</p>;
    } else if (error) {
        content = <p className="error-message">Error: {error}</p>;
    } else if (!article) {
        content = <p>Article data not available.</p>;
    } else {
        // Data is ready, render the details
        content = (
            <>
                <h2>{article.title || 'No Title Available'}</h2>

                <div className="modal-section">
                    <h3>Details</h3>
                    <p><strong>ID:</strong> {article.id}</p>
                    <p><strong>Pages:</strong> {article.Pages ?? 'N/A'}</p>
                </div>

                {article.Insights && (
                    <div className="modal-section">
                        <h3>Insights</h3>
                        <p>{article.Insights}</p>
                    </div>
                )}

                {/* Render Figures (same as before) */}
                {article.figures && Array.isArray(article.figures) && article.figures.length > 0 && (
                    <div className="modal-section">
                        <h3>Figures</h3>
                        {article.figures.map((figureData, index) => (
                            <div key={`fig-${index}`} className="figure-container">
                                {figureData[0] && <p className="figure-caption"><strong>Figure {index + 1}:</strong> {figureData[0]}</p>}
                                {figureData[1] && renderBase64Image(figureData[1], figureData[0] || `Figure ${index + 1}`, `fig-img-${index}`)}
                            </div>
                        ))}
                    </div>
                )}

                 {/* Render Tables (same as before) */}
                 {article.tables && Array.isArray(article.tables) && article.tables.length > 0 && (
                    <div className="modal-section">
                        <h3>Tables</h3>
                         {article.tables.map((tableData, index) => (
                            <div key={`tbl-${index}`} className="figure-container">
                                {tableData[0] && <p className="figure-caption"><strong>Table {index + 1}:</strong> {tableData[0]}</p>}
                                {tableData[1] && renderBase64Image(tableData[1], tableData[0] || `Table ${index + 1}`, `tbl-img-${index}`)}
                             </div>
                        ))}
                    </div>
                 )}

                {/* --- UPDATED Section Rendering --- */}
                {article.sections && typeof article.sections === 'object' && Object.keys(article.sections).length > 0 && (
                    <div className="modal-section">
                        <h3>Section Summaries</h3>
                        {Object.entries(article.sections).map(([title, data], index) => (
                            // data should be an object like { full_text: "...", summary: "..." }
                            <div key={`sec-${index}`} className="section-summary-item">
                                {/* Capitalize the first letter of the title */}
                                <h4>{title.charAt(0).toUpperCase() + title.slice(1)}</h4>
                                {/* Check if data is an object and has a summary property */}
                                {typeof data === 'object' && data !== null && data.summary ? (
                                    <p>{data.summary}</p>
                                ) : typeof data === 'object' && data !== null && data.full_text && !data.summary ? (
                                     // Fallback: Show first part of full_text if no summary
                                     <p><i>{data.full_text.substring(0, 150)}{data.full_text.length > 150 ? '...' : ''} (No specific summary)</i></p>
                                ) : (
                                     <p><i>No summary available.</i></p>
                                )}
                            </div>
                        ))}
                    </div>
                )}
                {/* Fallback if sections is missing or not an object */}
                {!article.sections || typeof article.sections !== 'object' ? (
                   <div className="modal-section">
                        <h3>Sections</h3>
                        <p><i>Sections data is not available or in an unexpected format.</i></p>
                   </div>
                ) : null}
                 {/* Fallback if sections object is empty */}
                {typeof article.sections === 'object' && Object.keys(article.sections).length === 0 ? (
                    <div className="modal-section">
                        <h3>Sections</h3>
                        <p><i>No sections found in the data.</i></p>
                    </div>
                ) : null}
                {/* --- END UPDATED Section Rendering --- */}


                <div className="modal-section timestamps">
                    <p><small>Created: {article.created_at ? new Date(article.created_at).toLocaleString() : 'N/A'}</small></p>
                    <p><small>Updated: {article.updated_at ? new Date(article.updated_at).toLocaleString() : 'N/A'}</small></p>
                </div>
            </>
        );
    }

    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal-content" onClick={(e) => e.stopPropagation()}>
                <button className="modal-close-button" onClick={onClose}>×</button>
                {content}
            </div>
        </div>
    );
}

export default Modal;