// src/components/ExpandableText.jsx (or your path)
import React, { useState } from 'react';
// Import the desired arrow icons
import { FaChevronRight, FaChevronDown } from 'react-icons/fa';

/**
 * A component that displays a button with an arrow icon to toggle
 * the visibility of longer text content.
 * It displays a label next to the arrow and can show children elements
 * (like translations) within the expanded view.
 *
 * @param {string} fullText - The main text content to be expanded/collapsed.
 * @param {React.ReactNode} label - The text or JSX element to display next to the arrow icon.
 * @param {boolean} [initiallyExpanded=false] - Whether the text should be expanded by default.
 * @param {React.ReactNode} children - Any additional content to display inside the expanded section below the fullText.
 */
const ExpandableText = ({ fullText, label = "Text", initiallyExpanded = false, children }) => {
    const [isExpanded, setIsExpanded] = useState(initiallyExpanded);

    const toggleExpand = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setIsExpanded(!isExpanded);
    };

    const hasFullText = fullText && typeof fullText === 'string' && fullText.trim() !== '';

    if (!hasFullText) {
        return null;
    }

    // Create a descriptive aria-label for accessibility
    const accessibleLabel = `${isExpanded ? 'Hide' : 'Show'} ${typeof label === 'string' ? label : 'content'}`;

    return (
        <div style={{ marginTop: '8px' }}>
            {/* Button now contains the icon and the label */}
            <button
                onClick={toggleExpand}
                style={{
                    background: 'none',
                    border: 'none',
                    color: '#81d4fa', // Use icon/text color from parent or theme
                    cursor: 'pointer',
                    padding: '5px 0', // Add some vertical padding for easier clicking
                    margin: 0,
                    textAlign: 'left',
                    marginBottom: '5px',
                    fontSize: '1em', // Make label text normal size
                    display: 'flex', // Align icon and label horizontally
                    alignItems: 'center', // Vertically center icon and label
                    width: '100%', // Take full width
                    // Remove text decoration if previously applied
                    // textDecoration: 'none',
                }}
                aria-expanded={isExpanded}
                aria-controls={`expandable-content-${label}`}
                aria-label={accessibleLabel} // Essential for screen readers
            >
                {/* Conditional Icon */}
                {isExpanded ? (
                    <FaChevronDown aria-hidden="true" style={{ marginRight: '8px', fontSize: '0.9em' }} />
                ) : (
                    <FaChevronRight aria-hidden="true" style={{ marginRight: '8px', fontSize: '0.9em' }} />
                )}

                {/* The Label passed from the parent */}
                <span style={{ flexGrow: 1 }}>{label}</span>

                {/* Optional: If the label itself contains buttons (like your TTS button),
                    ensure they still work. The stopPropagation in toggleExpand helps. */}

            </button>

            {/* Conditionally render the expanded content */}
            {isExpanded && (
                <div
                    id={`expandable-content-${label}`}
                    style={{
                        marginTop: '5px',
                        // Indent the content slightly more than the button text for clarity
                        paddingLeft: '25px', // Increased indentation
                        // borderLeft: '2px solid #555', // Border might be less necessary with indentation
                        backgroundColor: 'rgba(0,0,0,0.1)',
                        borderRadius: '4px',
                        paddingTop: '10px',
                        paddingBottom: '10px',
                        paddingRight: '10px', // Add some right padding too
                    }}
                >
                    {/* The actual full text content */}
                    <p style={{ whiteSpace: 'pre-wrap', marginTop: '0', marginBottom: children ? '15px' : '0', color: '#e0e0e0' }}>
                        {fullText}
                    </p>

                    {/* Render any children elements passed to the component */}
                    {children}
                </div>
            )}
        </div>
    );
};

export default ExpandableText;