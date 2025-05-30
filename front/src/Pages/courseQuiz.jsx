import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';
import Confetti from 'react-confetti';
import './coursePage.css'; // Ensure you have your CSS file

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:5000';

// Simple hook to get window dimensions for Confetti
const useWindowDimensions = () => {
    const [windowDimensions, setWindowDimensions] = useState({
        width: window.innerWidth,
        height: window.innerHeight,
    });

    useEffect(() => {
        function handleResize() {
            setWindowDimensions({
                width: window.innerWidth,
                height: window.innerHeight,
            });
        }
        window.addEventListener('resize', handleResize);
        return () => window.removeEventListener('resize', handleResize);
    }, []);

    return windowDimensions;
};

// Helper function to extract a meaningful title from markdown
function extractMeaningfulTitle(markdownText, fallbackIdentifier) {
    if (!markdownText || typeof markdownText !== 'string') {
        if (fallbackIdentifier && typeof fallbackIdentifier === 'string' && fallbackIdentifier.length > 5 && !fallbackIdentifier.match(/\.[a-zA-Z0-9]{3,4}$/)) {
            return fallbackIdentifier;
        }
        return "Course Document";
    }
    const explicitTitleMatch = markdownText.match(/^title:\s*(.*)/im);
    if (explicitTitleMatch && explicitTitleMatch[1] && explicitTitleMatch[1].trim().length > 0) {
        return explicitTitleMatch[1].trim();
    }
    const h1Match = markdownText.match(/^#\s+(.*)/m);
    if (h1Match && h1Match[1] && h1Match[1].trim().length > 0) {
        return h1Match[1].trim();
    }
    const h2Match = markdownText.match(/^##\s+(.*)/m);
    if (h2Match && h2Match[1] && h2Match[1].trim().length > 0) {
        return h2Match[1].trim();
    }
    if (fallbackIdentifier && typeof fallbackIdentifier === 'string' && fallbackIdentifier.length > 5 && !fallbackIdentifier.match(/\.[a-zA-Z0-9]{3,4}$/)) {
        return fallbackIdentifier;
    }
    if (fallbackIdentifier && typeof fallbackIdentifier === 'string') {
        return `Course from: ${fallbackIdentifier.split('.')[0]}`;
    }
    return "Course Document";
}


function parseQuizContent(quizText) {
    if (!quizText || typeof quizText !== 'string') {
        console.warn("parseQuizContent: Input quizText is empty or not a string.");
        return [];
    }

    const questions = [];
    // Regex to split questions: looks for "1. (MC)" etc.
    const questionBlocks = quizText.trim().split(/\s*\n+\s*(?=\d+\.\s*\((?:MC|TF|SA)\))/);
    let globalQuestionCounter = 0;

    for (const block of questionBlocks) {
        const trimmedBlock = block.trim();
        if (!trimmedBlock) continue;

        const lines = trimmedBlock.split('\n');
        if (lines.length === 0) continue;

        const headerLine = lines[0].trim();
        // Regex for question header: "1. (MC) Question text..."
        const headerMatch = headerLine.match(/^(\d+)\.\s*\((MC|TF|SA)\)\s*(.*)/);

        if (!headerMatch) {
            console.warn(`Could not parse header for block: "${lines[0]}"`);
            continue;
        }

        const questionNumberFromText = headerMatch[1];
        const type = headerMatch[2];
        let textFromHeaderContent = headerMatch[3].trim(); // Initial question text part from header

        let options = [];
        let correctAnswerFromText = ''; // Will be like "B)" or "True" or "Short answer"
        let sourceReferenceText = '';   // Will be the reference string
        const id = `q-${questionNumberFromText}-${type.toLowerCase()}-${globalQuestionCounter++}-${Date.now()}`;

        // Find the line starting with "Answer:"
        const answerLineIndex = lines.findIndex(line => /^\s*Answer:\s*/i.test(line.trim()));
        if (answerLineIndex === -1) {
            console.warn(`Could not find 'Answer:' line for question block (num ${questionNumberFromText}): "${lines[0]}"`);
            continue;
        }

        // --- Start: Robust Answer and Source Reference Parsing ---
        const rawAnswerLineContent = lines[answerLineIndex].trim(); // e.g., "Answer: B) Source Reference: ..." OR "Answer: B)"

        const srMarker = "source reference:"; // for case-insensitive search
        const answerMarker = "answer:";     // for case-insensitive search

        const srIndexOnAnswerLine = rawAnswerLineContent.toLowerCase().indexOf(srMarker);
        const answerMarkerIndexOnAnswerLine = rawAnswerLineContent.toLowerCase().indexOf(answerMarker);

        if (srIndexOnAnswerLine !== -1 && answerMarkerIndexOnAnswerLine !== -1 && srIndexOnAnswerLine > answerMarkerIndexOnAnswerLine) {
            // Case 1: "Source Reference:" is ON the answer line, AFTER "Answer:"
            const beforeSRPart = rawAnswerLineContent.substring(0, srIndexOnAnswerLine);
            correctAnswerFromText = beforeSRPart.replace(/^\s*Answer:\s*/i, '').trim();

            // Extract text after "Source Reference:"
            sourceReferenceText = rawAnswerLineContent.substring(srIndexOnAnswerLine + srMarker.length).replace(/^:\s*/, '').trim();
        } else {
            // Case 2: "Source Reference:" is NOT on the answer line (or not correctly formatted there)
            // So, the full answer line (minus "Answer:") is the candidate for correctAnswerFromText
            correctAnswerFromText = rawAnswerLineContent.replace(/^\s*Answer:\s*/i, '').trim();

            // Now, look for "Source Reference:" on SUBSEQUENT lines
            let foundSRNextLine = false;
            for (let i = answerLineIndex + 1; i < lines.length; i++) {
                const nextLineTrimmed = lines[i].trim();
                if (/^\s*Source Reference:\s*/i.test(nextLineTrimmed)) {
                    sourceReferenceText = nextLineTrimmed.replace(/^\s*Source Reference:\s*/i, '').trim();
                    foundSRNextLine = true;
                    break; // Found it, stop searching
                }
                // Heuristic: if we hit another question or answer, stop looking for SR for current question
                if (/^\d+\.\s*\((?:MC|TF|SA)\)/.test(nextLineTrimmed) || /^\s*Answer:\s*/i.test(nextLineTrimmed)) {
                    break;
                }
            }
            if (!foundSRNextLine) {
                sourceReferenceText = "Not available"; // Default if not found anywhere
            }
        }
        // --- End: Robust Answer and Source Reference Parsing ---

        // Lines between question header and "Answer:" line are part of question text or MC options
        const contentLinesAfterHeaderBeforeAnswer = lines.slice(1, answerLineIndex);
        let finalCorrectAnswer = correctAnswerFromText; // Will be refined for MCQs
        let questionTextContent = "";


        if (type === 'MC') {
            let linesToProcessForMcOptions = [];
            const startsWithOptionMarkerRegexForHeader = /^\s*[a-zA-Z]\)\s/;
            const containsOptionMarkerRegexForHeader = /\s+[a-zA-Z]\)\s/;

            if (containsOptionMarkerRegexForHeader.test(textFromHeaderContent)) {
                const matchInHeaderContent = textFromHeaderContent.match(containsOptionMarkerRegexForHeader);
                if (matchInHeaderContent && matchInHeaderContent.index > 0) {
                    questionTextContent = textFromHeaderContent.substring(0, matchInHeaderContent.index).trim();
                    linesToProcessForMcOptions.push(textFromHeaderContent.substring(matchInHeaderContent.index).trim());
                } else if (startsWithOptionMarkerRegexForHeader.test(textFromHeaderContent)) {
                     // Question text is empty, header starts with options
                    questionTextContent = "";
                    linesToProcessForMcOptions.push(textFromHeaderContent.trim());
                } else {
                    // No option marker in header, header is full question text
                    questionTextContent = textFromHeaderContent.trim();
                }
            } else {
                // Header is purely question text
                questionTextContent = textFromHeaderContent.trim();
            }

            // Add lines between header and answer line for option processing
            linesToProcessForMcOptions.push(...contentLinesAfterHeaderBeforeAnswer);

            const tempOptions = []; // To store { letter: 'A', text: 'Option A text' }
            let firstOptionMarkerFoundOnAnyLine = false;

            for (const line of linesToProcessForMcOptions) {
                const currentLineTrimmed = line.trim();
                if (!currentLineTrimmed) continue;

                const startsWithOptionMarkerRegex = /^[a-zA-Z]\)\s/; // e.g., "A) "
                const splitByOptionMarkerRegex = /\s+(?=[a-zA-Z]\)\s)/; // For splitting "A) OptA B) OptB"
                const singleOptionRegex = /^([a-zA-Z])\)\s+(.*)/; // To capture letter and text

                if (startsWithOptionMarkerRegex.test(currentLineTrimmed)) {
                    firstOptionMarkerFoundOnAnyLine = true;
                    const potentialOptionsOnThisLine = currentLineTrimmed.split(splitByOptionMarkerRegex);

                    for (const optStr of potentialOptionsOnThisLine) {
                        const trimmedOptStr = optStr.trim();
                        if (!trimmedOptStr) continue;
                        const optionMatch = trimmedOptStr.match(singleOptionRegex);
                        if (optionMatch) {
                            tempOptions.push({ letter: optionMatch[1].toUpperCase(), text: optionMatch[2].trim() });
                        } else if (tempOptions.length > 0 && trimmedOptStr) {
                             // Continuation of previous option on the same "split part" but not matching A) format
                             tempOptions[tempOptions.length-1].text += ' ' + trimmedOptStr;
                             tempOptions[tempOptions.length-1].text = tempOptions[tempOptions.length-1].text.trim();
                        }
                    }
                } else if (firstOptionMarkerFoundOnAnyLine) { // Line doesn't start with A) but options have started
                    if (tempOptions.length > 0) { // Append to the last option's text (multi-line option)
                        tempOptions[tempOptions.length - 1].text += ' ' + currentLineTrimmed;
                        tempOptions[tempOptions.length - 1].text = tempOptions[tempOptions.length - 1].text.trim();
                    } else {
                        // This case should ideally not happen if parsing is correct
                        console.warn(`MCQ Q${questionNumberFromText}: Orphaned line "${currentLineTrimmed}" after options supposedly started, but no tempOptions. Appending to question text.`);
                        questionTextContent += (questionTextContent ? ' ' : '') + currentLineTrimmed;
                    }
                } else { // Line is part of question text (before any A) option markers found)
                    questionTextContent += (questionTextContent ? ' ' : '') + currentLineTrimmed;
                }
            }
            questionTextContent = questionTextContent.trim();

            if (tempOptions.length > 0) {
                options = tempOptions.map(opt => opt.text); // Final list of option texts

                // Determine the actual correct answer text from `correctAnswerFromText` (e.g., "B)")
                let derivedLetter = "";
                // Match "B)" or "B." or "B" from correctAnswerFromText
                const letterMatch = String(correctAnswerFromText).trim().match(/^([a-zA-Z])(?=\W|$)/);
                if (letterMatch && letterMatch[1]) {
                    derivedLetter = letterMatch[1].toUpperCase(); // "B"
                }

                let matchedByLetter = false;
                if (derivedLetter) {
                    const correctOptionObject = tempOptions.find(opt => opt.letter === derivedLetter);
                    if (correctOptionObject) {
                        finalCorrectAnswer = correctOptionObject.text; // e.g., "CT"
                        matchedByLetter = true;
                    }
                }

                if (!matchedByLetter) { // Fallback: try direct text match if letter match failed or was ambiguous
                    const directTextMatch = options.find(optText =>
                        String(optText).trim().toLowerCase() === String(correctAnswerFromText).trim().toLowerCase()
                    );
                    if (directTextMatch) {
                        finalCorrectAnswer = directTextMatch;
                    } else {
                        console.warn(
                            `MCQ Q${questionNumberFromText}: Could not definitively match 'Answer: ${correctAnswerFromText}' ` +
                            `(derived letter: '${derivedLetter}') to any parsed option text. ` +
                            `Parsed options: ${JSON.stringify(tempOptions.map(o => `${o.letter}) ${o.text}`))} (Full list: ${JSON.stringify(options)}). ` +
                            `Using the raw text from 'Answer:' line ('${correctAnswerFromText}') as the correct answer placeholder.`
                        );
                        finalCorrectAnswer = correctAnswerFromText; // Use as placeholder
                    }
                }
            } else { // No options parsed for MC
                console.warn(`MCQ Q${questionNumberFromText}: No options were parsed for question: "${questionTextContent}". Using raw answer line content ('${correctAnswerFromText}') as correct answer.`);
                finalCorrectAnswer = correctAnswerFromText;
                options = []; // Ensure options array is empty
            }

        } else { // For SA (Short Answer) and TF (True/False)
            // Question text is header content + any lines before "Answer:"
            questionTextContent = textFromHeaderContent;
            for (const line of contentLinesAfterHeaderBeforeAnswer) {
                const currentLineTrimmed = line.trim();
                if (currentLineTrimmed) {
                    questionTextContent += (questionTextContent ? ' ' : '') + currentLineTrimmed;
                }
            }
            questionTextContent = questionTextContent.trim();

            if (type === 'TF') {
                options = ['True', 'False']; // Standard TF options
                // Normalize correctAnswerFromText to "True" or "False"
                const correctAnswerCleaned = String(correctAnswerFromText).trim().toLowerCase().replace(/[.)]$/, ''); // Remove trailing . or )
                if (correctAnswerCleaned === 'true') {
                    finalCorrectAnswer = 'True';
                } else if (correctAnswerCleaned === 'false') {
                    finalCorrectAnswer = 'False';
                } else {
                     console.warn(`TF Q${questionNumberFromText}: Answer '${correctAnswerFromText}' (cleaned: '${correctAnswerCleaned}') is not 'True' or 'False'. It will be treated as literal, which might cause comparison issues.`);
                     finalCorrectAnswer = correctAnswerFromText; // Keep original if not clearly true/false
                }
            }
            // For SA, finalCorrectAnswer is already correctAnswerFromText
        }

        if (questionTextContent) {
            questions.push({
                id,
                type,
                questionText: questionTextContent,
                options: (options && options.length > 0) ? options : undefined,
                correctAnswer: finalCorrectAnswer,
                sourceReference: sourceReferenceText, // Store the parsed source reference
                userAnswer: type === 'SA' ? '' : null,
                isCorrect: undefined,
                isRevealed: false,
            });
        } else {
            console.warn(`Could not parse question text for block starting with (num ${questionNumberFromText}):`, lines[0]);
        }
    }
    // console.log("Parsed Questions:", JSON.stringify(questions, null, 2)); // For debugging
    return questions;
}

function compareAnswers(type, userAnswer, correctAnswer) {
    if (userAnswer === null || userAnswer === undefined) return false;
    const uaTrimmed = String(userAnswer).trim();
    const caTrimmed = String(correctAnswer).trim();

    if (type === 'MC' || type === 'TF') {
        return uaTrimmed.toLowerCase() === caTrimmed.toLowerCase(); // Compare case-insensitively for robustness
    }
    if (type === 'SA') {
        // For SA, a more lenient comparison might be needed in a real app (e.g., regex, keyword matching)
        // For now, strict case-insensitive comparison:
        return uaTrimmed.toLowerCase() === caTrimmed.toLowerCase();
    }
    return false;
}

const CoursePage = () => {
    const navigate = useNavigate();
    const location = useLocation();
    const courseContentRef = useRef(null);
    const { width: windowWidth, height: windowHeight } = useWindowDimensions();

    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');
    const [message, setMessage] = useState('');
    const [courseContent, setCourseContent] = useState(null);
    const [quizData, setQuizData] = useState([]);
    const [articleId, setArticleId] = useState(null);
    
    const [sourceDocumentIdentifier, setSourceDocumentIdentifier] = useState('');
    const [courseDisplayTitle, setCourseDisplayTitle] = useState('Course Document');

    const [showCelebration, setShowCelebration] = useState(false);
    const [quizScore, setQuizScore] = useState({ score: 0, total: 0 });
    
    // --- NEW STATE FOR QUIZ VISIBILITY ---
    const [isQuizVisible, setIsQuizVisible] = useState(false);


    const fetchData = useCallback(async (id) => {
        setLoading(true);
        setError('');
        setMessage('');
        setCourseContent(null);
        setQuizData([]);
        setShowCelebration(false);
        // setIsQuizVisible(false); // Also reset quiz visibility on new data fetch if desired, or handle in handleRefresh

        try {
            const response = await axios.post(`${BACKEND_URL}/api/generate-course/${id}`);
            const {
                message: apiMessage,
                status,
                generated_course,
                generated_quiz
            } = response.data;

            setMessage(apiMessage || 'Request processed.');

            if (generated_course) {
                setCourseContent(generated_course);
                const extractedTitle = extractMeaningfulTitle(generated_course, sourceDocumentIdentifier);
                setCourseDisplayTitle(extractedTitle);
            } else {
                if (sourceDocumentIdentifier && !sourceDocumentIdentifier.match(/\.[a-zA-Z0-9]{3,4}$/)) {
                    setCourseDisplayTitle(sourceDocumentIdentifier);
                } else {
                     setCourseDisplayTitle(sourceDocumentIdentifier ? `Course from: ${sourceDocumentIdentifier.split('.')[0]}` : 'Course Document');
                }
            }

            if (generated_quiz) {
                const parsedQuiz = parseQuizContent(generated_quiz);
                console.log("Parsed Quiz Data:", parsedQuiz); // For debugging
                setQuizData(parsedQuiz.map(q => ({
                    ...q,
                    userAnswer: q.type === 'SA' ? '' : null,
                    isRevealed: false,
                    isCorrect: undefined // Ensure isCorrect is undefined initially
                })));
            } else {
                setQuizData([]);
            }

            if (status === "generated" && !generated_course && !generated_quiz) {
                 setMessage(apiMessage + " Content was generated. Refresh if not displayed.");
            }
        } catch (err) {
            const errorMsg = err.response?.data?.error || err.response?.data?.details || err.message || "Failed to generate or fetch course content.";
            setError(`Error: ${errorMsg}`);
        } finally {
            setLoading(false);
        }
    }, [sourceDocumentIdentifier]);

    useEffect(() => {
        if (location.state && location.state.contextId) {
            const currentArticleId = location.state.contextId;
            setArticleId(currentArticleId);
            setSourceDocumentIdentifier(location.state.pdfName || `Document ID ${currentArticleId}`);
            setIsQuizVisible(false); // Ensure quiz is hidden when component loads or contextId changes
            fetchData(currentArticleId);
        } else {
            setError("No document context ID provided.");
            setLoading(false);
        }
    }, [location.state, navigate, fetchData]);


    useEffect(() => {
        if (quizData.length > 0) {
            const allRevealed = quizData.every(q => q.isRevealed);
            if (allRevealed) {
                const currentScore = quizData.filter(q => q.isCorrect === true).length; // Explicitly check for true
                const total = quizData.length;
                setQuizScore({ score: currentScore, total: total });
                setShowCelebration(true);
            } else {
                setShowCelebration(false);
            }
        }
    }, [quizData]);

    const handleQuizAnswerChange = (questionId, answer) => {
        setQuizData(prevData =>
            prevData.map(q =>
                q.id === questionId ? { ...q, userAnswer: answer, isCorrect: undefined, isRevealed: q.isRevealed } : q
            )
        );
    };

    const handleCheckQuizAnswer = (questionId) => {
        setQuizData(prevData =>
            prevData.map(q => {
                if (q.id === questionId) {
                    const isCorrect = compareAnswers(q.type, q.userAnswer, q.correctAnswer);
                    return { ...q, isRevealed: true, isCorrect };
                }
                return q;
            })
        );
    };

    const handleRefresh = () => {
        if (articleId) {
            setIsQuizVisible(false); // --- RESET QUIZ VISIBILITY ON REFRESH ---
            fetchData(articleId);
        }
    };

    const renderMarkdown = (content) => {
        if (!content) return null;
        return <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>;
    };


    const handleDownloadPdf = async () => {
        if (!courseContentRef.current || !courseContent) {
            alert("Course content is not available to download.");
            return;
        }
        setLoading(true);
        setMessage("Generating PDF...");
        setError('');

        const inputElement = courseContentRef.current;

        try {
            const canvas = await html2canvas(inputElement, {
                scale: 2,
                useCORS: true,
                logging: false,
                onclone: (clonedDoc) => {
                    const contentRoot = clonedDoc.querySelector('.markdown-content');
                    if (contentRoot) {
                        clonedDoc.body.style.backgroundColor = 'white';
                        contentRoot.style.backgroundColor = 'white';
                        contentRoot.style.color = 'black';
                        contentRoot.style.padding = '20px';
                        contentRoot.style.boxShadow = 'none';

                        const elementsToStyle = contentRoot.querySelectorAll('h1, h2, h3, h4, h5, h6, p, li, span, div, strong, em, b, i, ul, ol, table, th, td, pre, code, a');
                        elementsToStyle.forEach(el => {
                            el.style.color = 'black'; 
                            el.style.textShadow = 'none';

                            if (!['IMG', 'HR', 'PRE', 'CODE'].includes(el.tagName)) {
                                el.style.backgroundColor = 'transparent';
                            } else if (['PRE', 'CODE'].includes(el.tagName)) {
                                el.style.backgroundColor = '#f0f0f0';
                                el.style.color = 'black';
                                el.style.border = '1px solid #ccc';
                                el.style.padding = '5px';
                            }
                            if (el.tagName === 'H1' || el.tagName === 'H2' || el.tagName === 'H3') {
                                 if (el.textContent.includes('Course Overview') || el.textContent.includes('Learning Outcomes') || el.textContent.includes('Modules') || el.textContent.includes('Module ')){
                                     el.style.color = '#FFA500'; 
                                 } else {
                                     el.style.color = 'black';
                                 }
                            }
                             if (el.tagName === 'A') {
                                el.style.color = '#0000EE';
                                el.style.textDecoration = 'underline';
                            }
                        });
                    }
                }
            });

            const pdf = new jsPDF({
                orientation: 'portrait',
                unit: 'px',
                format: 'a4',
                putOnlyUsedFonts: true,
                floatPrecision: 16
            });

            const pdfPageWidth = pdf.internal.pageSize.getWidth();
            const pdfPageHeight = pdf.internal.pageSize.getHeight();
            const titleForPdfDocument = courseDisplayTitle || "Course Document";
            const titleFontSize = 18;
            const titleTopMargin = 40;
            const spaceBelowTitle = 25;

            pdf.setFontSize(titleFontSize);
            pdf.setTextColor(0, 0, 0);

            const titleLines = pdf.splitTextToSize(titleForPdfDocument, pdfPageWidth - 40);
            let currentTitleY = titleTopMargin;
            titleLines.forEach(line => {
                const textWidth = pdf.getTextWidth(line);
                const titleX = (pdfPageWidth - textWidth) / 2;
                pdf.text(line, titleX > 20 ? titleX : 20, currentTitleY);
                currentTitleY += titleFontSize * 0.7; 
            });

            const contentStartYOnFirstPage = currentTitleY + spaceBelowTitle;
            const defaultPageTopMargin = 40;
            const defaultPageBottomMargin = 40;
            const canvasWidth = canvas.width;
            const canvasHeight = canvas.height;
            let yOnCanvas = 0;
            let pageNum = 0;

            while (yOnCanvas < canvasHeight) {
                pageNum++;
                if (pageNum > 1) {
                    pdf.addPage();
                }
                let currentContentStartYInPdf = (pageNum === 1) ? contentStartYOnFirstPage : defaultPageTopMargin;
                let availableHeightInPdf = pdfPageHeight - currentContentStartYInPdf - defaultPageBottomMargin;
                if (availableHeightInPdf <= 20) {
                    availableHeightInPdf = pdfPageHeight / 2;
                    currentContentStartYInPdf = (pageNum === 1) ? Math.max(contentStartYOnFirstPage, (pdfPageHeight - availableHeightInPdf) / 2) : defaultPageTopMargin;
                }
                const sliceHeightOnCanvas = (canvasWidth / pdfPageWidth) * availableHeightInPdf;
                const actualSliceHeightOnCanvas = Math.min(sliceHeightOnCanvas, canvasHeight - yOnCanvas);
                if (actualSliceHeightOnCanvas <= 0) break;
                const pageCanvas = document.createElement('canvas');
                pageCanvas.width = canvasWidth;
                pageCanvas.height = actualSliceHeightOnCanvas;
                const pageCtx = pageCanvas.getContext('2d');
                pageCtx.drawImage(canvas, 0, yOnCanvas, canvasWidth, actualSliceHeightOnCanvas, 0, 0, canvasWidth, actualSliceHeightOnCanvas);
                const pageImgData = pageCanvas.toDataURL('image/png');
                const imgHeightInPdf = (pdfPageWidth / canvasWidth) * actualSliceHeightOnCanvas;
                pdf.addImage(pageImgData, 'PNG', 0, currentContentStartYInPdf, pdfPageWidth, imgHeightInPdf);
                yOnCanvas += actualSliceHeightOnCanvas;
            }
            const baseFileName = courseDisplayTitle !== "Course Document" ? courseDisplayTitle : sourceDocumentIdentifier;
            const safePdfFileName = (baseFileName ? String(baseFileName).replace(/[^a-z0-9_-\s]/gi, '').replace(/\s+/g, '_') : 'course_content').toLowerCase();
            pdf.save(`${safePdfFileName}.pdf`);
            setMessage("PDF downloaded successfully.");

        } catch (pdfError) {
            console.error("Error generating PDF:", pdfError);
            setError("Failed to generate PDF. " + (pdfError.message || "Unknown error"));
            setMessage('');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="course-page-container">
            {showCelebration && (
                <>
                    <Confetti
                        width={windowWidth}
                        height={windowHeight}
                        recycle={false}
                        numberOfPieces={350}
                        gravity={0.12}
                        onConfettiComplete={(confetti) => {
                            setShowCelebration(false);
                            if (confetti) confetti.reset();
                        }}
                        initialVelocityY={25}
                    />
                    <div className="quiz-results-popup">
                        <h2>Quiz Completed!</h2>
                        <p>Your Score: {quizScore.score} / {quizScore.total}</p>
                        <p>
                            {quizScore.total > 0 && (
                                (quizScore.score / quizScore.total >= 0.8 ? "🎉 Excellent Work! 🎉" :
                                (quizScore.score / quizScore.total >= 0.5 ? "👍 Good Job! 👍" : "📖 Keep Practicing! 📖"))
                            )}
                        </p>
                        <button onClick={() => setShowCelebration(false)} className="button button-secondary">Close</button>
                    </div>
                </>
            )}

            <div className="course-main-content">
                <div className="course-control-panel">
                    <h1 className="course-page-title">
                        Course & Quiz {courseDisplayTitle && courseDisplayTitle !== "Course Document" ? `for: ${courseDisplayTitle.substring(0, 60)}${courseDisplayTitle.length > 60 ? '...' : ''}` : (sourceDocumentIdentifier ? `for source: ${sourceDocumentIdentifier}`: '')}
                    </h1>
                    {articleId && <p className="article-id-display">Article ID: {articleId}</p>}
                    <div className="course-action-buttons">
                        <button onClick={() => navigate(-1)} className="button button-secondary">
                            Back to Analyze
                        </button>
                        <button onClick={() => navigate('/home')} className="button button-home">
                            Home
                        </button>
                        <button onClick={handleRefresh} className="button button-refresh" disabled={loading || !articleId}>
                            {loading && message.includes("Generating") ? 'Working...' : loading ? 'Loading...' : 'Refresh Content'}
                        </button>
                        {courseContent && (
                             <button onClick={handleDownloadPdf} className="button button-download" disabled={loading}>
                                {loading && message.includes("Generating PDF") ? 'Generating PDF...' : 'Download Course PDF'}
                            </button>
                        )}
                    </div>
                </div>

                <div className="status-area">
                     {loading && <p className="loading-message">{message || "Loading content..."}</p>}
                    {!loading && message && !message.includes("Generating") && <p className="message-display">{message}</p>}
                    {!loading && error &&   <p className="error-message">{error}</p>}
                </div>

                {courseContent && (
                    <div className="content-section course-content-area">
                        <h2 className="content-title">Course Outline</h2>
                        <div className="markdown-content" ref={courseContentRef}>
                            {renderMarkdown(courseContent)}
                        </div>
                    </div>
                )}

                {/* --- BUTTON TO TOGGLE QUIZ VISIBILITY --- */}
                {courseContent && (
                    <div className="quiz-toggle-button-container" style={{ textAlign: 'center', margin: '20px 0' }}>
                        <button
                            onClick={() => setIsQuizVisible(prev => !prev)}
                            className="button button-primary button-toggle-quiz"
                        >
                            {isQuizVisible ? 'Hide Quiz Section' : 'Show Quiz Section'}
                        </button>
                    </div>
                )}
                {/* --- END BUTTON --- */}

                {/* --- CONDITIONALLY RENDERED QUIZ SECTION --- */}
                {isQuizVisible && (
                    <div className="quiz-section-wrapper"> {/* Optional wrapper if needed for styling expanded section */}
                        {quizData.length > 0 && !showCelebration && (
                            <div className="content-section quiz-interactive-area">
                                <h2 className="content-title">Interactive Quiz</h2>
                                {quizData.map((question) => (
                                    <div
                                        key={question.id}
                                        className={`quiz-question ${question.isRevealed ? (question.isCorrect ? 'correct' : 'incorrect') : ''}`}
                                    >
                                        <p className="question-text">
                                            <strong>({question.type})</strong> <ReactMarkdown remarkPlugins={[remarkGfm]} components={{ p: React.Fragment }}>{question.questionText}</ReactMarkdown>
                                        </p>

                                        {!question.isRevealed && (
                                            <div className="answer-input-area">
                                                {question.type === 'MC' && question.options && question.options.map((option, optIndex) => (
                                                    <div key={optIndex} className="quiz-option">
                                                        <input
                                                            type="radio"
                                                            id={`${question.id}-opt-${optIndex}`}
                                                            name={question.id}
                                                            value={option}
                                                            checked={question.userAnswer === option}
                                                            onChange={(e) => handleQuizAnswerChange(question.id, e.target.value)}
                                                        />
                                                        <label htmlFor={`${question.id}-opt-${optIndex}`}>{option}</label>
                                                    </div>
                                                ))}
                                                {question.type === 'TF' && question.options && question.options.map((option, optIndex) => (
                                                    <div key={optIndex} className="quiz-option">
                                                        <input
                                                            type="radio"
                                                            id={`${question.id}-opt-${optIndex}`}
                                                            name={question.id}
                                                            value={option}
                                                            checked={question.userAnswer === option}
                                                            onChange={(e) => handleQuizAnswerChange(question.id, e.target.value)}
                                                        />
                                                        <label htmlFor={`${question.id}-opt-${optIndex}`}>{option}</label>
                                                    </div>
                                                ))}
                                                {question.type === 'SA' && (
                                                    <textarea
                                                        className="quiz-sa-input"
                                                        value={question.userAnswer || ''} // Ensure controlled component
                                                        onChange={(e) => handleQuizAnswerChange(question.id, e.target.value)}
                                                        placeholder="Type your answer here..."
                                                        rows="3"
                                                    />
                                                )}
                                            </div>
                                        )}

                                        {question.isRevealed && (
                                            <div className="feedback-area">
                                                <p className="user-answer-display">
                                                    <strong>Your answer:</strong> {question.userAnswer || (question.type === 'SA' ? "No answer provided" : "No option selected")}
                                                </p>
                                                <p className="correct-answer-display">
                                                    <strong>Correct answer:</strong> {question.correctAnswer}
                                                </p>
                                                {question.sourceReference && question.sourceReference !== "Not available" && (
                                                    <div className="source-reference-display">
                                                        <strong>Source Reference:</strong>
                                                        <ReactMarkdown remarkPlugins={[remarkGfm]} components={{ p: React.Fragment }}>
                                                            {question.sourceReference}
                                                        </ReactMarkdown>
                                                    </div>
                                                )}
                                                {question.isCorrect !== undefined && (
                                                    question.isCorrect ? (
                                                        <p className="result-text correct-text">Correct!</p>
                                                    ) : (
                                                        <p className="result-text incorrect-text">Incorrect.</p>
                                                    )
                                                )}
                                            </div>
                                        )}

                                        {!question.isRevealed && (
                                            <button
                                                onClick={() => handleCheckQuizAnswer(question.id)}
                                                className="button button-check-answer"
                                                disabled={
                                                    (question.type === 'MC' || question.type === 'TF')
                                                        ? (question.userAnswer === null || question.userAnswer === undefined)
                                                        : (question.type === 'SA' && (!question.userAnswer || String(question.userAnswer).trim() === ''))
                                                }
                                            >
                                                Check Answer
                                            </button>
                                        )}
                                    </div>
                                ))}
                            </div>
                        )}

                        {/* Message if quiz is toggled open but no data (and no celebration pop-up) */}
                        {quizData.length === 0 && !showCelebration && (
                             <div className="content-section quiz-empty-placeholder" style={{ textAlign: 'center', padding: '20px', border: '1px dashed #eee', marginTop: '10px', backgroundColor: '#f9f9f9' }}>
                                <p>No quiz is currently available for this course.</p>
                            </div>
                        )}
                    </div>
                )}
                {/* --- END CONDITIONALLY RENDERED QUIZ SECTION --- */}


                {(!loading && !error && !courseContent && quizData.length === 0 && !(message && message.includes("generated"))) && (
                     <div className="content-placeholder">
                        <p>No course or quiz content to display.</p>
                    </div>
                )}
                 {(!loading && !error && !courseContent && quizData.length === 0 && message && message.includes("generated")) && (
                    <div className="content-placeholder">
                        <p>Course and quiz content may have been generated. If not displayed, try "Refresh Content".</p>
                    </div>
                )}
            </div>
        </div>
    );
};

export default CoursePage;