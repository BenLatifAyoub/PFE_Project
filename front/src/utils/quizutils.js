// Place these functions inside your CoursePage.jsx component or import them

function parseQuizContent(quizText) {
    if (!quizText || typeof quizText !== 'string') return [];
    const questions = [];
    const lines = quizText.split('\n').filter(line => line.trim() !== '' && /^\((MC|TF|SA)\)/.test(line));

    let questionCounter = 0;

    for (const line of lines) {
        const typeMatch = line.match(/^\((MC|TF|SA)\)/);
        if (!typeMatch) continue;

        const type = typeMatch[1];
        let questionText = '';
        let options = [];
        let correctAnswerFromText = ''; // This will be the letter for MC, or full answer for TF/SA
        const id = `q-${questionCounter++}-${Date.now()}`; // More unique ID

        const answerSplit = line.split(/\s*Answer:\s*/);
        if (answerSplit.length < 2) {
            console.warn("Malformed quiz line (no 'Answer:'):", line);
            continue;
        }

        // Content before "Answer:"
        const questionPartRaw = answerSplit[0].substring(typeMatch[0].length).trim();
        correctAnswerFromText = answerSplit[1].trim();

        let finalCorrectAnswer = correctAnswerFromText; // Will be updated for MC

        if (type === 'MC') {
            // Example: "What is X? A. Opt1 B. Opt2 C. Opt3"
            // correctAnswerFromText is 'B' (the letter)

            const optionMarkers = ['A.', 'B.', 'C.', 'D.', 'E.', 'F.']; // Common option markers
            let firstOptionMarkerIndex = -1;
            let firstOptionMarkerLength = 0;

            // Find the start of the options block
            for (const marker of optionMarkers) {
                const idx = questionPartRaw.indexOf(marker);
                if (idx !== -1) {
                    // Ensure it's a standalone marker (e.g., not part of a word like "A.M.")
                    // This check is basic; might need refinement for edge cases.
                    if (idx === 0 || /\s$/.test(questionPartRaw.charAt(idx -1))) {
                        if (firstOptionMarkerIndex === -1 || idx < firstOptionMarkerIndex) {
                            firstOptionMarkerIndex = idx;
                            firstOptionMarkerLength = marker.length;
                        }
                    }
                }
            }

            if (firstOptionMarkerIndex !== -1) {
                questionText = questionPartRaw.substring(0, firstOptionMarkerIndex).trim();
                const optionsPart = questionPartRaw.substring(firstOptionMarkerIndex);

                // Regex to extract "A. Option text"
                const optionRegex = /([A-Z])\.\s*(.*?)(?=\s*[A-Z]\.\s*|$)/g;
                let match;
                const tempOptions = []; // Stores { letter: 'A', text: 'Option A' }

                while ((match = optionRegex.exec(optionsPart)) !== null) {
                    tempOptions.push({ letter: match[1], text: match[2].trim() });
                }
                options = tempOptions.map(opt => opt.text);

                const correctOptionObject = tempOptions.find(opt => opt.letter === correctAnswerFromText);
                if (correctOptionObject) {
                    finalCorrectAnswer = correctOptionObject.text; // Use the full text of the correct option
                } else {
                    console.warn(`MCQ: Correct answer letter '${correctAnswerFromText}' not found in options for: ${questionText}. Using letter as answer.`);
                    finalCorrectAnswer = correctAnswerFromText; // Fallback, though user will select text
                }
            } else {
                questionText = questionPartRaw; // No A. B. C. options found
                console.warn("MCQ detected, but no standard A.B.C. options found:", questionPartRaw);
            }
        } else if (type === 'TF') {
            questionText = questionPartRaw.trim();
            options = ['True', 'False'];
            // finalCorrectAnswer is already "True" or "False"
        } else if (type === 'SA') {
            questionText = questionPartRaw.trim();
            // finalCorrectAnswer is the full text string
        }

        if (questionText) {
            questions.push({
                id,
                type,
                questionText,
                options: options.length > 0 ? options : undefined,
                correctAnswer: finalCorrectAnswer,
                userAnswer: type === 'SA' ? '' : null, // SA: empty string, MC/TF: null initially
                isRevealed: false,
            });
        } else {
            console.warn("Could not parse question text for line:", line);
        }
    }
    return questions;
}

function compareAnswers(type, userAnswer, correctAnswer) {
    if (userAnswer === null || userAnswer === undefined) return false;

    // Normalize by trimming; case sensitivity depends on type
    const uaTrimmed = String(userAnswer).trim();
    const caTrimmed = String(correctAnswer).trim();

    if (type === 'MC' || type === 'TF') {
        // MC/TF answers are selected from predefined options, so direct match (case-sensitive) is fine.
        return uaTrimmed === caTrimmed;
    }
    if (type === 'SA') {
        // For Short Answers, it's common to ignore case.
        // Exact match is strict. Consider partial matches or keyword spotting for more advanced SA.
        // For now, case-insensitive exact match.
        return uaTrimmed.toLowerCase() === caTrimmed.toLowerCase();
    }
    return false;
}