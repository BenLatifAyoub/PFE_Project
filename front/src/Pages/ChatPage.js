    // src/Pages/ChatPage.jsx
    import React, { useState, useEffect, useRef } from 'react';
    import { useNavigate, useLocation } from 'react-router-dom'; // Import useLocation
    import axios from 'axios';
    import './chatpage.css'; // Optional: Create this file for specific chat styles

    const ChatPage = () => {
        const navigate = useNavigate();
        const location = useLocation(); // Get location object

        // --- Retrieve contextId and pdfName from navigation state ---
        const contextId = location.state?.contextId; // Database article ID
        const pdfName = location.state?.pdfName || "Document"; // Name of the PDF for display

        const [messages, setMessages] = useState([
            // Updated initial message to mention the document context
            { sender: 'bot', text: `Hello! Ask me anything about "${pdfName}".` }
        ]);
        const [userInput, setUserInput] = useState('');
        const [isLoading, setIsLoading] = useState(false);
        const [error, setError] = useState('');
        const chatEndRef = useRef(null); // Ref to scroll to the bottom

        const BACKEND_URL = 'http://localhost:5000'; // Ensure this matches your Flask backend URL

        // Effect to handle missing context ID (e.g., direct navigation)
        useEffect(() => {
            if (contextId === undefined || contextId === null) {
                setError("Error: No document context ID found. Please analyze a document first.");
                // Disable input and send button? Or navigate back?
                // Option: Navigate back after a delay
                const timer = setTimeout(() => {
                    navigate('/analyze', { replace: true }); // Go back to analyze page
                }, 3000);
                return () => clearTimeout(timer); // Cleanup timer on component unmount
            }
        }, [contextId, navigate]);


        // Scroll to the bottom of the chat window when messages update
        useEffect(() => {
            chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
        }, [messages]);

        const handleInputChange = (event) => {
            setUserInput(event.target.value);
        };

        // --- MODIFIED: handleSendMessage ---
        const handleSendMessage = async () => {
            const trimmedInput = userInput.trim();
            // Also check if contextId is valid before sending
            if (!trimmedInput || isLoading || contextId === undefined || contextId === null) return;

            const newUserMessage = { sender: 'user', text: trimmedInput };
            setMessages(prevMessages => [...prevMessages, newUserMessage]);
            setUserInput('');
            setIsLoading(true);
            setError('');

            try {
                console.log(`Sending to backend: Message='${trimmedInput}', ContextID=${contextId}`);
                // --- Include context_id in the request body ---
                const response = await axios.post(`${BACKEND_URL}/api/chat`, {
                    message: trimmedInput,
                    context_id: contextId // Send the database article ID
                });

                const botResponse = response.data.response;

                if (botResponse === undefined || botResponse === null) { // Check for undefined/null specifically
                    throw new Error("Received invalid response from the bot.");
                }

                const newBotMessage = { sender: 'bot', text: botResponse };
                setMessages(prevMessages => [...prevMessages, newBotMessage]);

            } catch (err) {
                console.error('Error fetching bot response:', err.response?.data || err.message);
                const errorMsg = err.response?.data?.error || err.message || "Sorry, I couldn't get a response. Please try again.";
                setError(`Error: ${errorMsg}`);
                setMessages(prevMessages => [...prevMessages, { sender: 'bot', text: `Sorry, I encountered an error: ${errorMsg.substring(0,100)}...` }]);
            } finally {
                setIsLoading(false);
            }
        };

        // Allow sending message with Enter key (remains the same)
        const handleKeyDown = (event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                handleSendMessage();
            }
        };

        const handleReturn = () => {
            navigate('/analyze'); // Navigate back to the analyze page
        };

        // --- JSX ---
        return (
            // Outer container styling remains the same
            <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh', margin: 0, padding: '30px', boxSizing: 'border-box', backgroundColor: 'transparent', color: '#ffffff' }}>

                {/* Main Chat Area Container Styling remains the same */}
                <div style={{
                    maxWidth: '900px', width: '100%', margin: '0 auto',
                    display: 'flex', flexDirection: 'column', flexGrow: 1,
                    background: 'rgba(0, 0, 0, 0.8)', borderRadius: '12px',
                    boxShadow: '0 4px 8px rgba(0, 0, 0, 0.3)', padding: '20px',
                    overflow: 'hidden'
                }}>

                    {/* Header - Updated to show PDF name */}
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px', paddingBottom: '10px', borderBottom: '2px solid #ffb74d' }}>
                        <h2 style={{ fontWeight: 'bold', fontSize: '1.4em', /* Adjusted size */ color: '#ffb74d', margin: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                            {/* Display PDF Name */}
                            AI Chat: {pdfName} {contextId !== undefined ? `(ID: ${contextId})` : ''}
                        </h2>
                        <button onClick={handleReturn} style={{ backgroundColor: '#555', color: '#ffffff', fontWeight: 'bold', borderRadius: '8px', border: 'none', padding: '10px 20px', cursor: 'pointer', transition: 'background-color 0.3s ease', flexShrink: 0 /* Prevent shrinking */ }}>
                            Back to Analyze
                        </button>
                    </div>


                    {/* Chat Messages Display Styling remains the same */}
                    <div style={{ flexGrow: 1, overflowY: 'auto', marginBottom: '20px', paddingRight: '10px' }}>
                        {messages.map((msg, index) => (
                            <div
                                key={index}
                                style={{
                                    display: 'flex',
                                    justifyContent: msg.sender === 'user' ? 'flex-end' : 'flex-start',
                                    marginBottom: '12px',
                                }}
                            >
                                <div style={{
                                    background: msg.sender === 'user' ? '#4a5d7e' : '#424242',
                                    color: '#ffffff', padding: '10px 15px', borderRadius: '15px',
                                    maxWidth: '75%', wordWrap: 'break-word',
                                    textAlign: msg.sender === 'user' ? 'right' : 'left',
                                    border: msg.sender === 'bot' ? '1px solid #555' : '1px solid #6a8ed8',
                                    borderBottomRightRadius: msg.sender === 'user' ? '4px' : '15px',
                                    borderBottomLeftRadius: msg.sender === 'bot' ? '4px' : '15px',
                                }}>
                                    {msg.text}
                                </div>
                            </div>
                        ))}
                        <div ref={chatEndRef} />
                    </div>

                    {/* Loading Indicator Styling remains the same */}
                    {isLoading && (
                        <div style={{ display: 'flex', justifyContent: 'flex-start', alignItems: 'center', marginBottom: '15px', paddingLeft: '10px' }}>
                            <div className="loading-spinner" style={{ width: '20px', height: '20px', borderTopColor: '#ffb74d', marginRight: '10px' }}></div>
                            <p style={{ color: '#bdbdbd', fontStyle: 'italic', margin: 0 }}>AI is thinking...</p>
                        </div>
                    )}

                    {/* Error Message Display Styling remains the same */}
                    {error && (
                        <p style={{ color: '#ef5350', textAlign: 'center', marginBottom: '15px', fontSize: '0.9em' }}>
                            {error}
                        </p>
                    )}

                    {/* Input Area Styling remains the same */}
                    <div style={{ display: 'flex', gap: '10px', borderTop: '1px solid #555', paddingTop: '15px' }}>
                        <input
                            type="text"
                            value={userInput}
                            onChange={handleInputChange}
                            onKeyDown={handleKeyDown}
                            placeholder="Type your message here..."
                            // Disable input if loading or if contextId is missing
                            disabled={isLoading || contextId === undefined || contextId === null}
                            style={{
                                flexGrow: 1, padding: '12px', backgroundColor: '#424242', color: '#ffffff',
                                border: '1px solid #ffb74d', borderRadius: '8px', outline: 'none'
                            }}
                        />
                        <button
                            onClick={handleSendMessage}
                            // Disable button if loading, input is empty, or contextId is missing
                            disabled={isLoading || !userInput.trim() || contextId === undefined || contextId === null}
                            style={{
                                backgroundColor: !isLoading && userInput.trim() && contextId !== undefined && contextId !== null ? '#ffb74d' : '#9e9e9e',
                                color: '#ffffff', fontWeight: 'bold', borderRadius: '8px', border: 'none',
                                padding: '12px 25px', cursor: !isLoading && userInput.trim() && contextId !== undefined && contextId !== null ? 'pointer' : 'not-allowed',
                                transition: 'background-color 0.3s ease', flexShrink: 0
                            }}
                        >
                            Send
                        </button>
                    </div>
                </div>
            </div>
        );
    };

    export default ChatPage;