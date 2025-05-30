import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import axios from 'axios';
import { useSelector } from 'react-redux'; // <-- ADDED: Import useSelector

// ADDED: Import Redux selectors
import {
  selectCurrentUser,
  selectIsAuthenticated,
  selectAuthStatus,
} from '../features/auth/authSlice'; // <-- Adjust path if necessary

// Chart imports (keep if needed elsewhere or remove if not used)
import { Bar, Line, Doughnut } from 'react-chartjs-2';
import { Chart, CategoryScale, LinearScale, BarElement, LineElement, DoughnutController, PointElement, Title, Tooltip, Legend, ArcElement } from 'chart.js';

import Navbar from '../components/navbar2.js';
import Modal from '../components/modal';
import './Home.css';

// Register Chart.js components (keep if needed or remove)
Chart.register(CategoryScale, LinearScale, BarElement, LineElement, DoughnutController, PointElement, ArcElement, Title, Tooltip, Legend);

const API_BASE_URL = 'http://localhost:5000/api';

function Home({ onLogout }) {
  const navigate = useNavigate();

  // --- Get Authentication State from Redux ---
  const currentUser = useSelector(selectCurrentUser);         // <-- ADDED
  const isAuthenticated = useSelector(selectIsAuthenticated); // <-- ADDED
  const authStatus = useSelector(selectAuthStatus);           // <-- ADDED
  const username = currentUser?.username;                     // <-- ADDED: Safely access username

  // --- Component State (remains the same) ---
  const [recentArticles, setRecentArticles] = useState([]);
  const [isLoadingList, setIsLoadingList] = useState(false);
  const [listFetchError, setListFetchError] = useState(null);

  const [selectedArticleDetails, setSelectedArticleDetails] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isLoadingModal, setIsLoadingModal] = useState(false);
  const [modalFetchError, setModalFetchError] = useState(null);

  const handleAnalyze = () => {
    navigate('/analyze');
  };
    const handleGenrate = () => {
    navigate('/generate');
  };

  // --- MODIFIED: Fetch Initial Summarized List of Recent Articles ---
  useEffect(() => {
    const fetchRecentArticlesList = async () => {
      // This function is now called conditionally based on Redux state
      console.log(`Fetching list data for user: ${username}`);
      setIsLoadingList(true);
      setListFetchError(null);
      setRecentArticles([]); // Clear previous results

      try {
        const userResponse = await axios.get(`${API_BASE_URL}/users/${username}`);
        const recentIds = userResponse.data?.recently_analyzed_ids;

        if (!recentIds || !Array.isArray(recentIds) || recentIds.length === 0) {
          console.log("No recent article IDs found.");
          // No error, UI will show "No articles analyzed yet."
          setIsLoadingList(false);
          return;
        }

        const idsToFetch = recentIds.slice(0, 5);
        const idsQueryString = idsToFetch.join(',');
        console.log(`Fetching summarized details for IDs: ${idsQueryString}`);

        const articlesResponse = await axios.get(`${API_BASE_URL}/articles/details`, {
          params: { ids: idsQueryString }
        });

        if (Array.isArray(articlesResponse.data)) {
          setRecentArticles(articlesResponse.data);
        } else {
          console.error("Received non-array for summarized article details:", articlesResponse.data);
          setListFetchError('Unexpected data format for articles list.');
        }
      } catch (error) {
        console.error('Error fetching recent articles list:', error.response ? error.response.data : error.message, error);
        if (error.response?.status === 401 || error.response?.status === 403) {
            setListFetchError('Authentication error. Please log in again.');
        } else {
            setListFetchError('Failed to load recent articles list.');
        }
      } finally {
        setIsLoadingList(false);
      }
    };

    if (isAuthenticated && username) {
      // User is authenticated and we have a username, proceed to fetch.
      fetchRecentArticlesList();
    } else if (authStatus !== 'loading' && authStatus !== 'idle' && !isAuthenticated) {
      // Auth process is complete (not loading/idle), and user is NOT authenticated.
      setListFetchError("Login required to view recent articles.");
      setRecentArticles([]); // Clear any potentially stale data
      setIsLoadingList(false); // Ensure loading is off if it was on
    } else if (authStatus === 'loading' || authStatus === 'idle') {
      // Auth state is still being determined (e.g., rehydration).
      // Show a loading state for the list or wait.
      setIsLoadingList(true); // Indicate that we are waiting for auth/data
      setListFetchError(null); // Clear any previous errors
      setRecentArticles([]); // Clear articles
    }

  }, [username, isAuthenticated, authStatus]); // <-- MODIFIED: Dependencies

  // --- Function to Fetch Full Details for ONE Article (for the Modal) - UNCHANGED ---
  const fetchFullArticleDetails = useCallback(async (articleId) => {
    if (!articleId) return;

    setIsModalOpen(true);
    setIsLoadingModal(true);
    setModalFetchError(null);
    setSelectedArticleDetails(null);

    console.log(`Fetching full details for article ID: ${articleId}`);

    try {
      const response = await axios.get(`${API_BASE_URL}/articles/${articleId}`);
      setSelectedArticleDetails(response.data);
    } catch (error) {
      console.error(`Error fetching full details for article ${articleId}:`, error.response ? error.response.data : error.message, error);
      if (error.response?.status === 404) {
         setModalFetchError(`Article with ID ${articleId} not found.`);
      } else if (error.response?.status === 401 || error.response?.status === 403){
        setModalFetchError('Authentication error. Please log in again.');
      } else {
         setModalFetchError('Failed to load article details.');
      }
      setSelectedArticleDetails(null);
    } finally {
      setIsLoadingModal(false);
    }
  }, []); // API_BASE_URL could be added if it were dynamic

  // --- Click Handler for Table Rows - UNCHANGED ---
  const handleRowClick = (articleId) => {
    fetchFullArticleDetails(articleId);
  };

  // --- Close Modal Handler - UNCHANGED ---
  const closeModal = () => {
    setIsModalOpen(false);
    setSelectedArticleDetails(null);
    setModalFetchError(null);
    setIsLoadingModal(false);
  };

  // --- MODIFIED: JSX Rendering - Conditional logic for recent articles section updated ---
  return (
    <>
      <Navbar onLogout={onLogout} />
      <div className="dashboard-container">
        {/* Header and Analyze Button remain the same */}
        <header className="hero-section">
          <h1 className="page-title">Welcome to the Scientific Article Analyzer</h1>
          <p className="hero-subtitle">Streamline your article analysis process with ease.</p>
        </header>
        <div className="button-group">
          <button onClick={handleAnalyze} className="btn btn-analyze">Analyze a New Article</button>
        </div>
        <div className="button-group">
          <button onClick={handleGenrate} className="btn btn-generate">Multiple course generation</button>
        </div>

        {/* Recent Articles Section */}
        <section className="key-metrics">
          <div className="card table-card">
            <h2 className="card-title">Recently Analyzed</h2>
            {/* MODIFIED: Conditional rendering based on Redux and local loading states */}
            {(isLoadingList || (authStatus === 'loading' || authStatus === 'idle')) && !listFetchError && <p>Loading recent articles...</p>}
            {listFetchError && <p className="error-message">{listFetchError}</p>}
            {!isLoadingList && !listFetchError && isAuthenticated && recentArticles.length === 0 && (
              <p>No articles analyzed yet. Click "Analyze a New Article" to start!</p>
            )}
            {!isLoadingList && !listFetchError && isAuthenticated && recentArticles.length > 0 && (
              <table className="score-table">
                <thead>
                  <tr>
                    <th>Title</th>
                    <th>NB of Pages</th>
                    <th>Description</th>
                  </tr>
                </thead>
                <tbody>
                  {recentArticles.map((articleSummary) => (
                    <tr
                      key={articleSummary.id}
                      className="clickable-row"
                      onClick={() => handleRowClick(articleSummary.id)}
                    >
                      <td>
                        <Link to="#" onClick={(e) => { e.preventDefault(); handleRowClick(articleSummary.id); }} className="table-link">
                          {articleSummary.title || `Article ID ${articleSummary.id}`}
                        </Link>
                      </td>
                      <td>{typeof articleSummary.pages === 'number' ? `${articleSummary.pages}` : 'N/A'}</td>
                      <td>{articleSummary.description || 'N/A'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
            {/* Message if not authenticated and auth is processed (covered by listFetchError generally) */}
            {authStatus !== 'loading' && authStatus !== 'idle' && !isAuthenticated && !listFetchError && (
                 <p className="error-message">Login required to view recent articles.</p>
            )}
          </div>
        </section>
      </div>

      {/* Render the Modal - UNCHANGED */}
      <Modal
          isOpen={isModalOpen}
          onClose={closeModal}
          article={selectedArticleDetails}
          isLoading={isLoadingModal}
          error={modalFetchError}
      />
    </>
  );
}

export default Home;