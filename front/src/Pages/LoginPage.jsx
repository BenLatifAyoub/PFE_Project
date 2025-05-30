import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import { useDispatch, useSelector } from 'react-redux'; // Import Redux hooks
import { loginStart, loginSuccess, loginFailure, selectAuthStatus, selectAuthError } from '../features/auth/authSlice'; // Adjust path
import './form.css'; // Ensure this path is correct

// Define the base URL for your Flask API
const API_BASE_URL = 'http://localhost:5000/api'; // Adjust if your Flask runs elsewhere

// --- Remove the onLogin prop unless specifically needed for non-auth reasons ---
// function Login({ onLogin }) {
function Login() {
  const [formData, setFormData] = useState({
    username: '',
    password: ''
  });
  // Use Redux state for loading and potentially errors, or keep local error for form-specific feedback
  const dispatch = useDispatch();
  const authStatus = useSelector(selectAuthStatus); // Get status ('idle', 'loading', 'succeeded', 'failed')
  const authError = useSelector(selectAuthError); // Get error message from Redux state if login failed globally

  const [localError, setLocalError] = useState(null); // Keep local error for immediate form feedback
  const isSubmitting = authStatus === 'loading'; // Determine loading state from Redux

  const navigate = useNavigate();

  // Clear local error if global auth error changes (e.g., after a failed attempt)
  useEffect(() => {
    if (authStatus === 'failed') {
        setLocalError(authError); // Optionally sync Redux error to local state
    } else {
        setLocalError(null); // Clear local error if status is not 'failed'
    }
  }, [authStatus, authError]);


  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value
    });
     setLocalError(null); // Clear error on input change
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLocalError(null); // Clear previous local errors
    dispatch(loginStart()); // Dispatch action to indicate loading state

    try {
      const response = await axios.post(`${API_BASE_URL}/login`, formData, {
        headers: {
          'Content-Type': 'application/json',
        }
      });

      console.log('Login successful:', response.data);

      // --- Dispatch success action to Redux ---
      // Pass the relevant data from the response to the action payload
      // Adjust payload based on your API response structure (e.g., response.data.user, response.data.token)
      dispatch(loginSuccess(response.data));

      // --- Remove direct localStorage calls (handled by the slice if needed) ---
      // localStorage.setItem('user', JSON.stringify(response.data));
      // localStorage.setItem('username', response.data.username);
      // if (response.data.token) { localStorage.setItem('token', response.data.token); }

      // --- Remove onLogin prop call (state is managed globally) ---
      // if (typeof onLogin === 'function') {
      //   onLogin(response.data);
      // }

      navigate('/home'); // Navigate on success

    } catch (err) {
      console.error('Error logging in:', err.response ? err.response.data : err.message);
      let errorMessage = 'Login failed. Please try again.'; // Default error

      if (err.response && err.response.status === 401) {
        errorMessage = 'Invalid username or password.';
      } else if (err.response && err.response.data && err.response.data.error) {
         errorMessage = err.response.data.error; // Use specific error from backend if available
      } else if (!err.response) {
        errorMessage = 'Login failed. Could not connect to server.'; // Network error
      }

      setLocalError(errorMessage); // Set local error for immediate feedback on the form
      dispatch(loginFailure(errorMessage)); // Dispatch failure action to Redux

    }
    // No finally block needed to set isSubmitting = false, as it's derived from Redux state
  };

  const handleSubscribe = () => {
    navigate('/signup');
  };

  return (
    <div className="container">
      <div className="form-containe"> {/* Corrected class name */}
        <h2>Login</h2>
        {/* Display local error message */}
        {localError && <div className="error-message">{localError}</div>}
        {/* Optional: Display global error from Redux if needed */}
        {/* {authStatus === 'failed' && !localError && <div className="error-message">{authError}</div>} */}

        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="username">Username</label>
            <input
              type="text"
              id="username"
              name="username"
              value={formData.username}
              onChange={handleChange}
              required
              disabled={isSubmitting} // Disable input during submission (loading state)
            />
          </div>
          <div className="form-group">
            <label htmlFor="password">Password</label>
            <input
              type="password"
              id="password"
              name="password"
              value={formData.password}
              onChange={handleChange}
              required
              disabled={isSubmitting} // Disable input during submission
            />
          </div>
          <div className="button-container">
            <button type="submit" className="subscribe" disabled={isSubmitting}>
              {isSubmitting ? 'Logging in...' : 'Login'}
            </button>
            <button
              type="button"
              onClick={handleSubscribe}
              className="comeback"
              disabled={isSubmitting}
            >
              Subscribe
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export default Login;