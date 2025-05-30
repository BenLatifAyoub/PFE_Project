import React, { useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import './Signup.css'; // Ensure this path is correct
import arrowIcon from './google-Photoroom.png'; // Ensure this path is correct
import arrowIcon1 from './facebook-Photoroom.png'; // Ensure this path is correct

// Define the base URL for your Flask API
const API_BASE_URL = 'http://localhost:5000/api'; // Adjust if your Flask runs elsewhere

// Removed onLogin prop as signup doesn't automatically log in with this API
export default function SignUpPage() {
  const [formData, setFormData] = useState({
    email: '',
    username: '',
    password: ''
  });
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false); // Prevent double clicks
  const navigate = useNavigate();

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value
    });
    // Clear success/error message when user types again
    if (success) setSuccess(false);
    if (error) setError(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault(); // Prevent default form submission
    setError(null);     // Clear previous errors
    setSuccess(false);  // Clear previous success message
    setIsSubmitting(true); // Disable button

    try {
      // Make POST request to your Flask user creation endpoint
      const response = await axios.post(`${API_BASE_URL}/users`, formData, {
        headers: {
          'Content-Type': 'application/json',
          // No CSRF token needed for this Flask API setup
        }
      });

      // Response status 201 Created indicates success
      console.log('Signup successful:', response.data); // Should contain { message: '...', user_id: ... }
      setSuccess(true); // Show success message

      // --- Optional: Automatically navigate to login after a short delay ---
      setTimeout(() => {
        navigate('/login'); // Navigate to login page so user can sign in
      }, 1500); // Wait 1.5 seconds to show the success message

      // --- Alternative: Don't navigate automatically ---
      // Let the user click the login button after seeing the success message.
      // No navigate('/login') call here in that case.

    } catch (err) {
      console.error('Error signing up:', err.response ? err.response.data : err.message);

      if (err.response) {
        const status = err.response.status;
        const responseData = err.response.data;
        if (status === 409 && responseData.details) { // 409 Conflict (duplicate)
          setError(responseData.details); // E.g., "Username 'x' already exists."
        } else if (status === 400 && responseData.error) { // 400 Bad Request (validation)
           setError(responseData.error + (responseData.details ? ` (${responseData.details})` : ''));
        } else {
           // Other server-side error
           setError(responseData.error || 'Sign-up failed. Please try again.');
        }
      } else {
        // Network error or other issue where no response was received
        setError('Sign-up failed. Could not connect to server.');
      }
      setSuccess(false); // Ensure success is false if there was an error
    } finally {
      setIsSubmitting(false); // Re-enable button regardless of outcome
    }
  };

  const handleLogin = () => {
    navigate('/login'); // Navigate to your login page
  };

  return (
    <div className="container"> {/* Ensure class names match your CSS */}
      <div className="form-container"> {/* Ensure class names match your CSS */}
        <h2>Sign Up</h2>
        {/* Show success message OR error message */}
        {success && <div className="success-message">Sign-up successful! Redirecting to login...</div>}
        {error && <div className="error-message">{error}</div>}
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="email">Email</label>
            <input
              type="email"
              id="email"
              name="email"
              value={formData.email}
              onChange={handleChange}
              required
              disabled={isSubmitting} // Disable during submission
            />
          </div>
          <div className="form-group">
            <label htmlFor="username">Username</label>
            <input
              type="text"
              id="username"
              name="username"
              value={formData.username}
              onChange={handleChange}
              required
              disabled={isSubmitting} // Disable during submission
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
              minLength="6" // Add basic client-side validation matching backend
              disabled={isSubmitting} // Disable during submission
            />
          </div>
          <div className="button-container">
            <button type="submit" className="signup-button" disabled={isSubmitting}>
              {isSubmitting ? 'Signing Up...' : 'Sign Up'}
            </button>
            {/* Keep the Login button, useful if auto-redirect is removed or fails */}
            <button
              type="button"
              onClick={handleLogin}
              className="login-button"
              disabled={isSubmitting}
            >
              Login
            </button>
          </div>
          {/* Keep the social icons if they are for show or link elsewhere */}
          <div className="icon-container">
            <img src={arrowIcon} alt="Google Sign-in Placeholder" className="arrow-icon" />
            <img src={arrowIcon1} alt="Facebook Sign-in Placeholder" className="arrow-icon" />
          </div>
        </form>
      </div>
    </div>
  );
}