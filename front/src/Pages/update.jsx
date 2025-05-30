// src/pages/UserProfilePage.jsx

import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { useSelector, useDispatch } from 'react-redux';
import { selectCurrentUser, /* updateUserSuccess */ } from '../features/auth/authSlice'; // Assuming an updateUserSuccess action might exist
import './userProfile.css'; // We'll create this CSS file next
import { FaSave, FaTimesCircle, FaSpinner } from 'react-icons/fa'; // Optional icons

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:5000';

const UserProfilePage = () => {
    const navigate = useNavigate();
    const dispatch = useDispatch();
    const currentUser = useSelector(selectCurrentUser);

    const [formData, setFormData] = useState({
        email: '',
        firstName: '',
        lastName: '',
        currentPassword: '', // For security, if changing password requires old one
        newPassword: '',
        confirmNewPassword: '',
    });
    const [initialData, setInitialData] = useState({}); // To track changes
    const [loading, setLoading] = useState(false);
    const [message, setMessage] = useState('');
    const [messageType, setMessageType] = useState(''); // 'success', 'error'
    const [errors, setErrors] = useState({});

    useEffect(() => {
        if (currentUser) {
            const userData = {
                email: currentUser.email || '',
                firstName: currentUser.firstName || '',
                lastName: currentUser.lastName || '',
                currentPassword: '',
                newPassword: '',
                confirmNewPassword: '',
            };
            setFormData(userData);
            setInitialData(userData); // Store initial values to compare for changes
        } else {
            // If no current user, redirect or show message
            setMessage('User not found. Please log in.');
            setMessageType('error');
            // setTimeout(() => navigate('/login'), 3000); // Optional redirect
        }
    }, [currentUser, navigate]);

    const handleInputChange = (e) => {
        const { name, value } = e.target;
        setFormData(prev => ({ ...prev, [name]: value }));
        if (errors[name]) {
            setErrors(prev => ({ ...prev, [name]: null }));
        }
    };

    const validateForm = () => {
        const newErrors = {};
        if (!formData.email.trim()) {
            newErrors.email = 'Email is required.';
        } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
            newErrors.email = 'Email address is invalid.';
        }

        // Password validation (only if new password is being set)
        if (formData.newPassword || formData.confirmNewPassword) {
            if (formData.newPassword.length < 8) { // Example: Min 8 chars
                newErrors.newPassword = 'New password must be at least 8 characters.';
            }
            if (formData.newPassword !== formData.confirmNewPassword) {
                newErrors.confirmNewPassword = 'New passwords do not match.';
            }
            // Add this if backend requires current password for a password change
            // if (!formData.currentPassword) {
            //     newErrors.currentPassword = 'Current password is required to change password.';
            // }
        }
        setErrors(newErrors);
        return Object.keys(newErrors).length === 0;
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setMessage('');
        setMessageType('');

        if (!validateForm()) {
            setMessage('Please correct the errors in the form.');
            setMessageType('error');
            return;
        }

        if (!currentUser || !currentUser.username) {
            setMessage('Cannot update profile: User information is missing.');
            setMessageType('error');
            return;
        }

        setLoading(true);

        // Construct payload with only changed fields or fields relevant for update
        const payload = {};
        if (formData.email !== initialData.email) payload.email = formData.email;
        if (formData.firstName !== initialData.firstName) payload.firstName = formData.firstName;
        if (formData.lastName !== initialData.lastName) payload.lastName = formData.lastName;
        
        // Handle password change specifically
        if (formData.newPassword) {
            payload.password = formData.newPassword; // Backend expects 'password' for the new password
            // If your backend requires currentPassword to change password:
            // payload.currentPassword = formData.currentPassword;
        }

        if (Object.keys(payload).length === 0) {
            setMessage('No changes detected.');
            setMessageType(''); // Or 'info'
            setLoading(false);
            return;
        }

        try {
            // Assuming a PATCH request to /api/users/{username}/profile or similar
            // Adjust endpoint and method (PUT/PATCH) as per your backend
            const response = await axios.patch(
                `${BACKEND_URL}/api/users/${currentUser.username}/profile`,
                payload,
                {
                    headers: {
                        'Content-Type': 'application/json',
                        // 'Authorization': `Bearer ${currentUser.token}`, // If auth token is needed
                    },
                    timeout: 30000
                }
            );

            setMessage('Profile updated successfully!');
            setMessageType('success');
            
            // Update initialData to reflect new state
            setInitialData(prev => ({...prev, ...payload}));

            // Clear password fields after successful update
            setFormData(prev => ({
                ...prev,
                currentPassword: '',
                newPassword: '',
                confirmNewPassword: '',
            }));

            // Optionally, dispatch an action to update Redux state if needed
            // if (dispatch && updateUserSuccess) {
            // dispatch(updateUserSuccess({ ...currentUser, ...payload }));
            // }

            console.log("Profile update response:", response.data);

        } catch (error) {
            const errorDetail = error.response?.data?.message || error.response?.data?.error || error.message || 'Failed to update profile.';
            setMessage(`Error: ${errorDetail}`);
            setMessageType('error');
            console.error('Profile update error:', error.response?.data || error);
        } finally {
            setLoading(false);
        }
    };

    const handleCancel = () => {
        navigate('/home'); // Or to a dashboard, or wherever is appropriate
    };

    if (!currentUser) {
        return (
            <div className="user-profile-page-container">
                <div className="user-profile-main-content">
                    <div className="user-profile-panel">
                        <p className="status-message error">Loading user data or user not found...</p>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="user-profile-page-container">
            <div className="user-profile-main-content">
                <div className="user-profile-panel">
                    <h2 className="panel-title">User Profile</h2>
                    <form onSubmit={handleSubmit} className="profile-form">
                        <div className="form-group">
                            <label htmlFor="username">Username</label>
                            <input
                                type="text"
                                id="username"
                                name="username"
                                value={currentUser.username}
                                readOnly
                                className="readonly-input"
                            />
                        </div>

                        <div className="form-group">
                            <label htmlFor="email">Email Address</label>
                            <input
                                type="email"
                                id="email"
                                name="email"
                                value={formData.email}
                                onChange={handleInputChange}
                                disabled={loading}
                                aria-describedby="emailError"
                            />
                            {errors.email && <p id="emailError" className="field-error-message">{errors.email}</p>}
                        </div>

                        <div className="form-group">
                            <label htmlFor="firstName">First Name</label>
                            <input
                                type="text"
                                id="firstName"
                                name="firstName"
                                value={formData.firstName}
                                onChange={handleInputChange}
                                disabled={loading}
                            />
                        </div>

                        <div className="form-group">
                            <label htmlFor="lastName">Last Name</label>
                            <input
                                type="text"
                                id="lastName"
                                name="lastName"
                                value={formData.lastName}
                                onChange={handleInputChange}
                                disabled={loading}
                            />
                        </div>

                        <h3 className="password-section-title">Change Password (optional)</h3>
                        {/*
                        If backend requires current password for password change:
                        <div className="form-group">
                            <label htmlFor="currentPassword">Current Password</label>
                            <input
                                type="password"
                                id="currentPassword"
                                name="currentPassword"
                                value={formData.currentPassword}
                                onChange={handleInputChange}
                                disabled={loading}
                                aria-describedby="currentPasswordError"
                            />
                            {errors.currentPassword && <p id="currentPasswordError" className="field-error-message">{errors.currentPassword}</p>}
                        </div>
                        */}

                        <div className="form-group">
                            <label htmlFor="newPassword">New Password</label>
                            <input
                                type="password"
                                id="newPassword"
                                name="newPassword"
                                value={formData.newPassword}
                                onChange={handleInputChange}
                                disabled={loading}
                                aria-describedby="newPasswordError"
                                placeholder="Leave blank to keep current password"
                            />
                            {errors.newPassword && <p id="newPasswordError" className="field-error-message">{errors.newPassword}</p>}
                        </div>

                        <div className="form-group">
                            <label htmlFor="confirmNewPassword">Confirm New Password</label>
                            <input
                                type="password"
                                id="confirmNewPassword"
                                name="confirmNewPassword"
                                value={formData.confirmNewPassword}
                                onChange={handleInputChange}
                                disabled={loading}
                                aria-describedby="confirmNewPasswordError"
                            />
                            {errors.confirmNewPassword && <p id="confirmNewPasswordError" className="field-error-message">{errors.confirmNewPassword}</p>}
                        </div>

                        <div className="action-buttons-profile">
                            <button
                                type="button"
                                onClick={handleCancel}
                                className="button button-cancel-profile"
                                disabled={loading}
                            >
                                <FaTimesCircle style={{ marginRight: '8px' }} /> Cancel
                            </button>
                            <button
                                type="submit"
                                className="button button-save-profile"
                                disabled={loading || (
                                    formData.email === initialData.email &&
                                    formData.firstName === initialData.firstName &&
                                    formData.lastName === initialData.lastName &&
                                    !formData.newPassword // Disable if no changes
                                )}
                            >
                                {loading ? (
                                    <FaSpinner className="loading-spinner-button" />
                                ) : (
                                    <FaSave style={{ marginRight: '8px' }} />
                                )}
                                {loading ? 'Saving...' : 'Save Changes'}
                            </button>
                        </div>
                    </form>

                    {/* Status/Message Display */}
                    <div className="status-area-profile">
                        {loading && !message && ( /* Show general loading if no specific message yet */
                            <div className="loading-indicator" role="status">
                                <div className="loading-spinner" aria-hidden="true"></div>
                                <p>Processing...</p>
                            </div>
                        )}
                        {message && (
                            <p role="alert" className={`status-message ${messageType}`}>
                                {message}
                            </p>
                        )}
                    </div>

                </div>
            </div>
        </div>
    );
};

export default UserProfilePage;