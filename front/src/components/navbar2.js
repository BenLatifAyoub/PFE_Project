import React from 'react';
import './Navbar.css'; // Ensure this CSS file contains the styles
import {  useNavigate } from 'react-router-dom';

const Navbar = ({ onLogout }) => {
    const navigate = useNavigate();
    const handleUpdateInformation = () => {
        navigate('/update'); 
      };
  const handlelogout = () => {
    navigate('/');  
  };
    return (
      <nav className="navbar">
      <div className="navbar-container">
        <div className="navbar-logo">
          <a href="#head">Scientific Article Analyzer</a>
        </div>
        <ul className="navbar-links">
        <li><a onClick={handleUpdateInformation} >Update Information</a></li>
        </ul>
        <div className="navbar-auth">
        <a onClick={handlelogout} className="auth-button">Logout</a>
        </div>
      </div>
    </nav>
    );
  };
  

export default Navbar;
