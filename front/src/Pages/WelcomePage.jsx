import React from 'react';
import Header from '../components/Header';
import Features from '../components/Features';
import Pricing from '../components/Pricing';
import Navbar from '../components/navbar';
import './WelcomePage.css'; // Import your CSS file for styling

const WelcomePage = () => {
  return (
    <div className="main-container">
        <Navbar/>
        <section id="head"><Header /></section>
      
      <section id="features">
        <Features />
      </section>
      <section id="pricing">
        <Pricing />
      </section>
      
    </div>
  );
};

export default WelcomePage;
