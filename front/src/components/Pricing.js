import React from 'react';
import './Pricing.css';
import CTA from './CTA';
import { useNavigate } from 'react-router-dom'; // Import useNavigate for navigation

const pricingPlans = [
  { 
    title: "Basic Plan", 
    price: "$29/month", 
    description: "Perfect for individual researchers or students. Includes essential AI-powered article analysis and summarization features.",
    features: ["Generative Article Analysis", "Smart Summarization", "Email Support"]
  },
  { 
    title: "Pro Plan", 
    price: "$79/month", 
    description: "Ideal for research teams and educators who need deeper insights, multilingual translation, and enhanced collaboration tools.",
    features: ["All Basic Plan Features", "Advanced Article Insights", "Multilingual Translation", "Real-Time Collaboration", "Priority Support"]
  },
];

const Pricing = () => {
    
const navigate = useNavigate(); 
const handleSubscribe = () => {
    navigate('/form'); 
  };
  return (
    <div className="pricing-section">
      <h2>Pricing Plans</h2>
      <div className="pricing-cards">
        {pricingPlans.map((plan, index) => (
          <div key={index} className="pricing-card">
            <h3>{plan.title}</h3>
            <p className="price">{plan.price}</p>
            <p>{plan.description}</p>
            <ul>
              {plan.features.map((feature, idx) => (
                <li key={idx}>{feature}</li>
              ))}
            </ul>
            <button className="select-plan">Select Plan</button>
          </div>
        ))}
      </div>
      <CTA text="Subscribe Now" onClick={handleSubscribe} size="large" />
    </div>
  );
};

export default Pricing;
