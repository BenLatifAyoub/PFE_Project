import React from 'react';
import { motion } from 'framer-motion';
import './Features.css';

const features = [
  { 
    title: "Generative AI Analysis", 
    description: "Our AI-Powered Scientific Article Analyzer uses state-of-the-art generative AI to deeply analyze and understand the core ideas, structure, and nuances of your scientific articles. It offers insightful summaries, accurate translations, and even custom learning paths based on your content. The AI adapts to your past submissions, continuously improving its recommendations to ensure your analysis remains precise, efficient, and personalized."
  },
  {
    title: "Personalized Learning Recommendations",
    description: "Based on the content of your articles, our system suggests tailored courses, research papers, and learning materials to help you expand your knowledge and stay up-to-date with the latest developments in your field."
  }
];

const Features = () => {
  return (
    <div className="features-section">
      <h2>Features</h2>
      <div className="feature-cards">
        {features.map((feature, index) => (
          <motion.div
            key={index}
            className="feature-card"
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.2, duration: 0.5 }}
          >
            <h3>{feature.title}</h3>
            <p>{feature.description}</p>
          </motion.div>
        ))}
      </div>
    </div>
  );
};

export default Features;
