import React from 'react';
import './CTA.css';

const CTA = ({ text, onClick, size = "medium" }) => {
  return (
    <button className={`cta-btn ${size}`} onClick={onClick}>
      {text}
    </button>
  );
};

export default CTA;
