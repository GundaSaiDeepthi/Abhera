import React from 'react';
import { Shield } from 'lucide-react';

const LoadingIndicator = () => {
  return (
    <div className="chat-message-row assistant-row loading-row">
      <div className="message-avatar">
        <Shield size={18} className="assistant-icon" />
      </div>
      <div className="message-bubble-wrapper">
        <div className="message-bubble assistant-bubble loading-bubble">
          <div className="loading-dots">
            <span className="dot"></span>
            <span className="dot"></span>
            <span className="dot"></span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LoadingIndicator;
