import React, { useState } from 'react';
import { Send, Lock } from 'lucide-react';

const ChatInput = ({ onSendMessage, disabled }) => {
  const [inputText, setInputText] = useState('');

  const handleSend = () => {
    const trimmed = inputText.trim();
    if (!trimmed || disabled) return;
    onSendMessage(trimmed);
    setInputText('');
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="chat-input-wrapper">
      <form
        className="chat-input-form"
        onSubmit={(e) => {
          e.preventDefault();
          handleSend();
        }}
      >
        <textarea
          className="chat-textarea"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Describe what happened in your own words..."
          rows={2}
          disabled={disabled}
          aria-label="Message input"
        />
        <button
          type="submit"
          className="send-button"
          disabled={!inputText.trim() || disabled}
          aria-label="Send message"
          title="Send Message"
        >
          <span>SEND</span>
          <Send size={14} />
        </button>
      </form>
      <div className="chat-input-privacy-note">
        <Lock size={12} />
        <span>Your information is handled within this consultation session. • Shift+Enter for new line</span>
      </div>
    </div>
  );
};

export default ChatInput;
