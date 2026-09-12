import React from 'react';
import { ShieldCheck } from 'lucide-react';
import AbheraEmblem from './AbheraEmblem';

const ChatMessage = ({ message }) => {
  const isUser = message.role === 'user' || message.sender === 'user';
  const text = message.content || message.text;
  const timestamp = message.timestamp;

  return (
    <div className={`chat-consultation-row ${isUser ? 'user-row' : 'assistant-row'}`}>
      <div className={`consultation-note-block ${isUser ? 'user-note-block' : 'assistant-note-block'}`}>
        <div className="note-header">
          <div className="note-header-left">
            {isUser ? (
              <span className="note-kicker-user font-mono">YOUR DESCRIPTION</span>
            ) : (
              <div className="note-kicker-assistant-wrap">
                <AbheraEmblem size={16} />
                <div className="assistant-header-text">
                  <span className="note-kicker-brand serif-heading">ABHERA</span>
                  <span className="note-kicker-sub font-mono">VERIFIED ASSISTANCE</span>
                </div>
              </div>
            )}
          </div>
          {!isUser && (
            <div className="note-verified-tag font-mono">
              <ShieldCheck size={12} />
              <span>VERIFIED ASSISTANCE</span>
            </div>
          )}
        </div>

        <div className="note-body-text">{text}</div>
        
        {timestamp && <div className="note-timestamp-meta font-mono">{timestamp}</div>}
      </div>
    </div>
  );
};

export default ChatMessage;
