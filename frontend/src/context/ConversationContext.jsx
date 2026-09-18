/**
 * ABHERA — React Conversation Context & State Management
 *
 * This context provider manages the central global state for the ABHERA web application,
 * coordinating data flow between UI components, the conversational intake pipeline,
 * and REST API service calls.
 *
 * State Architecture & Responsibilities:
 * ---------------------------------------
 * 1. Session Persistence: Synchronizes `sessionId` and `submissionId` with `sessionStorage`
 *    to preserve active survivor intake sessions across browser refreshes.
 * 2. Message History: Tracks ordered list of user and bot messages with metadata.
 * 3. ML Results & Entities: Stores real-time predicted legal labels and extracted NER entities.
 * 4. Questionnaire State: Manages current active dynamic question and question answers.
 * 5. Report Sync: Holds final generated incident report markdown and metadata.
 * 6. UI Async Feedback: Controls global `loading` indicator state and `error` boundary messages.
 */

import React, { createContext, useContext, useState, useCallback } from 'react';

const initialConversationState = {
  sessionId: null,
  submissionId: null,
  messages: [],
  answers: [],
  predictedLabels: [],
  entities: {},
  currentQuestion: null,
  conversationStatus: 'ACTIVE',
  report: null,
  loading: false,
  error: null,
};

const getStoredSessionId = () => {
  try {
    return sessionStorage.getItem('abhera_session_id') || null;
  } catch (e) {
    return null;
  }
};

const getStoredSubmissionId = () => {
  try {
    return sessionStorage.getItem('abhera_submission_id') || null;
  } catch (e) {
    return null;
  }
};

export const ConversationContext = createContext(null);

export const ConversationProvider = ({ children }) => {
  const [sessionId, setSessionIdState] = useState(getStoredSessionId);
  const [submissionId, setSubmissionIdState] = useState(getStoredSubmissionId);
  const [messages, setMessages] = useState(initialConversationState.messages);
  const [answers, setAnswers] = useState(initialConversationState.answers);
  const [predictedLabels, setPredictedLabels] = useState(initialConversationState.predictedLabels);
  const [entities, setEntities] = useState(initialConversationState.entities);
  const [currentQuestion, setCurrentQuestion] = useState(initialConversationState.currentQuestion);
  const [conversationStatus, setConversationStatus] = useState(initialConversationState.conversationStatus);
  const [report, setReport] = useState(initialConversationState.report);
  const [loading, setLoading] = useState(initialConversationState.loading);
  const [error, setError] = useState(initialConversationState.error);

  const setSessionId = useCallback((id) => {
    setSessionIdState(id);
    try {
      if (id) sessionStorage.setItem('abhera_session_id', id);
      else sessionStorage.removeItem('abhera_session_id');
    } catch (e) {}
  }, []);

  const setSubmissionId = useCallback((id) => {
    setSubmissionIdState((prevId) => {
      if (prevId !== id) {
        setReport(null);
      }
      return id;
    });
    try {
      if (id) sessionStorage.setItem('abhera_submission_id', id);
      else sessionStorage.removeItem('abhera_submission_id');
    } catch (e) {}
  }, []);

  const addMessage = useCallback((message) => {
    setMessages((prevMessages) => [...prevMessages, message]);
  }, []);

  const resetConversation = useCallback(() => {
    try {
      sessionStorage.removeItem('abhera_session_id');
      sessionStorage.removeItem('abhera_submission_id');
    } catch (e) {}
    setSessionIdState(null);
    setSubmissionIdState(null);
    setMessages(initialConversationState.messages);
    setAnswers(initialConversationState.answers);
    setPredictedLabels(initialConversationState.predictedLabels);
    setEntities(initialConversationState.entities);
    setCurrentQuestion(initialConversationState.currentQuestion);
    setConversationStatus(initialConversationState.conversationStatus);
    setReport(initialConversationState.report);
    setLoading(initialConversationState.loading);
    setError(initialConversationState.error);
  }, []);

  const value = {
    sessionId,
    setSessionId,
    submissionId,
    setSubmissionId,
    messages,
    setMessages,
    addMessage,
    answers,
    setAnswers,
    predictedLabels,
    setPredictedLabels,
    entities,
    setEntities,
    currentQuestion,
    setCurrentQuestion,
    conversationStatus,
    setConversationStatus,
    report,
    setReport,
    loading,
    setLoading,
    error,
    setError,
    resetConversation,
  };

  return (
    <ConversationContext.Provider value={value}>
      {children}
    </ConversationContext.Provider>
  );
};

export const useConversation = () => {
  const context = useContext(ConversationContext);
  if (!context) {
    throw new Error('useConversation must be used within a ConversationProvider');
  }
  return context;
};

export default ConversationContext;
