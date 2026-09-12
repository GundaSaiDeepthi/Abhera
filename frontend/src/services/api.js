import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * Health Check API
 * GET /health
 */
export const checkHealth = async () => {
  const response = await api.get('/health');
  return response.data;
};

/**
 * Start a new conversation session
 * POST /api/session/start
 */
export const startSession = async () => {
  const response = await api.post('/api/session/start', {});
  return response.data;
};

/**
 * Get existing session details by session_id
 * GET /api/session/{session_id}
 * @param {string} sessionId
 */
export const getSession = async (sessionId) => {
  const response = await api.get(`/api/session/${sessionId}`);
  return response.data;
};

/**
 * Send user chat message to main chat API endpoint
 * POST /api/chat/message
 * @param {string} sessionId
 * @param {string} message
 */
export const sendChatMessage = async (sessionId, message) => {
  const response = await api.post('/api/chat/message', {
    session_id: sessionId,
    message: message,
  });
  return response.data;
};

/**
 * Generate Final Incident Report
 * POST /api/report/generate
 * @param {Object} data - Contains submission_id, predicted_categories, extracted_entities, Q&A responses, etc.
 */
export const generateReport = async (data) => {
  const response = await api.post('/api/report/generate', data);
  return response.data;
};

/**
 * Retrieve Existing Report by Submission ID
 * GET /api/report/{submission_id}
 * @param {string} submissionId
 */
export const getReport = async (submissionId) => {
  const response = await api.get(`/api/report/${submissionId}`);
  return response.data;
};

export default api;
