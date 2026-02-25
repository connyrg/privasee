/**
 * API Service
 * Client for backend API communication
 */

import axios from 'axios';

// Create axios instance with base configuration
const apiClient = axios.create({
  baseURL: '/api',
  // No global timeout — per-call timeouts are set below
  headers: {
    'Content-Type': 'application/json',
  },
});

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    const errorMessage = error.response?.data?.detail || error.message || 'An error occurred';
    console.error('API Error:', errorMessage);
    return Promise.reject(new Error(errorMessage));
  }
);

/**
 * Upload PDF file
 * @param {File} file - PDF file to upload
 * @returns {Promise<Object>} Upload response with session_id
 */
export const uploadPdf = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await apiClient.post('/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    timeout: 120000, // 2 min — upload + image extraction for large PDFs
  });

  return response.data;
};

/**
 * Process document to extract entities
 * @param {string} sessionId - Session identifier
 * @param {Array} fieldDefinitions - Array of field definitions
 * @returns {Promise<Object>} Process response with entities
 */
export const processDocument = async (sessionId, fieldDefinitions) => {
  const response = await apiClient.post('/process', {
    session_id: sessionId,
    field_definitions: fieldDefinitions,
  }, {
    timeout: 900000, // 15 min — OCR + Claude per page, up to ~30 pages
  });

  return response.data;
};

/**
 * Approve entities and generate masked PDF
 * @param {string} sessionId - Session identifier
 * @param {Array<string>} approvedEntityIds - Array of entity IDs to mask
 * @param {Array} updatedEntities - Optional array of entities with updated replacement text
 * @returns {Promise<Object>} Approval response with PDF URLs
 */
export const approveAndMask = async (sessionId, approvedEntityIds, updatedEntities = null) => {
  const response = await apiClient.post('/approve-and-mask', {
    session_id: sessionId,
    approved_entity_ids: approvedEntityIds,
    updated_entities: updatedEntities,
  }, {
    timeout: 300000, // 5 min — masking all pages + PDF assembly
  });

  return response.data;
};

/**
 * Get session information
 * @param {string} sessionId - Session identifier
 * @returns {Promise<Object>} Session information
 */
export const getSessionInfo = async (sessionId) => {
  const response = await apiClient.get(`/sessions/${sessionId}`);
  return response.data;
};

/**
 * Delete session and associated files
 * @param {string} sessionId - Session identifier
 * @returns {Promise<Object>} Deletion confirmation
 */
export const deleteSession = async (sessionId) => {
  const response = await apiClient.delete(`/sessions/${sessionId}`);
  return response.data;
};

/**
 * Health check
 * @returns {Promise<Object>} Health status
 */
export const healthCheck = async () => {
  const response = await apiClient.get('/health');
  return response.data;
};

/**
 * Get file URL
 * @param {string} folder - Folder name (uploads, temp_images, output)
 * @param {string} filename - File name
 * @returns {string} File URL
 */
export const getFileUrl = (folder, filename) => {
  return `/api/files/${folder}/${filename}`;
};

/**
 * Build file URL from API path
 * @param {string} path - API path (e.g., '/api/files/uploads/file.pdf')
 * @returns {string} Full URL
 */
export const buildFileUrl = (path) => {
  return path.startsWith('/api') ? path : `/api${path}`;
};

export default {
  uploadPdf,
  processDocument,
  approveAndMask,
  getSessionInfo,
  deleteSession,
  healthCheck,
  getFileUrl,
  buildFileUrl,
};
