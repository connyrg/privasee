/**
 * PrivaSee App Component
 * Main application orchestrating the de-identification workflow
 */

import { useState } from 'react';
import { Shield, Loader2, AlertCircle } from 'lucide-react';
import UploadZone from './components/UploadZone';
import ConfigPanel from './components/ConfigPanel';
import ReviewTable from './components/ReviewTable';
import ComparisonView from './components/ComparisonView';
import { uploadPdf, processDocument, approveAndMask } from './services/api';

const STEPS = {
  CONFIGURE: 'configure',
  REVIEW: 'review',
  COMPARE: 'compare',
};

function App() {
  // State
  const [step, setStep] = useState(STEPS.CONFIGURE);
  const [sessionId, setSessionId] = useState(null);
  const [filename, setFilename] = useState(null);
  const [pageCount, setPageCount] = useState(1);
  const [fieldDefinitions, setFieldDefinitions] = useState([]);
  const [entities, setEntities] = useState([]);
  const [originalPdfUrl, setOriginalPdfUrl] = useState(null);
  const [maskedPdfUrl, setMaskedPdfUrl] = useState(null);
  const [entitiesMasked, setEntitiesMasked] = useState(0);

  // Loading and error states
  const [isUploading, setIsUploading] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState(null);

  // Handlers
  const handleUploadSuccess = async (file) => {
    setError(null);
    setIsUploading(true);

    try {
      const response = await uploadPdf(file);
      setSessionId(response.session_id);
      setFilename(response.filename);
      setPageCount(response.page_count ?? 1);
      setOriginalPdfUrl(`/api/files/uploads/${response.session_id}.pdf`);
      console.log('Upload successful:', response);
    } catch (err) {
      setError(`Upload failed: ${err.message}`);
      console.error('Upload error:', err);
    } finally {
      setIsUploading(false);
    }
  };

  const handleFieldsChange = (fields) => {
    setFieldDefinitions(fields);
  };

  const handleProcessDocument = async () => {
    if (!sessionId || fieldDefinitions.length === 0) {
      setError('Please upload a document and define at least one field');
      return;
    }

    setError(null);
    setIsProcessing(true);

    try {
      const response = await processDocument(sessionId, fieldDefinitions);
      setEntities(response.entities);
      setStep(STEPS.REVIEW);
      console.log('Processing successful:', response);
    } catch (err) {
      setError(`Processing failed: ${err.message}`);
      console.error('Processing error:', err);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleEntitiesChange = (updatedEntities) => {
    setEntities(updatedEntities);
  };

  const handleApprove = async (approvedEntities) => {
    if (approvedEntities.length === 0) {
      setError('Please select at least one entity to mask');
      return;
    }

    setError(null);
    setIsGenerating(true);

    try {
      const approvedIds = approvedEntities.map((e) => e.id);
      const response = await approveAndMask(sessionId, approvedIds, approvedEntities);

      setMaskedPdfUrl(response.masked_pdf_url);
      setEntitiesMasked(response.entities_masked);
      setStep(STEPS.COMPARE);
      console.log('Masking successful:', response);
    } catch (err) {
      setError(`Masking failed: ${err.message}`);
      console.error('Masking error:', err);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleReset = () => {
    setStep(STEPS.CONFIGURE);
    setSessionId(null);
    setFilename(null);
    setPageCount(1);
    setFieldDefinitions([]);
    setEntities([]);
    setOriginalPdfUrl(null);
    setMaskedPdfUrl(null);
    setEntitiesMasked(0);
    setError(null);
  };

  const canProcess = sessionId && fieldDefinitions.length > 0;

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-primary-600 rounded-lg">
              <Shield className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-900">PrivaSee</h1>
              <p className="text-sm text-gray-500">Intelligent Document De-identification</p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Progress Steps */}
        <div className="mb-8">
          <div className="flex items-center justify-center space-x-4">
            {/* Step 1: Configure */}
            <div className="flex items-center">
              <div
                className={`
                  flex items-center justify-center w-10 h-10 rounded-full font-semibold
                  ${step === STEPS.CONFIGURE
                    ? 'bg-primary-600 text-white'
                    : 'bg-green-500 text-white'
                  }
                `}
              >
                1
              </div>
              <span className="ml-2 text-sm font-medium text-gray-700">Configure</span>
            </div>

            <div className="w-16 h-1 bg-gray-300 rounded"></div>

            {/* Step 2: Review */}
            <div className="flex items-center">
              <div
                className={`
                  flex items-center justify-center w-10 h-10 rounded-full font-semibold
                  ${step === STEPS.REVIEW
                    ? 'bg-primary-600 text-white'
                    : step === STEPS.COMPARE
                    ? 'bg-green-500 text-white'
                    : 'bg-gray-300 text-gray-600'
                  }
                `}
              >
                2
              </div>
              <span className="ml-2 text-sm font-medium text-gray-700">Review</span>
            </div>

            <div className="w-16 h-1 bg-gray-300 rounded"></div>

            {/* Step 3: Compare */}
            <div className="flex items-center">
              <div
                className={`
                  flex items-center justify-center w-10 h-10 rounded-full font-semibold
                  ${step === STEPS.COMPARE
                    ? 'bg-primary-600 text-white'
                    : 'bg-gray-300 text-gray-600'
                  }
                `}
              >
                3
              </div>
              <span className="ml-2 text-sm font-medium text-gray-700">Compare</span>
            </div>
          </div>
        </div>

        {/* Error Message */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg flex items-start space-x-3">
            <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
            <div className="flex-1">
              <p className="text-sm font-medium text-red-900">Error</p>
              <p className="text-sm text-red-700 mt-1">{error}</p>
            </div>
            <button
              onClick={() => setError(null)}
              className="text-red-600 hover:text-red-800"
            >
              ×
            </button>
          </div>
        )}

        {/* Content Area */}
        <div className="bg-white rounded-xl shadow-lg p-8">
          {/* Step 1: Configure */}
          {step === STEPS.CONFIGURE && (
            <div className="space-y-8">
              <UploadZone onUploadSuccess={handleUploadSuccess} disabled={isUploading} />

              {sessionId && (
                <>
                  {pageCount > 1 && (
                    <div className="px-4 py-2 bg-blue-50 border border-blue-200 rounded-lg text-sm text-blue-800">
                      Multi-page PDF detected: <strong>{pageCount} pages</strong> will all be processed.
                    </div>
                  )}
                  <div className="border-t border-gray-200 pt-8">
                    <ConfigPanel onFieldsChange={handleFieldsChange} />
                  </div>

                  <div className="flex justify-end pt-4">
                    <button
                      onClick={handleProcessDocument}
                      disabled={!canProcess || isProcessing}
                      className={`
                        flex items-center space-x-2 px-6 py-3 rounded-lg font-medium
                        transition-all duration-200
                        ${!canProcess || isProcessing
                          ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                          : 'bg-primary-600 text-white hover:bg-primary-700'
                        }
                      `}
                    >
                      {isProcessing && <Loader2 className="w-5 h-5 animate-spin" />}
                      <span>{isProcessing ? 'Processing...' : 'Process Document'}</span>
                    </button>
                  </div>
                </>
              )}
            </div>
          )}

          {/* Step 2: Review */}
          {step === STEPS.REVIEW && (
            <div className="space-y-6">
              {isGenerating && (
                <div className="flex items-center justify-center space-x-3 p-8">
                  <Loader2 className="w-8 h-8 animate-spin text-primary-600" />
                  <p className="text-lg font-medium text-gray-700">Generating masked PDF...</p>
                </div>
              )}

              {!isGenerating && (
                <>
                  <ReviewTable
                    entities={entities}
                    onEntitiesChange={handleEntitiesChange}
                    onApprove={handleApprove}
                  />

                  <div className="flex justify-between pt-4">
                    <button
                      onClick={() => setStep(STEPS.CONFIGURE)}
                      className="px-6 py-3 text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
                    >
                      Back to Configure
                    </button>
                  </div>
                </>
              )}
            </div>
          )}

          {/* Step 3: Compare */}
          {step === STEPS.COMPARE && (
            <ComparisonView
              originalPdfUrl={originalPdfUrl}
              maskedPdfUrl={maskedPdfUrl}
              entitiesMasked={entitiesMasked}
              onReset={handleReset}
            />
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="mt-12 pb-8 text-center text-sm text-gray-600">
        <p>
          PrivaSee - Powered by Azure Document Intelligence and Claude AI
        </p>
        <p className="mt-2">
          All processing happens locally. Your privacy is our priority.
        </p>
      </footer>
    </div>
  );
}

export default App;
