/**
 * ComparisonView Component
 * Side-by-side comparison of original and masked PDFs
 */

import { useState } from 'react';
import { Download, FileText, Shield, Eye, EyeOff } from 'lucide-react';

const ComparisonView = ({ originalPdfUrl, maskedPdfUrl, entitiesMasked, onReset }) => {
  const [showOriginal, setShowOriginal] = useState(true);
  const [showMasked, setShowMasked] = useState(true);

  const handleDownload = (url, filename) => {
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="w-full space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold text-gray-900">
            Document Comparison
          </h3>
          <p className="text-sm text-gray-500 mt-1">
            {entitiesMasked} {entitiesMasked === 1 ? 'entity' : 'entities'} masked successfully
          </p>
        </div>
        <button
          onClick={onReset}
          className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
        >
          Process New Document
        </button>
      </div>

      {/* Success Banner */}
      <div className="p-4 bg-green-50 border border-green-200 rounded-lg flex items-center space-x-3">
        <Shield className="w-6 h-6 text-green-600 flex-shrink-0" />
        <div>
          <p className="text-sm font-medium text-green-900">
            Document de-identified successfully!
          </p>
          <p className="text-sm text-green-700 mt-1">
            Your sensitive information has been masked. Download the protected version below.
          </p>
        </div>
      </div>

      {/* View Toggle */}
      <div className="flex items-center justify-center space-x-4 p-4 bg-gray-50 rounded-lg">
        <label className="flex items-center space-x-2 cursor-pointer">
          <input
            type="checkbox"
            checked={showOriginal}
            onChange={(e) => setShowOriginal(e.target.checked)}
            className="w-4 h-4 text-primary-600 border-gray-300 rounded focus:ring-primary-500"
          />
          <span className="text-sm font-medium text-gray-700">Show Original</span>
        </label>
        <label className="flex items-center space-x-2 cursor-pointer">
          <input
            type="checkbox"
            checked={showMasked}
            onChange={(e) => setShowMasked(e.target.checked)}
            className="w-4 h-4 text-primary-600 border-gray-300 rounded focus:ring-primary-500"
          />
          <span className="text-sm font-medium text-gray-700">Show Masked</span>
        </label>
      </div>

      {/* PDF Comparison Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Original PDF */}
        {showOriginal && (
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <FileText className="w-5 h-5 text-gray-600" />
                <h4 className="text-sm font-semibold text-gray-900">Original Document</h4>
              </div>
              <button
                onClick={() => handleDownload(originalPdfUrl, 'original.pdf')}
                className="flex items-center space-x-2 px-3 py-1.5 text-sm text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
              >
                <Download className="w-4 h-4" />
                <span>Download</span>
              </button>
            </div>
            <div className="border-2 border-gray-300 rounded-lg overflow-hidden bg-gray-50">
              <iframe
                src={originalPdfUrl}
                className="w-full h-[600px]"
                title="Original PDF"
              />
            </div>
          </div>
        )}

        {/* Masked PDF */}
        {showMasked && (
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Shield className="w-5 h-5 text-green-600" />
                <h4 className="text-sm font-semibold text-gray-900">Masked Document</h4>
              </div>
              <button
                onClick={() => handleDownload(maskedPdfUrl, 'masked.pdf')}
                className="flex items-center space-x-2 px-3 py-1.5 text-sm text-white bg-primary-600 rounded-lg hover:bg-primary-700 transition-colors"
              >
                <Download className="w-4 h-4" />
                <span>Download</span>
              </button>
            </div>
            <div className="border-2 border-green-400 rounded-lg overflow-hidden bg-gray-50">
              <iframe
                src={maskedPdfUrl}
                className="w-full h-[600px]"
                title="Masked PDF"
              />
            </div>
          </div>
        )}
      </div>

      {/* Info Card */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="p-4 bg-white border border-gray-200 rounded-lg">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-blue-100 rounded-lg">
              <FileText className="w-5 h-5 text-blue-600" />
            </div>
            <div>
              <p className="text-xs text-gray-500 uppercase tracking-wide">Original</p>
              <p className="text-sm font-semibold text-gray-900">Unprotected</p>
            </div>
          </div>
        </div>

        <div className="p-4 bg-white border border-gray-200 rounded-lg">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-green-100 rounded-lg">
              <Shield className="w-5 h-5 text-green-600" />
            </div>
            <div>
              <p className="text-xs text-gray-500 uppercase tracking-wide">Entities Masked</p>
              <p className="text-sm font-semibold text-gray-900">{entitiesMasked}</p>
            </div>
          </div>
        </div>

        <div className="p-4 bg-white border border-gray-200 rounded-lg">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-purple-100 rounded-lg">
              <Eye className="w-5 h-5 text-purple-600" />
            </div>
            <div>
              <p className="text-xs text-gray-500 uppercase tracking-wide">Processing</p>
              <p className="text-sm font-semibold text-gray-900">Local Only</p>
            </div>
          </div>
        </div>
      </div>

      {/* Privacy Notice */}
      <div className="p-4 bg-purple-50 border border-purple-200 rounded-lg">
        <div className="flex items-start space-x-3">
          <Shield className="w-5 h-5 text-purple-600 flex-shrink-0 mt-0.5" />
          <div className="text-sm text-purple-800">
            <p className="font-medium mb-1">Privacy First</p>
            <p>
              All processing happens locally on your machine. Your documents are never uploaded to
              external servers (except for AI processing via secure APIs). Download and delete your
              files when done.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ComparisonView;
