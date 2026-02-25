/**
 * ReviewTable Component
 * Entity review and approval interface
 */

import { useState, useEffect } from 'react';
import { Check, X, Edit2, Save } from 'lucide-react';

const ReviewTable = ({ entities, onEntitiesChange, onApprove }) => {
  const [localEntities, setLocalEntities] = useState([]);
  const [editingId, setEditingId] = useState(null);
  const [editValue, setEditValue] = useState('');

  useEffect(() => {
    const sorted = [...entities].sort((a, b) => {
      const pageDiff = (a.page_number ?? 1) - (b.page_number ?? 1);
      if (pageDiff !== 0) return pageDiff;
      return (a.entity_type ?? '').localeCompare(b.entity_type ?? '');
    });
    setLocalEntities(sorted);
  }, [entities]);

  const handleToggleApproval = (id) => {
    const updated = localEntities.map((entity) =>
      entity.id === id ? { ...entity, approved: !entity.approved } : entity
    );
    setLocalEntities(updated);
    if (onEntitiesChange) {
      onEntitiesChange(updated);
    }
  };

  const handleToggleAll = () => {
    const allApproved = localEntities.every((e) => e.approved);
    const updated = localEntities.map((entity) => ({
      ...entity,
      approved: !allApproved,
    }));
    setLocalEntities(updated);
    if (onEntitiesChange) {
      onEntitiesChange(updated);
    }
  };

  const handleStartEdit = (entity) => {
    setEditingId(entity.id);
    setEditValue(entity.replacement_text);
  };

  const handleSaveEdit = (id) => {
    const updated = localEntities.map((entity) =>
      entity.id === id ? { ...entity, replacement_text: editValue } : entity
    );
    setLocalEntities(updated);
    if (onEntitiesChange) {
      onEntitiesChange(updated);
    }
    setEditingId(null);
    setEditValue('');
  };

  const handleCancelEdit = () => {
    setEditingId(null);
    setEditValue('');
  };

  const handleApproveClick = () => {
    if (onApprove) {
      const approvedEntities = localEntities.filter((e) => e.approved);
      onApprove(approvedEntities);
    }
  };

  const approvedCount = localEntities.filter((e) => e.approved).length;
  const allApproved = localEntities.length > 0 && localEntities.every((e) => e.approved);

  return (
    <div className="w-full space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold text-gray-900">
            Review Identified Entities
          </h3>
          <p className="text-sm text-gray-500 mt-1">
            {approvedCount} of {localEntities.length} entities selected for masking
          </p>
        </div>
        <div className="flex items-center space-x-3">
          <button
            onClick={handleToggleAll}
            className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
          >
            {allApproved ? 'Deselect All' : 'Select All'}
          </button>
          <button
            onClick={handleApproveClick}
            disabled={approvedCount === 0}
            className={`
              px-6 py-2 text-sm font-medium rounded-lg transition-colors
              ${approvedCount === 0
                ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                : 'bg-primary-600 text-white hover:bg-primary-700'
              }
            `}
          >
            Generate Masked PDF
          </button>
        </div>
      </div>

      {/* Table */}
      <div className="border border-gray-200 rounded-lg overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-4 py-3 text-center w-16">
                  <input
                    type="checkbox"
                    checked={allApproved}
                    onChange={handleToggleAll}
                    className="w-4 h-4 text-primary-600 border-gray-300 rounded focus:ring-primary-500 cursor-pointer"
                  />
                </th>
                <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider w-16">
                  Page
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Entity Type
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Original Text
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Replacement Text
                </th>
                <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Confidence
                </th>
                <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider w-24">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {localEntities.length === 0 ? (
                <tr>
                  <td colSpan="7" className="px-4 py-8 text-center text-gray-500">
                    No entities found
                  </td>
                </tr>
              ) : (
                localEntities.map((entity) => (
                  <tr
                    key={entity.id}
                    className={`
                      transition-colors
                      ${entity.approved ? 'bg-green-50 hover:bg-green-100' : 'hover:bg-gray-50'}
                    `}
                  >
                    {/* Checkbox */}
                    <td className="px-4 py-3 text-center">
                      <input
                        type="checkbox"
                        checked={entity.approved}
                        onChange={() => handleToggleApproval(entity.id)}
                        className="w-4 h-4 text-primary-600 border-gray-300 rounded focus:ring-primary-500 cursor-pointer"
                      />
                    </td>

                    {/* Page Number */}
                    <td className="px-4 py-3 text-center">
                      <span className="text-sm text-gray-600 font-medium">
                        {entity.page_number ?? 1}
                      </span>
                    </td>

                    {/* Entity Type */}
                    <td className="px-4 py-3">
                      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                        {entity.entity_type}
                      </span>
                    </td>

                    {/* Original Text */}
                    <td className="px-4 py-3">
                      <span className="text-sm text-gray-900 font-medium">
                        {entity.original_text}
                      </span>
                    </td>

                    {/* Replacement Text */}
                    <td className="px-4 py-3">
                      {editingId === entity.id ? (
                        <input
                          type="text"
                          value={editValue}
                          onChange={(e) => setEditValue(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') handleSaveEdit(entity.id);
                            if (e.key === 'Escape') handleCancelEdit();
                          }}
                          className="w-full px-2 py-1 text-sm border border-primary-500 rounded focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                          autoFocus
                        />
                      ) : (
                        <span className="text-sm text-primary-700 font-medium">
                          {entity.replacement_text}
                        </span>
                      )}
                    </td>

                    {/* Confidence */}
                    <td className="px-4 py-3 text-center">
                      <span
                        className={`
                          inline-flex items-center px-2 py-1 rounded text-xs font-medium
                          ${entity.confidence >= 0.9
                            ? 'bg-green-100 text-green-800'
                            : entity.confidence >= 0.7
                            ? 'bg-yellow-100 text-yellow-800'
                            : 'bg-red-100 text-red-800'
                          }
                        `}
                      >
                        {(entity.confidence * 100).toFixed(0)}%
                      </span>
                    </td>

                    {/* Actions */}
                    <td className="px-4 py-3 text-center">
                      {editingId === entity.id ? (
                        <div className="flex items-center justify-center space-x-1">
                          <button
                            onClick={() => handleSaveEdit(entity.id)}
                            className="p-1 text-green-600 hover:bg-green-100 rounded transition-colors"
                            title="Save"
                          >
                            <Save className="w-4 h-4" />
                          </button>
                          <button
                            onClick={handleCancelEdit}
                            className="p-1 text-gray-600 hover:bg-gray-100 rounded transition-colors"
                            title="Cancel"
                          >
                            <X className="w-4 h-4" />
                          </button>
                        </div>
                      ) : (
                        <button
                          onClick={() => handleStartEdit(entity)}
                          className="p-1 text-gray-600 hover:bg-gray-100 rounded transition-colors"
                          title="Edit replacement text"
                        >
                          <Edit2 className="w-4 h-4" />
                        </button>
                      )}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Info */}
      {localEntities.length > 0 && (
        <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
          <p className="text-sm text-blue-800">
            <strong>Tip:</strong> You can edit replacement text by clicking the edit icon.
            Only checked entities will be masked in the output PDF.
          </p>
        </div>
      )}
    </div>
  );
};

export default ReviewTable;
