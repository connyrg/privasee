/**
 * ConfigPanel Component
 * Dynamic table for defining de-identification rules
 */

import { useState } from 'react';
import { Plus, Trash2, HelpCircle } from 'lucide-react';

const STRATEGIES = [
  {
    value: 'Fake Data',
    label: 'Fake Data',
    description: 'Replace with realistic fake data (names, emails, etc.)',
  },
  {
    value: 'Black Out',
    label: 'Black Out',
    description: 'Completely redact with black rectangles',
  },
  {
    value: 'Entity Label',
    label: 'Entity Label',
    description: 'Replace with labels like Person_A, Email_1',
  },
];

const DEFAULT_FIELD = {
  id: '',
  name: '',
  description: '',
  strategy: 'Fake Data',
};

const ConfigPanel = ({ onFieldsChange, initialFields = [] }) => {
  const [fields, setFields] = useState(
    initialFields.length > 0
      ? initialFields.map((f, i) => ({ ...f, id: `field-${i}` }))
      : [{ ...DEFAULT_FIELD, id: 'field-0' }]
  );

  const handleAddField = () => {
    const newField = {
      ...DEFAULT_FIELD,
      id: `field-${Date.now()}`,
    };
    const updatedFields = [...fields, newField];
    setFields(updatedFields);
    notifyChange(updatedFields);
  };

  const handleRemoveField = (id) => {
    if (fields.length === 1) return; // Keep at least one field

    const updatedFields = fields.filter((f) => f.id !== id);
    setFields(updatedFields);
    notifyChange(updatedFields);
  };

  const handleFieldChange = (id, key, value) => {
    const updatedFields = fields.map((field) =>
      field.id === id ? { ...field, [key]: value } : field
    );
    setFields(updatedFields);
    notifyChange(updatedFields);
  };

  const notifyChange = (updatedFields) => {
    if (onFieldsChange) {
      // Filter out empty fields and format for API
      const validFields = updatedFields
        .filter((f) => f.name.trim() && f.description.trim())
        .map((f) => ({
          name: f.name.trim(),
          description: f.description.trim(),
          strategy: f.strategy,
        }));
      onFieldsChange(validFields);
    }
  };

  const isFormValid = () => {
    return fields.every((f) => f.name.trim() && f.description.trim());
  };

  return (
    <div className="w-full space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold text-gray-900">
            De-identification Rules
          </h3>
          <p className="text-sm text-gray-500 mt-1">
            Define which fields to identify and how to replace them
          </p>
        </div>
        <button
          onClick={handleAddField}
          className="flex items-center space-x-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
        >
          <Plus className="w-4 h-4" />
          <span>Add Field</span>
        </button>
      </div>

      {/* Table */}
      <div className="border border-gray-200 rounded-lg overflow-hidden">
        <table className="w-full">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider w-1/4">
                Field Name
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider w-2/5">
                Description
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider w-1/4">
                Replacement Strategy
              </th>
              <th className="px-4 py-3 w-16"></th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {fields.map((field, index) => (
              <tr key={field.id} className="hover:bg-gray-50">
                {/* Field Name */}
                <td className="px-4 py-3">
                  <input
                    type="text"
                    value={field.name}
                    onChange={(e) => handleFieldChange(field.id, 'name', e.target.value)}
                    placeholder="e.g., Full Name"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                  />
                </td>

                {/* Description */}
                <td className="px-4 py-3">
                  <input
                    type="text"
                    value={field.description}
                    onChange={(e) => handleFieldChange(field.id, 'description', e.target.value)}
                    placeholder="e.g., Person's full name"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                  />
                </td>

                {/* Strategy */}
                <td className="px-4 py-3">
                  <div className="relative group">
                    <select
                      value={field.strategy}
                      onChange={(e) => handleFieldChange(field.id, 'strategy', e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-transparent appearance-none cursor-pointer"
                    >
                      {STRATEGIES.map((strategy) => (
                        <option key={strategy.value} value={strategy.value}>
                          {strategy.label}
                        </option>
                      ))}
                    </select>
                    {/* Tooltip */}
                    <div className="absolute left-0 bottom-full mb-2 hidden group-hover:block w-64 p-2 bg-gray-900 text-white text-xs rounded shadow-lg z-10">
                      {STRATEGIES.find((s) => s.value === field.strategy)?.description}
                    </div>
                  </div>
                </td>

                {/* Delete Button */}
                <td className="px-4 py-3 text-center">
                  <button
                    onClick={() => handleRemoveField(field.id)}
                    disabled={fields.length === 1}
                    className={`
                      p-2 rounded-full transition-colors
                      ${fields.length === 1
                        ? 'text-gray-300 cursor-not-allowed'
                        : 'text-gray-500 hover:text-red-600 hover:bg-red-50'
                      }
                    `}
                    title={fields.length === 1 ? 'At least one field required' : 'Remove field'}
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Info Box */}
      <div className="flex items-start space-x-3 p-4 bg-blue-50 border border-blue-200 rounded-lg">
        <HelpCircle className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
        <div className="text-sm text-blue-800">
          <p className="font-medium mb-1">Strategy Guide:</p>
          <ul className="space-y-1 ml-4 list-disc">
            <li>
              <strong>Fake Data:</strong> Generates realistic replacements (e.g., John Doe → Emma Rodriguez)
            </li>
            <li>
              <strong>Black Out:</strong> Completely removes text with black boxes
            </li>
            <li>
              <strong>Entity Label:</strong> Uses generic labels (e.g., Person_A, Email_1)
            </li>
          </ul>
        </div>
      </div>

      {/* Validation Warning */}
      {!isFormValid() && (
        <div className="p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
          <p className="text-sm text-yellow-800">
            Please fill in all field names and descriptions before processing.
          </p>
        </div>
      )}
    </div>
  );
};

export default ConfigPanel;
