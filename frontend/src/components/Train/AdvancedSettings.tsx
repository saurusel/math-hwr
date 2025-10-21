import { useState } from 'react';
import type { ModelType, TrainingConfig } from '../../types/training';

interface AdvancedSettingsProps {
  modelType: ModelType;
  config: TrainingConfig;
  onChange: (config: TrainingConfig) => void;
}

export function AdvancedSettings({ modelType, config, onChange }: AdvancedSettingsProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  const updateM1Advanced = (key: string, value: number) => {
    onChange({
      ...config,
      hyper: {
        ...config.hyper,
        advanced: {
          ...config.hyper.advanced,
          m1: {
            ...config.hyper.advanced?.m1,
            [key]: value,
          },
        },
      },
    });
  };

  const updateM2Advanced = (key: string, value: number) => {
    onChange({
      ...config,
      hyper: {
        ...config.hyper,
        advanced: {
          ...config.hyper.advanced,
          m2: {
            ...config.hyper.advanced?.m2,
            [key]: value,
          },
        },
      },
    });
  };


  return (
    <div className="bg-white rounded-xl shadow p-6">
      <button
        type="button"
        className="flex items-center justify-between w-full text-left"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <h2 className="text-xl font-semibold text-slate-700">Advanced Settings</h2>
        <span className="text-slate-400">{isExpanded ? '▼' : '▶'}</span>
      </button>

      {isExpanded && (
        <div className="mt-4 space-y-4">
          {modelType === 'M1' && (
            <div className="border-l-4 border-emerald-500 pl-4 space-y-3">
              <h3 className="font-medium text-slate-700">M1 (CRNN-CTC) Settings</h3>
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">
                  Weight Decay
                </label>
                <input
                  type="number"
                  step="0.001"
                  className="w-full px-3 py-2 border border-slate-300 rounded-lg"
                  value={config.hyper.advanced?.m1?.weight_decay || 0.01}
                  onChange={(e) => updateM1Advanced('weight_decay', parseFloat(e.target.value))}
                />
                <p className="text-xs text-slate-500 mt-1">
                  L2 regularization strength (typically 0.001-0.1)
                </p>
              </div>
            </div>
          )}

          {modelType === 'M2' && (
            <div className="border-l-4 border-blue-500 pl-4 space-y-3">
              <h3 className="font-medium text-slate-700">M2 (Segmentation + MLP) Settings</h3>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1">
                    Dropout Rate
                  </label>
                  <input
                    type="number"
                    step="0.05"
                    min="0"
                    max="0.9"
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg"
                    value={config.hyper.advanced?.m2?.dropout || 0.3}
                    onChange={(e) => updateM2Advanced('dropout', parseFloat(e.target.value))}
                  />
                  <p className="text-xs text-slate-500 mt-1">MLP dropout for regularization</p>
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1">
                    Min Segment Area
                  </label>
                  <input
                    type="number"
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg"
                    value={config.hyper.advanced?.m2?.min_seg_area || 20}
                    onChange={(e) => updateM2Advanced('min_seg_area', parseInt(e.target.value))}
                  />
                  <p className="text-xs text-slate-500 mt-1">Min character area (pixels)</p>
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1">
                    Max Segment Area
                  </label>
                  <input
                    type="number"
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg"
                    value={config.hyper.advanced?.m2?.max_seg_area || 4000}
                    onChange={(e) => updateM2Advanced('max_seg_area', parseInt(e.target.value))}
                  />
                  <p className="text-xs text-slate-500 mt-1">Max character area (pixels)</p>
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1">
                    Target Character Size
                  </label>
                  <input
                    type="number"
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg"
                    value={config.hyper.advanced?.m2?.target_char_size || 32}
                    onChange={(e) => updateM2Advanced('target_char_size', parseInt(e.target.value))}
                  />
                  <p className="text-xs text-slate-500 mt-1">Character normalization size (32x32)</p>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
