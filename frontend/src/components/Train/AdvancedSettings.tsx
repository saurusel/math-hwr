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

  const updateM3Advanced = (key: string, value: number) => {
    onChange({
      ...config,
      hyper: {
        ...config.hyper,
        advanced: {
          ...config.hyper.advanced,
          m3: {
            ...config.hyper.advanced?.m3,
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
              <h3 className="font-medium text-slate-700">M2 (Attention Seq2Seq) Settings</h3>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1">
                    Model Dimension (d_model)
                  </label>
                  <input
                    type="number"
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg"
                    value={config.hyper.advanced?.m2?.d_model || 256}
                    onChange={(e) => updateM2Advanced('d_model', parseInt(e.target.value))}
                  />
                  <p className="text-xs text-slate-500 mt-1">Hidden dimension size</p>
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1">
                    Number of Heads
                  </label>
                  <input
                    type="number"
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg"
                    value={config.hyper.advanced?.m2?.n_heads || 4}
                    onChange={(e) => updateM2Advanced('n_heads', parseInt(e.target.value))}
                  />
                  <p className="text-xs text-slate-500 mt-1">Attention heads (must divide d_model)</p>
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1">
                    Encoder Layers
                  </label>
                  <input
                    type="number"
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg"
                    value={config.hyper.advanced?.m2?.n_layers_enc || 4}
                    onChange={(e) => updateM2Advanced('n_layers_enc', parseInt(e.target.value))}
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1">
                    Decoder Layers
                  </label>
                  <input
                    type="number"
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg"
                    value={config.hyper.advanced?.m2?.n_layers_dec || 4}
                    onChange={(e) => updateM2Advanced('n_layers_dec', parseInt(e.target.value))}
                  />
                </div>

                <div className="col-span-2">
                  <label className="block text-sm font-medium text-slate-700 mb-1">
                    Teacher Forcing Ratio
                  </label>
                  <input
                    type="number"
                    step="0.1"
                    min="0"
                    max="1"
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg"
                    value={config.hyper.advanced?.m2?.teacher_forcing || 0.5}
                    onChange={(e) => updateM2Advanced('teacher_forcing', parseFloat(e.target.value))}
                  />
                  <p className="text-xs text-slate-500 mt-1">
                    Probability of using ground truth during training (0-1)
                  </p>
                </div>
              </div>
            </div>
          )}

          {modelType === 'M3' && (
            <div className="border-l-4 border-purple-500 pl-4 space-y-3">
              <h3 className="font-medium text-slate-700">M3 (Symbol Classifier) Settings</h3>
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
                  value={config.hyper.advanced?.m3?.dropout || 0.2}
                  onChange={(e) => updateM3Advanced('dropout', parseFloat(e.target.value))}
                />
                <p className="text-xs text-slate-500 mt-1">
                  Dropout probability for regularization (0-0.9)
                </p>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
