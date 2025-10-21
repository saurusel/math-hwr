import { useState, useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { createTrainingJob } from '../../api/training';
import { AdvancedSettings } from '../../components/Train/AdvancedSettings';
import { useToast } from '../../components/Toast/useToast';
import { Toast } from '../../components/Toast/Toast';
import type { ModelType, TrainingConfig } from '../../types/training';

export function NewTraining() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const { toasts, removeToast, error, success } = useToast();
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Support duplicated config from URL state
  const duplicatedConfig = searchParams.get('duplicate');

  const [formData, setFormData] = useState<TrainingConfig>({
    model_type: 'M1',
    run_name: `run_${Date.now()}`,
    dataset: {
      name: 'synth',
      train_manifest: 'data/synth/train/labels.jsonl',
      val_manifest: 'data/synth/val/labels.jsonl',
    },
    hyper: {
      batch_size: 64,
      epochs: 20,
      img_h: 64,
      img_w_max: 512,
      lr: 0.001,
      optimizer: 'adamw',
      scheduler: 'cosine',
      seed: 42,
      max_samples: 100,  // NEW: Default 100 samples
    },
    augment: {
      invert: true,
      random_pad: false,
    },
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);

    try {
      const result = await createTrainingJob(formData);
      success(`Training job created: ${result.job_id}`);
      setTimeout(() => {
        navigate(`/train/monitor?job=${result.job_id}`);
      }, 1000);
    } catch (err) {
      error(err instanceof Error ? err.message : 'Failed to create training job');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-3xl font-bold text-slate-800 mb-6">New Training Run</h1>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Basic Settings */}
        <div className="bg-white rounded-xl shadow p-6 space-y-4">
          <h2 className="text-xl font-semibold text-slate-700">Basic Settings</h2>

          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Model Type</label>
            <select
              className="w-full px-3 py-2 border border-slate-300 rounded-lg"
              value={formData.model_type}
              onChange={(e) =>
                setFormData({ ...formData, model_type: e.target.value as ModelType })
              }
            >
              <option value="M1">M1 (CRNN-CTC) - Fast sequence model</option>
              <option value="M2">M2 (Segmentation + MLP) - Classical OCR with character segmentation</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Run Name</label>
            <input
              type="text"
              className="w-full px-3 py-2 border border-slate-300 rounded-lg"
              value={formData.run_name}
              onChange={(e) => setFormData({ ...formData, run_name: e.target.value })}
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Batch Size</label>
              <input
                type="number"
                className="w-full px-3 py-2 border border-slate-300 rounded-lg"
                value={formData.hyper.batch_size}
                onChange={(e) =>
                  setFormData({
                    ...formData,
                    hyper: { ...formData.hyper, batch_size: parseInt(e.target.value) },
                  })
                }
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Epochs</label>
              <input
                type="number"
                className="w-full px-3 py-2 border border-slate-300 rounded-lg"
                value={formData.hyper.epochs}
                onChange={(e) =>
                  setFormData({
                    ...formData,
                    hyper: { ...formData.hyper, epochs: parseInt(e.target.value) },
                  })
                }
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Learning Rate</label>
              <input
                type="number"
                step="0.0001"
                className="w-full px-3 py-2 border border-slate-300 rounded-lg"
                value={formData.hyper.lr}
                onChange={(e) =>
                  setFormData({
                    ...formData,
                    hyper: { ...formData.hyper, lr: parseFloat(e.target.value) },
                  })
                }
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Seed</label>
              <input
                type="number"
                className="w-full px-3 py-2 border border-slate-300 rounded-lg"
                value={formData.hyper.seed}
                onChange={(e) =>
                  setFormData({
                    ...formData,
                    hyper: { ...formData.hyper, seed: parseInt(e.target.value) },
                  })
                }
              />
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">
              Max Samples (Dataset Size Limit)
            </label>
            <input
              type="number"
              className="w-full px-3 py-2 border border-slate-300 rounded-lg"
              value={formData.hyper.max_samples || ''}
              placeholder="Leave empty for full dataset"
              onChange={(e) =>
                setFormData({
                  ...formData,
                  hyper: {
                    ...formData.hyper,
                    max_samples: e.target.value ? parseInt(e.target.value) : undefined,
                  },
                })
              }
            />
            <p className="text-xs text-slate-500 mt-1">
              Limit training to N samples (80% train, 20% val). Default: 100. Leave empty for full dataset (~96K samples).
            </p>
          </div>

          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="invert"
              checked={formData.augment.invert}
              onChange={(e) =>
                setFormData({
                  ...formData,
                  augment: { ...formData.augment, invert: e.target.checked },
                })
              }
            />
            <label htmlFor="invert" className="text-sm text-slate-700">
              Invert Colors
            </label>
          </div>
        </div>

        {/* Advanced Settings */}
        <AdvancedSettings
          modelType={formData.model_type}
          config={formData}
          onChange={setFormData}
        />

        <div className="flex gap-3">
          <button
            type="button"
            className="px-6 py-2 border border-slate-300 rounded-lg hover:bg-slate-50"
            onClick={() => navigate('/train/experiments')}
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={isSubmitting}
            className={`px-6 py-2 rounded-lg font-medium ${
              isSubmitting
                ? 'bg-slate-400 cursor-not-allowed'
                : 'bg-emerald-600 hover:bg-emerald-700 text-white'
            }`}
          >
            {isSubmitting ? 'Creating...' : 'Start Training'}
          </button>
        </div>
      </form>

      {/* Toast notifications */}
      <div className="fixed top-4 right-4 space-y-2 z-50">
        {toasts.map((toast) => (
          <Toast
            key={toast.id}
            message={toast.message}
            type={toast.type}
            onClose={() => removeToast(toast.id)}
          />
        ))}
      </div>
    </div>
  );
}
