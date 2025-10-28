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
    model_type: 'M2',  // UPDATED: M2 recommended for new dataset
    run_name: `run_${Date.now()}`,
    dataset: {
      name: 'synth_small_1k',  // UPDATED: Small dataset to prevent overfitting
      train_manifest: 'data/synth_small_1k/train/labels.jsonl',
      val_manifest: 'data/synth_small_1k/val/labels.jsonl',
    },
    hyper: {
      batch_size: 32,  // UPDATED: Smaller batches for small dataset (800 samples)
      epochs: 50,      // UPDATED: More epochs for small dataset
      img_h: 64,
      img_w_max: 256,  // UPDATED: 256 for 4x uniform scaling (no distortion)
      lr: 0.001,
      optimizer: 'adamw',
      scheduler: 'cosine',
      seed: 42,
      max_samples: undefined,  // UPDATED: Use full dataset by default
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
      <h1 className="text-3xl font-bold text-slate-800 mb-6">Новое обучение</h1>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Basic Settings */}
        <div className="bg-white rounded-xl shadow p-6 space-y-4">
          <h2 className="text-xl font-semibold text-slate-700">Основные настройки</h2>

          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Тип модели</label>
            <select
              className="w-full px-3 py-2 border border-slate-300 rounded-lg"
              value={formData.model_type}
              onChange={(e) =>
                setFormData({ ...formData, model_type: e.target.value as ModelType })
              }
            >
              <option value="M1">M1 (CRNN-CTC) - Быстрая последовательная модель</option>
              <option value="M2">M2 (Сегментация + MLP) - Классическое OCR с посимвольной сегментацией</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Название запуска</label>
            <input
              type="text"
              className="w-full px-3 py-2 border border-slate-300 rounded-lg"
              value={formData.run_name}
              onChange={(e) => setFormData({ ...formData, run_name: e.target.value })}
            />
          </div>

          {/* Dataset Paths */}
          <div className="border-t pt-4 space-y-3">
            <h3 className="text-sm font-semibold text-slate-700">Пути к датасету</h3>

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">
                Манифест обучающей выборки
              </label>
              <input
                type="text"
                className="w-full px-3 py-2 border border-slate-300 rounded-lg font-mono text-sm"
                value={formData.dataset.train_manifest}
                onChange={(e) =>
                  setFormData({
                    ...formData,
                    dataset: { ...formData.dataset, train_manifest: e.target.value },
                  })
                }
                placeholder="data/synth_v3_large/train/labels.jsonl"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">
                Манифест валидационной выборки
              </label>
              <input
                type="text"
                className="w-full px-3 py-2 border border-slate-300 rounded-lg font-mono text-sm"
                value={formData.dataset.val_manifest}
                onChange={(e) =>
                  setFormData({
                    ...formData,
                    dataset: { ...formData.dataset, val_manifest: e.target.value },
                  })
                }
                placeholder="data/synth_v3_large/val/labels.jsonl"
              />
            </div>

            <div className="bg-emerald-50 border border-emerald-200 rounded-lg p-3">
              <p className="text-xs text-emerald-800">
                <strong>✅ Рекомендуется:</strong><br/>
                • <code className="bg-emerald-100 px-1 rounded">data/synth_small_1k</code> - <strong>Лучше для обобщения!</strong> (1K примеров, предотвращает переобучение)<br/>
                <br/>
                <strong>Другие датасеты:</strong><br/>
                • <code className="bg-blue-100 px-1 rounded">data/synth_v3_large</code> - Большой датасет (100K примеров) - может переобучиться<br/>
                • <code className="bg-blue-100 px-1 rounded">data/synth_v3</code> - Средний датасет (10K примеров)<br/>
              </p>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Размер батча</label>
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
              <label className="block text-sm font-medium text-slate-700 mb-1">Эпохи</label>
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
              <label className="block text-sm font-medium text-slate-700 mb-1">Скорость обучения</label>
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
              Размер изображения
            </label>
            <div className="px-4 py-3 bg-emerald-50 border border-emerald-200 rounded-lg">
              <p className="text-sm font-semibold text-emerald-800">256×64 пикселей</p>
              <p className="text-xs text-emerald-600 mt-1">
                Равномерное масштабирование 4× обеспечивает отсутствие искажений и лучшую точность распознавания
              </p>
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">
              Макс. примеров (ограничение размера датасета)
            </label>
            <input
              type="number"
              className="w-full px-3 py-2 border border-slate-300 rounded-lg"
              value={formData.hyper.max_samples || ''}
              placeholder="Оставьте пустым для полного датасета"
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
              Ограничить обучение N примерами (80% обучение, 20% валидация). Оставьте пустым для полного датасета (~100K примеров из synth_v3_large).
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
              Инверсия цветов
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
            Отмена
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
            {isSubmitting ? 'Создание...' : 'Начать обучение'}
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
