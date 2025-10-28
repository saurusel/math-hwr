import { useEffect, useState } from 'react';
import { getAvailableModels, AvailableModel } from '../../api/models';

interface CheckpointPickerProps {
  selectedCheckpoint: AvailableModel | null;
  onCheckpointChange: (model: AvailableModel | null) => void;
}

export function CheckpointPicker({ selectedCheckpoint, onCheckpointChange }: CheckpointPickerProps) {
  const [models, setModels] = useState<AvailableModel[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getAvailableModels()
      .then((m) => {
        setModels(m);
        // Auto-select first model if none selected
        if (!selectedCheckpoint && m.length > 0) {
          onCheckpointChange(m[0]);
        }
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex items-center gap-2">
        <label className="text-sm text-slate-700 font-medium">Модель:</label>
        <div className="text-sm text-slate-500">Загрузка моделей...</div>
      </div>
    );
  }

  if (models.length === 0) {
    return (
      <div className="bg-amber-50 border border-amber-200 rounded-lg p-3">
        <div className="text-sm text-amber-800">
          <strong>Обученные модели не найдены.</strong> Пожалуйста, сначала обучите модель в разделе Обучение.
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <label className="text-sm text-slate-700 font-medium">Выберите обученную модель:</label>
      <select
        className="w-full px-4 py-2 border border-slate-300 rounded-lg"
        value={selectedCheckpoint?.id || ''}
        onChange={(e) => {
          const model = models.find((m) => m.id === e.target.value);
          onCheckpointChange(model || null);
        }}
      >
        <option value="">-- Выберите чекпоинт --</option>
        {models.map((model) => (
          <option key={model.id} value={model.id}>
            {model.run_id} / {model.checkpoint_kind} ({model.size_mb} МБ)
          </option>
        ))}
      </select>

      {selectedCheckpoint && (
        <div className="text-xs text-slate-500 flex items-center gap-3 mt-2">
          <span>📦 {selectedCheckpoint.checkpoint_name}</span>
          <span>📊 {selectedCheckpoint.model_type}</span>
          <span>💾 {selectedCheckpoint.size_mb} MB</span>
        </div>
      )}
    </div>
  );
}
