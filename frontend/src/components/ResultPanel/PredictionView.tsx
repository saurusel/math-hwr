import { useState } from 'react';

interface PredictionViewProps {
  text: string;
  tokens?: string[];
  device?: string;
  ckpt?: string;
  error?: string;
  modelType?: string;
}

export function PredictionView({ text, tokens, device, ckpt, error, modelType }: PredictionViewProps) {
  const [showTokens, setShowTokens] = useState(false);
  const [copySuccess, setCopySuccess] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text || '(empty)');
      setCopySuccess(true);
      setTimeout(() => setCopySuccess(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  // Check if we have a result (even if empty text)
  const hasResult = text !== undefined && text !== null;

  if (!hasResult && !tokens && !device && !ckpt && !error) {
    return (
      <div className="bg-white rounded-xl shadow p-4">
        <div className="text-slate-400 text-center py-8">
          Результатов пока нет. Нарисуйте или загрузите изображение и нажмите "Распознать".
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-xl shadow p-4 space-y-3">
      {/* Error message for M2 segmentation failures */}
      {error && modelType === 'M2_Segmentation_MLP' && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-3">
          <div className="text-sm font-medium text-red-800 mb-1">
            Ошибка сегментации:
          </div>
          <div className="text-sm text-red-700">
            {error}
          </div>
          <div className="text-xs text-red-600 mt-2">
            💡 Возможные причины:
            <ul className="list-disc list-inside mt-1 space-y-0.5">
              <li>Изображение слишком светлое или темное</li>
              <li>Очень низкий контраст</li>
              <li>Символы слишком маленькие или большие</li>
              <li>Попробуйте использовать модель M1 (CRNN-CTC)</li>
            </ul>
          </div>
        </div>
      )}

      <div className="flex items-start justify-between gap-3">
        <div className="flex-1">
          <label className="text-sm font-medium text-slate-700 block mb-1">Распознанный текст:</label>
          <div className={`font-mono text-lg p-3 rounded border break-all ${
            text && text.length > 0
              ? 'bg-slate-50 border-slate-200'
              : error
              ? 'bg-red-50 border-red-300'
              : 'bg-amber-50 border-amber-300'
          }`}>
            {text && text.length > 0 ? text : error ? '(ошибка сегментации - символы не обнаружены)' : '(пустое предсказание - модель может быть не обучена должным образом)'}
          </div>
        </div>
        <button
          className={`px-3 py-2 rounded-lg border text-sm whitespace-nowrap ${
            copySuccess
              ? 'bg-emerald-50 border-emerald-300 text-emerald-700'
              : 'border-slate-300 hover:border-slate-400 hover:bg-slate-50'
          }`}
          onClick={handleCopy}
        >
          {copySuccess ? '✓ Скопировано!' : '📋 Копировать'}
        </button>
      </div>

      {tokens && tokens.length > 0 && (
        <div>
          <button
            className="text-sm text-slate-600 hover:text-slate-900 flex items-center gap-1"
            onClick={() => setShowTokens(!showTokens)}
          >
            <span>{showTokens ? '▼' : '▶'}</span>
            Токены ({tokens.length})
          </button>
          {showTokens && (
            <div className="mt-2 font-mono text-sm bg-slate-50 p-3 rounded border border-slate-200 flex flex-wrap gap-1">
              {tokens.map((token, i) => (
                <span
                  key={i}
                  className="px-2 py-0.5 bg-white border border-slate-300 rounded"
                >
                  {token}
                </span>
              ))}
            </div>
          )}
        </div>
      )}

      {(device || ckpt) && (
        <div className="text-xs text-slate-500 space-y-1">
          {device && <div>Устройство: {device}</div>}
          {ckpt && <div>Чекпоинт: {ckpt}</div>}
        </div>
      )}
    </div>
  );
}
