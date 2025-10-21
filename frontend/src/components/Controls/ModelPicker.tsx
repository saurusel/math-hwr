export type ModelType = 'M1' | 'M2' | 'M3';

interface ModelPickerProps {
  selectedModel: ModelType;
  onModelChange: (model: ModelType) => void;
}

export function ModelPicker({ selectedModel, onModelChange }: ModelPickerProps) {
  return (
    <div className="flex items-center gap-2">
      <label className="text-sm text-slate-700 font-medium">Model:</label>
      <div className="flex gap-2">
        <button
          className={`px-3 py-1 rounded-lg border text-sm ${
            selectedModel === 'M1'
              ? 'bg-emerald-600 text-white border-emerald-600'
              : 'border-slate-300 hover:border-slate-400'
          }`}
          onClick={() => onModelChange('M1')}
        >
          M1 (CRNN-CTC)
        </button>
        <button
          className={`px-3 py-1 rounded-lg border text-sm relative ${
            selectedModel === 'M2'
              ? 'bg-amber-500 text-white border-amber-500'
              : 'border-slate-300 hover:border-slate-400'
          }`}
          onClick={() => onModelChange('M2')}
        >
          M2 (Attention)
          <span className="ml-1 text-xs opacity-75">β</span>
        </button>
        <button
          className={`px-3 py-1 rounded-lg border text-sm relative ${
            selectedModel === 'M3'
              ? 'bg-amber-500 text-white border-amber-500'
              : 'border-slate-300 hover:border-slate-400'
          }`}
          onClick={() => onModelChange('M3')}
        >
          M3 (Transformer)
          <span className="ml-1 text-xs opacity-75">β</span>
        </button>
      </div>
    </div>
  );
}
