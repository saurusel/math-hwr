interface ToolbarProps {
  tool: 'brush' | 'eraser' | 'pan';
  onToolChange: (tool: 'brush' | 'eraser' | 'pan') => void;
  brushSize: number;
  onBrushSizeChange: (size: number) => void;
  zoom: number;
  onZoomChange: (zoom: number) => void;
  onResetView: () => void;
  onClear: () => void;
  onLoadFile: () => void;
}

export function Toolbar({
  tool,
  onToolChange,
  brushSize,
  onBrushSizeChange,
  zoom,
  onZoomChange,
  onResetView,
  onClear,
  onLoadFile,
}: ToolbarProps) {
  return (
    <div className="flex flex-wrap items-center gap-3 bg-white rounded-xl shadow p-3">
      {/* Tool Selection */}
      <div className="flex items-center gap-2">
        <button
          className={`px-3 py-1 rounded-lg border ${
            tool === 'brush'
              ? 'bg-slate-900 text-white border-slate-900'
              : 'border-slate-300 hover:border-slate-400'
          }`}
          onClick={() => onToolChange('brush')}
          title="Кисть (B)"
        >
          ✏️ Кисть
        </button>
        <button
          className={`px-3 py-1 rounded-lg border ${
            tool === 'eraser'
              ? 'bg-slate-900 text-white border-slate-900'
              : 'border-slate-300 hover:border-slate-400'
          }`}
          onClick={() => onToolChange('eraser')}
          title="Ластик (E)"
        >
          🩹 Ластик
        </button>
      </div>

      {/* Brush Size */}
      <div className="flex items-center gap-2">
        <label className="text-sm text-slate-700">Размер: {brushSize}px</label>
        <input
          type="range"
          min="1"
          max="32"
          step="1"
          value={brushSize}
          onChange={(e) => onBrushSizeChange(parseInt(e.target.value, 10))}
          className="w-24"
        />
      </div>

      {/* Zoom Controls */}
      <div className="flex items-center gap-2">
        <label className="text-sm text-slate-700">Масштаб:</label>
        <button
          className="px-2 py-1 rounded border border-slate-300 hover:border-slate-400 text-sm"
          onClick={() => onZoomChange(Math.max(0.25, zoom - 0.25))}
          title="Zoom out"
        >
          −
        </button>
        <select
          className="px-2 py-1 rounded border border-slate-300 text-sm"
          value={zoom}
          onChange={(e) => onZoomChange(parseFloat(e.target.value))}
        >
          <option value="0.25">25%</option>
          <option value="0.5">50%</option>
          <option value="0.75">75%</option>
          <option value="1">100%</option>
          <option value="1.5">150%</option>
          <option value="2">200%</option>
          <option value="3">300%</option>
          <option value="4">400%</option>
        </select>
        <button
          className="px-2 py-1 rounded border border-slate-300 hover:border-slate-400 text-sm"
          onClick={() => onZoomChange(Math.min(4, zoom + 0.25))}
          title="Zoom in"
        >
          +
        </button>
      </div>

      {/* Actions */}
      <div className="ml-auto flex items-center gap-2">
        <button
          className="px-3 py-1 rounded-lg border border-slate-300 hover:border-slate-400 text-sm"
          onClick={onResetView}
          title="Сбросить масштаб и панорамирование"
        >
          Сбросить вид
        </button>
        <button
          className="px-3 py-1 rounded-lg border border-slate-300 hover:border-slate-400 text-sm"
          onClick={onLoadFile}
          title="Загрузить изображение"
        >
          📂 Открыть файл
        </button>
        <button
          className="px-3 py-1 rounded-lg border border-red-300 text-red-700 hover:border-red-400 hover:bg-red-50 text-sm"
          onClick={onClear}
          title="Очистить холст"
        >
          Очистить
        </button>
      </div>
    </div>
  );
}
