import { useState, useRef } from 'react';
import { CanvasBoard } from '../CanvasBoard/CanvasBoard';
import { predictM1 } from '../../api/predict';
import type { Checkpoint } from '../../types/training';
import type { PredictResponse } from '../../api/predict';

interface QuickTestProps {
  checkpoint: Checkpoint;
}

export function QuickTest({ checkpoint }: QuickTestProps) {
  const [mode, setMode] = useState<'canvas' | 'upload'>('canvas');
  const [brushSize, setBrushSize] = useState(3);
  const [zoom, setZoom] = useState(1);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [inferenceTime, setInferenceTime] = useState<number>();

  const canvasDataURLRef = useRef<string>('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleExport = (dataURL: string) => {
    canvasDataURLRef.current = dataURL;
  };

  const handleClear = () => {
    setResult(null);
    setInferenceTime(undefined);
  };

  const handleRecognize = async () => {
    if (!canvasDataURLRef.current) {
      alert('Please draw or load an image first');
      return;
    }

    setIsLoading(true);
    setInferenceTime(undefined);

    const startTime = performance.now();

    try {
      // For M1, use existing predict endpoint
      // TODO: Extend for M2/M3 when available
      const response = await predictM1(canvasDataURLRef.current);
      const endTime = performance.now();
      setInferenceTime(Math.round(endTime - startTime));
      setResult(response);
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Recognition failed');
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.type.match(/^image\/(png|jpeg|jpg)$/)) {
      alert('Please select a PNG or JPG image');
      return;
    }

    const reader = new FileReader();
    reader.onload = (event) => {
      const dataURL = event.target?.result as string;
      canvasDataURLRef.current = dataURL;
      // Load onto canvas would go here
    };
    reader.readAsDataURL(file);
  };

  return (
    <div className="space-y-6">
      {/* Mode Selection */}
      <div className="flex gap-2">
        <button
          className={`px-4 py-2 rounded-lg ${
            mode === 'canvas'
              ? 'bg-emerald-600 text-white'
              : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
          }`}
          onClick={() => setMode('canvas')}
        >
          ✏️ Canvas
        </button>
        <button
          className={`px-4 py-2 rounded-lg ${
            mode === 'upload'
              ? 'bg-emerald-600 text-white'
              : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
          }`}
          onClick={() => setMode('upload')}
        >
          📂 Upload
        </button>
      </div>

      {mode === 'canvas' ? (
        <>
          {/* Controls */}
          <div className="flex items-center gap-4 bg-slate-50 p-3 rounded-lg">
            <div className="flex items-center gap-2">
              <label className="text-sm text-slate-700">Size: {brushSize}px</label>
              <input
                type="range"
                min="1"
                max="32"
                value={brushSize}
                onChange={(e) => setBrushSize(parseInt(e.target.value))}
                className="w-24"
              />
            </div>
            <div className="flex items-center gap-2">
              <label className="text-sm text-slate-700">Zoom:</label>
              <select
                className="px-2 py-1 border border-slate-300 rounded text-sm"
                value={zoom}
                onChange={(e) => setZoom(parseFloat(e.target.value))}
              >
                <option value="0.5">50%</option>
                <option value="1">100%</option>
                <option value="1.5">150%</option>
                <option value="2">200%</option>
              </select>
            </div>
            <button
              className="ml-auto px-4 py-1 border border-red-300 text-red-700 rounded hover:bg-red-50 text-sm"
              onClick={handleClear}
            >
              Clear
            </button>
          </div>

          {/* Canvas */}
          <div className="border border-slate-200 rounded-lg overflow-auto" style={{ maxHeight: 300 }}>
            <CanvasBoard
              width={1024}
              height={256}
              zoom={zoom}
              brushSize={brushSize}
              tool="brush"
              onExport={handleExport}
              onClear={handleClear}
            />
          </div>
        </>
      ) : (
        <div className="border-2 border-dashed border-slate-300 rounded-lg p-12 text-center">
          <input
            ref={fileInputRef}
            type="file"
            accept="image/png,image/jpeg,image/jpg"
            className="hidden"
            onChange={handleFileChange}
          />
          <button
            className="px-6 py-3 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700"
            onClick={() => fileInputRef.current?.click()}
          >
            📂 Select Image
          </button>
          <p className="text-sm text-slate-500 mt-2">PNG or JPG, max 5MB</p>
        </div>
      )}

      {/* Recognize Button */}
      <button
        className={`w-full py-3 rounded-lg font-medium ${
          isLoading
            ? 'bg-slate-400 cursor-not-allowed'
            : 'bg-emerald-600 hover:bg-emerald-700 text-white'
        }`}
        onClick={handleRecognize}
        disabled={isLoading}
      >
        {isLoading ? 'Recognizing...' : 'Recognize'}
      </button>

      {/* Results */}
      {result && (
        <div className="bg-slate-50 rounded-lg p-4 space-y-3">
          <div>
            <div className="text-sm text-slate-600 mb-1">Recognized Text:</div>
            <div className="font-mono text-lg bg-white p-3 rounded border border-slate-200">
              {result.text}
            </div>
          </div>

          {inferenceTime && (
            <div className="text-sm text-slate-600">
              Inference time: <span className="font-mono font-semibold">{inferenceTime}ms</span>
            </div>
          )}

          {result.tokens && result.tokens.length > 0 && (
            <details className="text-sm">
              <summary className="cursor-pointer text-slate-600 hover:text-slate-800">
                Tokens ({result.tokens.length})
              </summary>
              <div className="mt-2 flex flex-wrap gap-1">
                {result.tokens.map((token, i) => (
                  <span
                    key={i}
                    className="px-2 py-0.5 bg-white border border-slate-300 rounded font-mono text-xs"
                  >
                    {token}
                  </span>
                ))}
              </div>
            </details>
          )}

          {result.preprocessed_b64 && (
            <details className="text-sm">
              <summary className="cursor-pointer text-slate-600 hover:text-slate-800">
                Preprocessed Image
              </summary>
              <div className="mt-2 bg-white p-2 rounded border border-slate-200">
                <img
                  src={result.preprocessed_b64}
                  alt="Preprocessed"
                  className="max-w-full h-auto"
                  style={{ imageRendering: 'pixelated' }}
                />
              </div>
            </details>
          )}
        </div>
      )}
    </div>
  );
}
