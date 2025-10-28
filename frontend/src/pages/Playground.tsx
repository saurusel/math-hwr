import { useState, useRef } from 'react';
import { CanvasBoard } from '../components/CanvasBoard/CanvasBoard';
import { Toolbar } from '../components/Controls/Toolbar';
import { CheckpointPicker } from '../components/Controls/CheckpointPicker';
import { SubmitBar } from '../components/Controls/SubmitBar';
import { PredictionView } from '../components/ResultPanel/PredictionView';
import { DebugImages } from '../components/ResultPanel/DebugImages';
import { SegmentationView } from '../components/ResultPanel/SegmentationView';
import { predictM1, predictM2, predictM3, PredictResponse } from '../api/predict';
import { AvailableModel } from '../api/models';
import { APIError } from '../api/client';
import { useToast } from '../components/Toast/useToast';
import { Toast } from '../components/Toast/Toast';

type Tool = 'brush' | 'eraser' | 'pan';

export function Playground() {
  const [tool, setTool] = useState<Tool>('brush');
  const [brushSize, setBrushSize] = useState(3);
  const [zoom, setZoom] = useState(1);
  const [selectedCheckpoint, setSelectedCheckpoint] = useState<AvailableModel | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [inferenceTime, setInferenceTime] = useState<number>();
  const [result, setResult] = useState<PredictResponse | null>(null);

  const canvasDataURLRef = useRef<string>('');
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [loadedImageURL, setLoadedImageURL] = useState<string>('');

  const { toasts, removeToast, error, success } = useToast();

  const handleExport = (dataURL: string) => {
    canvasDataURLRef.current = dataURL;
  };

  const handleClear = () => {
    setResult(null);
    setInferenceTime(undefined);
    canvasDataURLRef.current = '';
    setLoadedImageURL('');
  };

  const handleResetView = () => {
    setZoom(1);
  };

  const handleLoadFile = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.type.match(/^image\/(png|jpeg|jpg)$/)) {
      error('Пожалуйста, выберите PNG или JPG изображение');
      return;
    }

    const reader = new FileReader();
    reader.onload = (event) => {
      const dataURL = event.target?.result as string;
      setLoadedImageURL(dataURL);
      canvasDataURLRef.current = dataURL;
      success('Изображение загружено успешно');
    };
    reader.readAsDataURL(file);
  };

  const handleImageLoad = (dataURL: string) => {
    canvasDataURLRef.current = dataURL;
  };

  const handleRecognize = async () => {
    if (!canvasDataURLRef.current) {
      error('Пожалуйста, нарисуйте или загрузите изображение');
      return;
    }

    if (!selectedCheckpoint) {
      error('Пожалуйста, выберите обученную модель');
      return;
    }

    setIsLoading(true);
    setInferenceTime(undefined);

    const startTime = performance.now();

    try {
      // Use selected checkpoint for inference
      const checkpointInfo = {
        run_id: selectedCheckpoint.run_id,
        checkpoint_name: selectedCheckpoint.checkpoint_name,
      };

      // Select predictor based on model type
      let response: PredictResponse;

      if (selectedCheckpoint.model_type === 'M2' || selectedCheckpoint.run_id.includes('attn')) {
        response = await predictM2(canvasDataURLRef.current, checkpointInfo);
      } else if (selectedCheckpoint.model_type === 'M3' || selectedCheckpoint.run_id.includes('vit')) {
        response = await predictM3(canvasDataURLRef.current, checkpointInfo);
      } else {
        // M1 or default
        response = await predictM1(canvasDataURLRef.current, checkpointInfo);
      }

      const endTime = performance.now();
      setInferenceTime(Math.round(endTime - startTime));
      setResult(response);

      console.log('Prediction response:', response); // Debug logging

      if (response.text && response.text.length > 0) {
        success('Распознавание завершено');
      } else {
        error('Модель вернула пустое предсказание. Модель может быть не обучена должным образом.');
      }
    } catch (err) {
      if (err instanceof APIError) {
        error(`Ошибка распознавания: ${err.message}`);
      } else if (err instanceof Error) {
        error(`Ошибка: ${err.message}`);
      } else {
        error('Произошла неизвестная ошибка');
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 p-4">
      <div className="max-w-7xl mx-auto space-y-4">
        {/* Header */}
        <div>
          <h1 className="text-3xl font-bold text-slate-800">Math HWR — Распознавание</h1>
          <p className="text-slate-600 mt-1">
            Рисуйте математические выражения или загрузите изображение для распознавания
          </p>
        </div>

        {/* Checkpoint Picker */}
        <div className="bg-white rounded-xl shadow p-4">
          <CheckpointPicker
            selectedCheckpoint={selectedCheckpoint}
            onCheckpointChange={setSelectedCheckpoint}
          />
        </div>

        {/* Toolbar */}
        <Toolbar
          tool={tool}
          onToolChange={setTool}
          brushSize={brushSize}
          onBrushSizeChange={setBrushSize}
          zoom={zoom}
          onZoomChange={setZoom}
          onResetView={handleResetView}
          onClear={handleClear}
          onLoadFile={handleLoadFile}
        />

        {/* Canvas Area */}
        <div className="bg-white rounded-xl shadow p-4">
          <div className="text-sm text-slate-600 mb-3 space-y-1">
            <div>💡 <strong>Пробел + перетаскивание:</strong> панорамирование</div>
            <div>💡 <strong>Колесо мыши:</strong> масштабирование (по центру курсора)</div>
            <div>💡 <strong>Ластик:</strong> показывает круглый контур</div>
          </div>
          <div className="overflow-auto border border-slate-200 rounded-lg" style={{ maxHeight: 400 }}>
            <CanvasBoard
              width={1024}
              height={256}
              zoom={zoom}
              brushSize={brushSize}
              tool={tool}
              onExport={handleExport}
              onClear={handleClear}
              onImageLoad={handleImageLoad}
              loadImageURL={loadedImageURL}
            />
          </div>
        </div>

        {/* Submit Bar */}
        <div className="bg-white rounded-xl shadow p-3">
          <SubmitBar
            onRecognize={handleRecognize}
            isLoading={isLoading}
            inferenceTime={inferenceTime}
          />
        </div>

        {/* Results */}
        <div className="space-y-4">
          <PredictionView
            text={result?.text || ''}
            tokens={result?.tokens}
            device={result?.device}
            ckpt={result?.ckpt}
            error={result?.error}
            modelType={result?.model_type}
          />

          {/* Debug binary image for M2 segmentation failures */}
          {result?.debug_binary && (
            <div className="bg-white rounded-xl shadow p-4">
              <h3 className="text-sm font-medium text-slate-700 mb-2">
                Debug: Binarized Image (what segmentation sees)
              </h3>
              <div className="border border-slate-200 rounded-lg overflow-hidden">
                <img src={result.debug_binary} alt="Binarized" className="max-w-full h-auto" />
              </div>
              <div className="text-xs text-slate-500 mt-2">
                This is the binarized image after Otsu thresholding. If you don't see clear white characters on black background, the image may need adjustment.
              </div>
            </div>
          )}

          {/* Segmentation Visualization for M2 models */}
          {result?.segments && result?.segments.length > 0 && result?.image_size && canvasDataURLRef.current && (
            <SegmentationView
              imageDataURL={canvasDataURLRef.current}
              segments={result.segments}
              imageSize={result.image_size}
            />
          )}

          {result?.preprocessed_b64 && <DebugImages preprocessedB64={result.preprocessed_b64} />}
        </div>

        {/* Hidden file input */}
        <input
          ref={fileInputRef}
          type="file"
          accept="image/png,image/jpeg,image/jpg"
          className="hidden"
          onChange={handleFileChange}
        />
      </div>

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
