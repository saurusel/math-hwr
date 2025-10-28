import { useEffect, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { TrainingSSEClient } from '../../services/sseClient';
import { useTrainingStore } from '../../stores/trainingStore';
import { getJobStatus, getTrainingHistory, stopJob } from '../../api/training';
import { NotesAndTags } from '../../components/Train/NotesAndTags';
import { ArtifactPreview } from '../../components/Train/ArtifactPreview';
import { ImprovedCharts } from '../../components/Train/ImprovedCharts';
import { BestMetricsTracker } from '../../components/Train/BestMetricsTracker';
import { SampleGallery } from '../../components/Train/SampleGallery';
import { useToast } from '../../components/Toast/useToast';
import { Toast } from '../../components/Toast/Toast';

export function Monitor() {
  const [searchParams] = useSearchParams();
  const jobId = searchParams.get('job');
  const [sseClient, setSSEClient] = useState<TrainingSSEClient | null>(null);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [isStopping, setIsStopping] = useState(false);

  const { currentJob, metrics, logs, samples, setCurrentJob, addMetric, addLog, addSamples, addCheckpoint, clearJobData } = useTrainingStore();
  const { toasts, removeToast, success, error, warning } = useToast();

  useEffect(() => {
    if (!jobId) return;

    clearJobData();

    // Fetch initial job status
    getJobStatus(jobId)
      .then(async (job) => {
        setCurrentJob(job);

        // If job is FINISHED, STOPPED, or FAILED - load historical data, DON'T use SSE
        if (['FINISHED', 'STOPPED', 'FAILED'].includes(job.status)) {
          setLoadingHistory(true);
          try {
            const history = await getTrainingHistory(jobId);
            console.log(`[Monitor] Loaded history for ${job.status} job:`, {
              metrics: history.metrics.length,
              logs: history.logs.length,
              samples: history.samples?.length || 0
            });

            // Load metrics into store
            history.metrics.forEach(addMetric);
            history.logs.forEach(addLog);
            if (history.samples) {
              history.samples.forEach(addSamples);
            }
          } catch (err) {
            console.error('Failed to load history:', err);
          } finally {
            setLoadingHistory(false);
          }

          // IMPORTANT: Don't create SSE client for finished jobs
          return;
        }

        // For RUNNING jobs only - connect to SSE
        const client = new TrainingSSEClient(jobId, {
          onMetric: (metric) => {
            addMetric(metric);
            // Update current epoch in job
            setCurrentJob((prev) => prev ? { ...prev, current_epoch: metric.epoch } : null);
          },
          onLog: addLog,
          onSamplePred: addSamples,
          onCheckpoint: addCheckpoint,
          onStatus: (status) => {
            setCurrentJob((prev) => prev ? { ...prev, status: status.status } : null);
          },
        });

        client.connect();
        setSSEClient(client);
      })
      .catch(console.error);

    return () => {
      if (sseClient) {
        sseClient.disconnect();
      }
    };
  }, [jobId]);

  if (!jobId) {
    return (
      <div className="max-w-7xl mx-auto p-6">
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-800">
          ID задачи не указан
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="bg-white rounded-xl shadow p-4">
        <h1 className="text-2xl font-bold text-slate-800">{currentJob?.run_name || jobId}</h1>
        <div className="mt-2 flex items-center gap-4 text-sm flex-wrap">
          <span className={`px-3 py-1 rounded-full ${
            currentJob?.status === 'RUNNING' ? 'bg-emerald-100 text-emerald-800' :
            currentJob?.status === 'STOPPING' ? 'bg-orange-100 text-orange-800' :
            currentJob?.status === 'FINISHED' ? 'bg-blue-100 text-blue-800' :
            currentJob?.status === 'STOPPED' ? 'bg-slate-100 text-slate-800' :
            currentJob?.status === 'FAILED' ? 'bg-red-100 text-red-800' :
            'bg-slate-100 text-slate-800'
          }`}>
            {currentJob?.status === 'RUNNING' ? 'ОБУЧЕНИЕ' :
             currentJob?.status === 'STOPPING' ? 'ОСТАНОВКА' :
             currentJob?.status === 'FINISHED' ? 'ЗАВЕРШЕНО' :
             currentJob?.status === 'STOPPED' ? 'ОСТАНОВЛЕНО' :
             currentJob?.status === 'FAILED' ? 'ОШИБКА' :
             currentJob?.status || 'ЗАГРУЗКА...'}
          </span>
          <span className="text-slate-600">
            Эпоха: {metrics.length > 0 ? metrics[metrics.length - 1].epoch : (currentJob?.current_epoch || 0)}
          </span>
          {currentJob?.status === 'FINISHED' && metrics.length === 0 && !loadingHistory && (
            <span className="px-3 py-1 bg-amber-100 text-amber-800 rounded text-xs">
              ⚠ Исторические данные не найдены. Эта задача могла быть обучена до добавления сохранения событий.
            </span>
          )}
          {loadingHistory && (
            <span className="px-3 py-1 bg-blue-100 text-blue-800 rounded text-xs">
              📊 Загрузка исторических данных...
            </span>
          )}
        </div>

        {/* Stop Training Button */}
        {(currentJob?.status === 'RUNNING' || currentJob?.status === 'STOPPING') && jobId && (
          <div className="mt-3 flex items-center gap-3">
            <button
              className={`px-4 py-2 rounded-lg font-medium ${
                isStopping || currentJob?.status === 'STOPPING'
                  ? 'bg-slate-400 cursor-not-allowed'
                  : 'bg-red-600 hover:bg-red-700 text-white'
              }`}
              onClick={async () => {
                if (!confirm('Остановить обучение досрочно? Текущий чекпоинт будет сохранен.\n\nЭто завершит текущую эпоху и сохранит checkpoint_latest.pt.\n\nЛучший чекпоинт будет сохранен, если он лучше текущей эпохи.')) {
                  return;
                }
                setIsStopping(true);
                try {
                  await stopJob(jobId);
                  warning('Запрошена остановка обучения. Завершаем текущую эпоху и сохраняем чекпоинт...');
                } catch (err) {
                  error(err instanceof Error ? err.message : 'Не удалось остановить обучение');
                  setIsStopping(false);
                }
              }}
              disabled={isStopping || currentJob?.status === 'STOPPING'}
            >
              {currentJob?.status === 'STOPPING' ? '⏳ Завершаем эпоху...' : isStopping ? '⏳ Остановка...' : '⏹️ Остановить обучение'}
            </button>
            {currentJob?.status === 'STOPPING' && (
              <span className="text-sm text-orange-700">
                Корректная остановка... Пожалуйста, дождитесь завершения текущей эпохи.
              </span>
            )}
          </div>
        )}
      </div>

      {/* Best Metrics Tracker */}
      <BestMetricsTracker metrics={metrics} />

      {/* Improved Charts with Multiple Views */}
      <ImprovedCharts metrics={metrics} />

      {/* Metrics Table */}
      {metrics.length > 0 && (
        <div className="bg-white rounded-xl shadow p-6">
          <h2 className="text-lg font-semibold text-slate-700 mb-4">Последние метрики (Эпоха {metrics[metrics.length - 1]?.epoch})</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {/* Error Rates */}
            {metrics[metrics.length - 1]?.cer !== undefined && (
              <div className="bg-slate-50 p-4 rounded-lg">
                <div className="text-sm text-slate-600">Ошибка символов (CER)</div>
                <div className="text-2xl font-bold text-emerald-700">{(metrics[metrics.length - 1].cer! * 100).toFixed(2)}%</div>
              </div>
            )}
            {metrics[metrics.length - 1]?.exact !== undefined && (
              <div className="bg-slate-50 p-4 rounded-lg">
                <div className="text-sm text-slate-600">Полное совпадение</div>
                <div className="text-2xl font-bold text-purple-700">{(metrics[metrics.length - 1].exact! * 100).toFixed(2)}%</div>
              </div>
            )}
            {metrics[metrics.length - 1]?.valid !== undefined && (
              <div className="bg-slate-50 p-4 rounded-lg">
                <div className="text-sm text-slate-600">Валидность %</div>
                <div className="text-2xl font-bold text-indigo-700">{(metrics[metrics.length - 1].valid! * 100).toFixed(2)}%</div>
              </div>
            )}

            {/* Learning Info */}
            <div className="bg-slate-50 p-4 rounded-lg">
              <div className="text-sm text-slate-600">Скорость обучения</div>
              <div className="text-xl font-bold font-mono">{metrics[metrics.length - 1]?.lr.toExponential(2)}</div>
            </div>

            {/* Performance */}
            {metrics[metrics.length - 1]?.epoch_time_sec !== undefined && (
              <div className="bg-slate-50 p-4 rounded-lg">
                <div className="text-sm text-slate-600">Время эпохи</div>
                <div className="text-2xl font-bold">{metrics[metrics.length - 1].epoch_time_sec!.toFixed(1)}с</div>
              </div>
            )}
            {metrics[metrics.length - 1]?.samples_processed !== undefined && (
              <div className="bg-slate-50 p-4 rounded-lg">
                <div className="text-sm text-slate-600">Примеров/Эпоха</div>
                <div className="text-2xl font-bold">{metrics[metrics.length - 1].samples_processed}</div>
              </div>
            )}
            {metrics[metrics.length - 1]?.batches_processed !== undefined && (
              <div className="bg-slate-50 p-4 rounded-lg">
                <div className="text-sm text-slate-600">Батчей/Эпоха</div>
                <div className="text-2xl font-bold">{metrics[metrics.length - 1].batches_processed}</div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Logs */}
      <div className="bg-white rounded-xl shadow p-6">
        <h2 className="text-lg font-semibold text-slate-700 mb-4">Логи</h2>
        <div className="bg-slate-900 text-slate-100 p-4 rounded-lg font-mono text-sm max-h-96 overflow-y-auto">
          {logs.length > 0 ? (
            logs.map((log, i) => (
              <div key={i} className="border-b border-slate-700 py-1">
                <span className="text-slate-400">[{log.ts}]</span> {log.line}
              </div>
            ))
          ) : (
            <div className="text-slate-400">Логов пока нет</div>
          )}
        </div>
      </div>

      {/* Sample Predictions Gallery with Images */}
      <SampleGallery samples={samples} />

      {/* Notes & Tags */}
      {jobId && (
        <div className="bg-white rounded-xl shadow p-6">
          <h2 className="text-lg font-semibold text-slate-700 mb-4">Notes & Tags</h2>
          <NotesAndTags runId={jobId} />
        </div>
      )}

      {/* Artifact Preview */}
      {jobId && currentJob?.status === 'FINISHED' && (
        <ArtifactPreview runId={jobId} epoch={currentJob.current_epoch} />
      )}

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
