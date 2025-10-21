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

        // If job is FINISHED, try to load historical data
        if (job.status === 'FINISHED') {
          setLoadingHistory(true);
          try {
            const history = await getTrainingHistory(jobId);
            console.log('[Monitor] Loaded historical data:', history.metrics.length, 'metrics');

            // Load metrics into store
            history.metrics.forEach(addMetric);
            history.logs.forEach(addLog);
          } catch (err) {
            console.error('Failed to load history:', err);
          } finally {
            setLoadingHistory(false);
          }
        }
      })
      .catch(console.error);

    // Connect SSE (will handle FINISHED jobs gracefully)
    const client = new TrainingSSEClient(jobId, {
      onMetric: addMetric,
      onLog: addLog,
      onSamplePred: addSamples,
      onCheckpoint: addCheckpoint,
      onStatus: (status) => {
        setCurrentJob((prev) => prev ? { ...prev, status: status.status } : null);
      },
    });

    client.connect();
    setSSEClient(client);

    return () => {
      client.disconnect();
    };
  }, [jobId]);

  if (!jobId) {
    return (
      <div className="max-w-7xl mx-auto p-6">
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-800">
          No job ID provided
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
            {currentJob?.status || 'UNKNOWN'}
          </span>
          <span className="text-slate-600">
            Epoch: {currentJob?.current_epoch || 0}
          </span>
          {currentJob?.status === 'FINISHED' && metrics.length === 0 && !loadingHistory && (
            <span className="px-3 py-1 bg-amber-100 text-amber-800 rounded text-xs">
              ⚠ No historical data found. This job may have been trained before events persistence was added.
            </span>
          )}
          {loadingHistory && (
            <span className="px-3 py-1 bg-blue-100 text-blue-800 rounded text-xs">
              📊 Loading historical data...
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
                if (!confirm('Stop training early? The current checkpoint will be saved.\n\nThis will finish the current epoch and save checkpoint_latest.pt.\n\nBest checkpoint will be preserved if better than current epoch.')) {
                  return;
                }
                setIsStopping(true);
                try {
                  await stopJob(jobId);
                  warning('Training stop requested. Finishing current epoch and saving checkpoint...');
                } catch (err) {
                  error(err instanceof Error ? err.message : 'Failed to stop training');
                  setIsStopping(false);
                }
              }}
              disabled={isStopping || currentJob?.status === 'STOPPING'}
            >
              {currentJob?.status === 'STOPPING' ? '⏳ Finishing epoch...' : isStopping ? '⏳ Stopping...' : '⏹️ Stop Training Early'}
            </button>
            {currentJob?.status === 'STOPPING' && (
              <span className="text-sm text-orange-700">
                Stopping gracefully... Please wait for current epoch to complete.
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
          <h2 className="text-lg font-semibold text-slate-700 mb-4">Latest Metrics (Epoch {metrics[metrics.length - 1]?.epoch})</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {/* Error Rates */}
            {metrics[metrics.length - 1]?.cer !== undefined && (
              <div className="bg-slate-50 p-4 rounded-lg">
                <div className="text-sm text-slate-600">CER</div>
                <div className="text-2xl font-bold text-emerald-700">{(metrics[metrics.length - 1].cer! * 100).toFixed(2)}%</div>
              </div>
            )}
            {metrics[metrics.length - 1]?.wer !== undefined && (
              <div className="bg-slate-50 p-4 rounded-lg">
                <div className="text-sm text-slate-600">WER</div>
                <div className="text-2xl font-bold text-blue-700">{(metrics[metrics.length - 1].wer! * 100).toFixed(2)}%</div>
              </div>
            )}
            {metrics[metrics.length - 1]?.exact !== undefined && (
              <div className="bg-slate-50 p-4 rounded-lg">
                <div className="text-sm text-slate-600">Exact Match</div>
                <div className="text-2xl font-bold text-purple-700">{(metrics[metrics.length - 1].exact! * 100).toFixed(2)}%</div>
              </div>
            )}
            {metrics[metrics.length - 1]?.valid !== undefined && (
              <div className="bg-slate-50 p-4 rounded-lg">
                <div className="text-sm text-slate-600">Valid %</div>
                <div className="text-2xl font-bold text-indigo-700">{(metrics[metrics.length - 1].valid! * 100).toFixed(2)}%</div>
              </div>
            )}

            {/* Learning Info */}
            <div className="bg-slate-50 p-4 rounded-lg">
              <div className="text-sm text-slate-600">Learning Rate</div>
              <div className="text-xl font-bold font-mono">{metrics[metrics.length - 1]?.lr.toExponential(2)}</div>
            </div>

            {/* Performance */}
            {metrics[metrics.length - 1]?.epoch_time_sec !== undefined && (
              <div className="bg-slate-50 p-4 rounded-lg">
                <div className="text-sm text-slate-600">Epoch Time</div>
                <div className="text-2xl font-bold">{metrics[metrics.length - 1].epoch_time_sec!.toFixed(1)}s</div>
              </div>
            )}
            {metrics[metrics.length - 1]?.samples_processed !== undefined && (
              <div className="bg-slate-50 p-4 rounded-lg">
                <div className="text-sm text-slate-600">Samples/Epoch</div>
                <div className="text-2xl font-bold">{metrics[metrics.length - 1].samples_processed}</div>
              </div>
            )}
            {metrics[metrics.length - 1]?.batches_processed !== undefined && (
              <div className="bg-slate-50 p-4 rounded-lg">
                <div className="text-sm text-slate-600">Batches/Epoch</div>
                <div className="text-2xl font-bold">{metrics[metrics.length - 1].batches_processed}</div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Logs */}
      <div className="bg-white rounded-xl shadow p-6">
        <h2 className="text-lg font-semibold text-slate-700 mb-4">Logs</h2>
        <div className="bg-slate-900 text-slate-100 p-4 rounded-lg font-mono text-sm max-h-96 overflow-y-auto">
          {logs.length > 0 ? (
            logs.map((log, i) => (
              <div key={i} className="border-b border-slate-700 py-1">
                <span className="text-slate-400">[{log.ts}]</span> {log.line}
              </div>
            ))
          ) : (
            <div className="text-slate-400">No logs yet</div>
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
