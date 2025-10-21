import { useEffect, useState } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import { getCheckpoints, promoteCheckpoint } from '../../api/training';
import { deleteCheckpoint } from '../../api/experiments';
import { QuickTest } from '../../components/Train/QuickTest';
import { useToast } from '../../components/Toast/useToast';
import { Toast } from '../../components/Toast/Toast';
import type { Checkpoint } from '../../types/training';

export function Models() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const runId = searchParams.get('run');
  const ckptId = searchParams.get('ckpt');

  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [selectedCheckpoint, setSelectedCheckpoint] = useState<Checkpoint | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'info' | 'test'>('info');

  const { toasts, removeToast, success, error } = useToast();

  useEffect(() => {
    if (!runId) return;

    setLoading(true);
    getCheckpoints(runId)
      .then((ckpts) => {
        setCheckpoints(ckpts);
        // Auto-select checkpoint from URL or best checkpoint
        if (ckptId) {
          const ckpt = ckpts.find((c) => c.ckpt_id === ckptId);
          if (ckpt) setSelectedCheckpoint(ckpt);
        } else {
          const best = ckpts.find((c) => c.kind === 'best');
          if (best) setSelectedCheckpoint(best);
        }
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [runId, ckptId]);

  const handleSelectCheckpoint = (ckpt: Checkpoint) => {
    setSelectedCheckpoint(ckpt);
    navigate(`/train/models?run=${runId}&ckpt=${ckpt.ckpt_id}`, { replace: true });
  };

  const handlePromote = async (ckptId: string) => {
    try {
      await promoteCheckpoint(ckptId, 'production');
      success('Checkpoint promoted to production');
    } catch (err) {
      error(err instanceof Error ? err.message : 'Failed to promote checkpoint');
    }
  };

  const handleDelete = async (ckptId: string) => {
    if (!confirm('Are you sure you want to delete this checkpoint? This action cannot be undone.')) {
      return;
    }

    try {
      await deleteCheckpoint(ckptId);
      success('Checkpoint deleted successfully');

      // Refresh checkpoints list
      if (runId) {
        const ckpts = await getCheckpoints(runId);
        setCheckpoints(ckpts);

        // If deleted checkpoint was selected, clear selection
        if (selectedCheckpoint?.ckpt_id === ckptId) {
          setSelectedCheckpoint(null);
        }
      }
    } catch (err) {
      error(err instanceof Error ? err.message : 'Failed to delete checkpoint');
    }
  };

  if (!runId) {
    return (
      <div className="max-w-7xl mx-auto p-6">
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-800">
          No run ID provided. Please select a run from experiments.
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto p-6">
      <h1 className="text-3xl font-bold text-slate-800 mb-6">Models & Checkpoints</h1>

      <div className="grid grid-cols-12 gap-6">
        {/* Left: Checkpoints List */}
        <div className="col-span-4">
          <div className="bg-white rounded-xl shadow p-4">
            <h2 className="text-lg font-semibold text-slate-700 mb-4">
              Checkpoints ({checkpoints.length})
            </h2>

            {loading ? (
              <div className="text-center text-slate-400 py-8">Loading...</div>
            ) : checkpoints.length === 0 ? (
              <div className="text-center text-slate-400 py-8">No checkpoints yet</div>
            ) : (
              <div className="space-y-2">
                {checkpoints.map((ckpt) => (
                  <button
                    key={ckpt.ckpt_id}
                    className={`w-full text-left p-3 rounded-lg border transition-colors ${
                      selectedCheckpoint?.ckpt_id === ckpt.ckpt_id
                        ? 'border-emerald-500 bg-emerald-50'
                        : 'border-slate-200 hover:border-slate-300 hover:bg-slate-50'
                    }`}
                    onClick={() => handleSelectCheckpoint(ckpt)}
                  >
                    <div className="flex items-start justify-between">
                      <div>
                        <div className="font-medium text-slate-800">
                          Epoch {ckpt.epoch}
                        </div>
                        <div className="text-xs text-slate-500">
                          {ckpt.kind === 'best' && '⭐ Best'}
                          {ckpt.kind === 'last' && '📍 Latest'}
                          {ckpt.kind === 'epoch' && '📦 Checkpoint'}
                        </div>
                      </div>
                      <div className="text-xs text-slate-500">
                        {(ckpt.size / 1024 / 1024).toFixed(1)} MB
                      </div>
                    </div>
                    {ckpt.metrics_at_save && (
                      <div className="mt-2 text-xs text-slate-600">
                        {ckpt.metrics_at_save.cer !== undefined && (
                          <span>CER: {(ckpt.metrics_at_save.cer * 100).toFixed(2)}%</span>
                        )}
                      </div>
                    )}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Right: Checkpoint Details & Quick Test */}
        <div className="col-span-8">
          {selectedCheckpoint ? (
            <div className="space-y-6">
              {/* Tabs */}
              <div className="bg-white rounded-xl shadow">
                <div className="border-b border-slate-200">
                  <div className="flex">
                    <button
                      className={`px-6 py-3 font-medium ${
                        activeTab === 'info'
                          ? 'border-b-2 border-emerald-500 text-emerald-600'
                          : 'text-slate-600 hover:text-slate-800'
                      }`}
                      onClick={() => setActiveTab('info')}
                    >
                      Info
                    </button>
                    <button
                      className={`px-6 py-3 font-medium ${
                        activeTab === 'test'
                          ? 'border-b-2 border-emerald-500 text-emerald-600'
                          : 'text-slate-600 hover:text-slate-800'
                      }`}
                      onClick={() => setActiveTab('test')}
                    >
                      Quick Test
                    </button>
                  </div>
                </div>

                <div className="p-6">
                  {activeTab === 'info' ? (
                    <div className="space-y-4">
                      <div>
                        <h3 className="text-lg font-semibold text-slate-800 mb-3">
                          Checkpoint Details
                        </h3>
                        <div className="grid grid-cols-2 gap-4">
                          <div className="bg-slate-50 p-3 rounded-lg">
                            <div className="text-sm text-slate-600">Epoch</div>
                            <div className="text-xl font-bold">{selectedCheckpoint.epoch}</div>
                          </div>
                          <div className="bg-slate-50 p-3 rounded-lg">
                            <div className="text-sm text-slate-600">Type</div>
                            <div className="text-xl font-bold capitalize">
                              {selectedCheckpoint.kind}
                            </div>
                          </div>
                          <div className="bg-slate-50 p-3 rounded-lg">
                            <div className="text-sm text-slate-600">Size</div>
                            <div className="text-xl font-bold">
                              {(selectedCheckpoint.size / 1024 / 1024).toFixed(1)} MB
                            </div>
                          </div>
                          <div className="bg-slate-50 p-3 rounded-lg">
                            <div className="text-sm text-slate-600">Path</div>
                            <div className="text-sm font-mono text-slate-700 truncate">
                              {selectedCheckpoint.path.split('/').pop()}
                            </div>
                          </div>
                        </div>
                      </div>

                      {selectedCheckpoint.metrics_at_save && (
                        <div>
                          <h3 className="text-lg font-semibold text-slate-800 mb-3">
                            Metrics at Save
                          </h3>
                          <div className="grid grid-cols-3 gap-4">
                            {Object.entries(selectedCheckpoint.metrics_at_save).map(
                              ([key, value]) => (
                                <div key={key} className="bg-slate-50 p-3 rounded-lg">
                                  <div className="text-sm text-slate-600 uppercase">{key}</div>
                                  <div className="text-xl font-bold">
                                    {typeof value === 'number'
                                      ? (value * 100).toFixed(2) + '%'
                                      : value}
                                  </div>
                                </div>
                              )
                            )}
                          </div>
                        </div>
                      )}

                      <div className="flex gap-3 pt-4">
                        <button
                          className="px-6 py-2 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700"
                          onClick={() => handlePromote(selectedCheckpoint.ckpt_id)}
                        >
                          ⭐ Promote to Production
                        </button>
                        <button
                          className="px-6 py-2 border border-slate-300 rounded-lg hover:bg-slate-50"
                          onClick={() => {
                            // Download checkpoint (future implementation)
                            alert('Download feature coming soon');
                          }}
                        >
                          ⬇️ Download
                        </button>
                        <button
                          className="px-6 py-2 border border-red-300 text-red-700 rounded-lg hover:bg-red-50"
                          onClick={() => handleDelete(selectedCheckpoint.ckpt_id)}
                          disabled={selectedCheckpoint.kind === 'best'}
                          title={
                            selectedCheckpoint.kind === 'best'
                              ? 'Cannot delete best checkpoint'
                              : 'Delete checkpoint'
                          }
                        >
                          🗑️ Delete
                        </button>
                      </div>
                    </div>
                  ) : (
                    <QuickTest checkpoint={selectedCheckpoint} />
                  )}
                </div>
              </div>
            </div>
          ) : (
            <div className="bg-white rounded-xl shadow p-12 text-center text-slate-400">
              Select a checkpoint to view details
            </div>
          )}
        </div>
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
