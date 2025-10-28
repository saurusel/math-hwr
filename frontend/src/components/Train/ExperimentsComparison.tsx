import { useEffect, useState } from 'react';
import { getJobStatus } from '../../api/training';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import type { Experiment, MetricEvent } from '../../types/training';

interface ExperimentsComparisonProps {
  experiments: Experiment[];
  onClose: () => void;
}

export function ExperimentsComparison({ experiments, onClose }: ExperimentsComparisonProps) {
  const [metricsData, setMetricsData] = useState<Record<string, MetricEvent[]>>({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchMetrics = async () => {
      setLoading(true);
      const data: Record<string, MetricEvent[]> = {};

      // In real implementation, we'd fetch metrics from events.jsonl or API
      // For MVP, we'll use mock data structure
      for (const exp of experiments) {
        // TODO: Implement actual metrics fetching from backend
        // For now, just store empty array
        data[exp.run_id] = [];
      }

      setMetricsData(data);
      setLoading(false);
    };

    fetchMetrics();
  }, [experiments]);

  const colors = ['#f59e0b', '#3b82f6', '#10b981', '#8b5cf6', '#ef4444'];

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl shadow-2xl max-w-6xl w-full max-h-[90vh] overflow-y-auto m-4">
        <div className="sticky top-0 bg-white border-b border-slate-200 p-6 flex items-center justify-between">
          <h2 className="text-2xl font-bold text-slate-800">
            Compare Experiments ({experiments.length})
          </h2>
          <button
            className="text-slate-400 hover:text-slate-600 text-2xl"
            onClick={onClose}
          >
            ×
          </button>
        </div>

        <div className="p-6 space-y-6">
          {/* Experiments Summary */}
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            {experiments.map((exp, i) => (
              <div key={exp.run_id} className="bg-slate-50 p-4 rounded-lg">
                <div className="flex items-center gap-2 mb-2">
                  <div
                    className="w-4 h-4 rounded-full"
                    style={{ backgroundColor: colors[i % colors.length] }}
                  />
                  <div className="font-medium text-slate-800 truncate">{exp.run_name}</div>
                </div>
                <div className="text-sm text-slate-600">Model: {exp.model_type}</div>
                <div className="text-sm text-slate-600">
                  Status: <span className="font-medium">{exp.status}</span>
                </div>
                {exp.best_metrics && (
                  <div className="text-sm text-slate-600 mt-2">
                    Best CER: <span className="font-mono">{(exp.best_metrics.cer * 100).toFixed(2)}%</span>
                  </div>
                )}
              </div>
            ))}
          </div>

          {loading ? (
            <div className="text-center py-12 text-slate-400">Loading metrics...</div>
          ) : (
            <>
              {/* Loss Comparison Chart */}
              <div className="bg-white border border-slate-200 rounded-lg p-6">
                <h3 className="text-lg font-semibold text-slate-700 mb-4">Loss Curves Comparison</h3>
                <div className="text-center text-slate-400 py-12">
                  <p>Metrics comparison will be displayed here</p>
                  <p className="text-sm mt-2">
                    (Requires backend to provide historical metrics via API)
                  </p>
                </div>
                {/* TODO: Implement actual chart when backend provides metrics
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="epoch" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    {experiments.map((exp, i) => (
                      <Line
                        key={exp.run_id}
                        type="monotone"
                        dataKey="val_loss"
                        stroke={colors[i % colors.length]}
                        name={exp.run_name}
                        data={metricsData[exp.run_id] || []}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
                */}
              </div>

              {/* Metrics Table */}
              <div className="bg-white border border-slate-200 rounded-lg overflow-hidden">
                <table className="w-full">
                  <thead className="bg-slate-50">
                    <tr>
                      <th className="px-4 py-3 text-left text-sm font-semibold text-slate-700">
                        Run Name
                      </th>
                      <th className="px-4 py-3 text-left text-sm font-semibold text-slate-700">
                        Model
                      </th>
                      <th className="px-4 py-3 text-left text-sm font-semibold text-slate-700">
                        Status
                      </th>
                      <th className="px-4 py-3 text-left text-sm font-semibold text-slate-700">
                        Best CER
                      </th>
                      <th className="px-4 py-3 text-left text-sm font-semibold text-slate-700">
                        Best WER
                      </th>
                      <th className="px-4 py-3 text-left text-sm font-semibold text-slate-700">
                        Exact Match
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-200">
                    {experiments.map((exp, i) => (
                      <tr key={exp.run_id}>
                        <td className="px-4 py-3">
                          <div className="flex items-center gap-2">
                            <div
                              className="w-3 h-3 rounded-full"
                              style={{ backgroundColor: colors[i % colors.length] }}
                            />
                            <span className="font-medium text-slate-800">{exp.run_name}</span>
                          </div>
                        </td>
                        <td className="px-4 py-3 text-sm text-slate-600">{exp.model_type}</td>
                        <td className="px-4 py-3">
                          <span className={`px-2 py-1 rounded text-xs ${
                            exp.status === 'FINISHED' ? 'bg-blue-100 text-blue-700' :
                            exp.status === 'RUNNING' ? 'bg-emerald-100 text-emerald-700' :
                            'bg-slate-100 text-slate-700'
                          }`}>
                            {exp.status}
                          </span>
                        </td>
                        <td className="px-4 py-3 text-sm font-mono">
                          {exp.best_metrics?.cer
                            ? (exp.best_metrics.cer * 100).toFixed(2) + '%'
                            : '-'}
                        </td>
                        <td className="px-4 py-3 text-sm font-mono">
                          {exp.best_metrics?.wer
                            ? (exp.best_metrics.wer * 100).toFixed(2) + '%'
                            : '-'}
                        </td>
                        <td className="px-4 py-3 text-sm font-mono">
                          {exp.best_metrics?.exact
                            ? (exp.best_metrics.exact * 100).toFixed(2) + '%'
                            : '-'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
