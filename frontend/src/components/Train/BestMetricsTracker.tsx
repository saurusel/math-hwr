import type { MetricEvent } from '../../types/training';

interface BestMetricsTrackerProps {
  metrics: MetricEvent[];
}

export function BestMetricsTracker({ metrics }: BestMetricsTrackerProps) {
  if (metrics.length === 0) return null;

  // Find best values across all epochs
  const bestCER = Math.min(...metrics.map((m) => m.cer !== undefined ? m.cer : 1));
  const bestWER = Math.min(...metrics.map((m) => m.wer !== undefined ? m.wer : 1));
  const bestExact = Math.max(...metrics.map((m) => m.exact !== undefined ? m.exact : 0));
  const bestValid = Math.max(...metrics.map((m) => m.valid !== undefined ? m.valid : 0));

  // Find epochs where best was achieved
  const bestCEREpoch = metrics.find((m) => m.cer === bestCER)?.epoch || 0;
  const bestWEREpoch = metrics.find((m) => m.wer === bestWER)?.epoch || 0;
  const bestExactEpoch = metrics.find((m) => m.exact === bestExact)?.epoch || 0;

  // Calculate improvement
  const firstCER = metrics[0]?.cer || 1;
  const lastCER = metrics[metrics.length - 1]?.cer || 1;
  const improvement = ((firstCER - lastCER) / firstCER) * 100;

  return (
    <div className="bg-gradient-to-br from-emerald-50 to-blue-50 rounded-xl shadow p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-slate-800">🏆 Best Metrics Achieved</h2>
        <div className="text-sm text-slate-600">
          {improvement > 0 ? (
            <span className="text-emerald-700 font-semibold">
              ↓ {improvement.toFixed(1)}% improvement
            </span>
          ) : (
            <span className="text-red-700">No improvement</span>
          )}
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-white rounded-lg p-4 border-2 border-emerald-200">
          <div className="text-xs text-slate-600 uppercase">Best CER</div>
          <div className="text-2xl font-bold text-emerald-700">
            {(bestCER * 100).toFixed(2)}%
          </div>
          <div className="text-xs text-slate-500 mt-1">Epoch {bestCEREpoch}</div>
        </div>

        <div className="bg-white rounded-lg p-4 border-2 border-blue-200">
          <div className="text-xs text-slate-600 uppercase">Best WER</div>
          <div className="text-2xl font-bold text-blue-700">
            {(bestWER * 100).toFixed(2)}%
          </div>
          <div className="text-xs text-slate-500 mt-1">Epoch {bestWEREpoch}</div>
        </div>

        <div className="bg-white rounded-lg p-4 border-2 border-purple-200">
          <div className="text-xs text-slate-600 uppercase">Best Exact Match</div>
          <div className="text-2xl font-bold text-purple-700">
            {(bestExact * 100).toFixed(1)}%
          </div>
          <div className="text-xs text-slate-500 mt-1">Epoch {bestExactEpoch}</div>
        </div>

        <div className="bg-white rounded-lg p-4 border-2 border-indigo-200">
          <div className="text-xs text-slate-600 uppercase">Best Valid %</div>
          <div className="text-2xl font-bold text-indigo-700">
            {(bestValid * 100).toFixed(1)}%
          </div>
          <div className="text-xs text-slate-500 mt-1">
            {bestValid === 1 ? '✓ Perfect' : 'Grammar check'}
          </div>
        </div>
      </div>

      {/* Quality Assessment */}
      <div className="mt-4 p-3 bg-white rounded-lg">
        <div className="text-sm font-medium text-slate-700 mb-2">Quality Assessment:</div>
        <div className="flex flex-wrap gap-2">
          {bestCER < 0.05 && (
            <span className="px-3 py-1 bg-emerald-100 text-emerald-800 rounded-full text-xs">
              ✓ Excellent CER (&lt;5%)
            </span>
          )}
          {bestWER < 0.10 && (
            <span className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-xs">
              ✓ Good WER (&lt;10%)
            </span>
          )}
          {bestExact > 0.60 && (
            <span className="px-3 py-1 bg-purple-100 text-purple-800 rounded-full text-xs">
              ✓ Strong Exact Match (&gt;60%)
            </span>
          )}
          {bestCER >= 0.05 && bestCER < 0.15 && (
            <span className="px-3 py-1 bg-amber-100 text-amber-800 rounded-full text-xs">
              ⚠ Moderate CER (5-15%)
            </span>
          )}
          {bestCER >= 0.15 && (
            <span className="px-3 py-1 bg-red-100 text-red-800 rounded-full text-xs">
              ✗ Poor CER (&gt;15%) - Need more training or data
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
