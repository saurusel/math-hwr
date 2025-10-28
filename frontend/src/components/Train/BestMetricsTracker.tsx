import type { MetricEvent } from '../../types/training';

interface BestMetricsTrackerProps {
  metrics: MetricEvent[];
}

export function BestMetricsTracker({ metrics }: BestMetricsTrackerProps) {
  if (metrics.length === 0) return null;

  // Detect model type
  const isM2 = metrics.some((m) => m.train_char_acc !== undefined);

  // M1 metrics
  const bestCER = Math.min(...metrics.map((m) => m.cer !== undefined ? m.cer : 1));
  const bestExact = Math.max(...metrics.map((m) => m.exact !== undefined ? m.exact : 0));
  const bestValid = Math.max(...metrics.map((m) => m.valid !== undefined ? m.valid : 0));

  // M2 metrics
  const bestCharAcc = Math.max(...metrics.map((m) => m.val_char_acc !== undefined ? m.val_char_acc : 0));
  const bestSeqAcc = Math.max(...metrics.map((m) => m.val_seq_acc !== undefined ? m.val_seq_acc : 0));
  const bestSegAcc = Math.max(...metrics.map((m) => m.val_seg_acc !== undefined ? m.val_seg_acc : 0));
  const bestTrainCharAcc = Math.max(...metrics.map((m) => m.train_char_acc !== undefined ? m.train_char_acc : 0));

  // Find epochs where best was achieved
  const bestCEREpoch = metrics.find((m) => m.cer === bestCER)?.epoch || 0;
  const bestExactEpoch = metrics.find((m) => m.exact === bestExact)?.epoch || 0;
  const bestCharAccEpoch = metrics.find((m) => m.val_char_acc === bestCharAcc)?.epoch || 0;
  const bestSeqAccEpoch = metrics.find((m) => m.val_seq_acc === bestSeqAcc)?.epoch || 0;
  const bestSegAccEpoch = metrics.find((m) => m.val_seg_acc === bestSegAcc)?.epoch || 0;

  // Calculate improvement
  const firstMetric = isM2 ? (metrics[0]?.val_seq_acc || 0) : (metrics[0]?.cer || 1);
  const lastMetric = isM2 ? (metrics[metrics.length - 1]?.val_seq_acc || 0) : (metrics[metrics.length - 1]?.cer || 1);
  const improvement = isM2
    ? ((lastMetric - firstMetric) / 100) * 100  // For accuracy (higher is better)
    : ((firstMetric - lastMetric) / firstMetric) * 100;  // For error rate (lower is better)

  return (
    <div className="bg-gradient-to-br from-emerald-50 to-blue-50 rounded-xl shadow p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-slate-800">
          🏆 Лучшие достигнутые метрики {isM2 ? '(Модель M2)' : '(Модель M1)'}
        </h2>
        <div className="text-sm text-slate-600">
          {improvement > 0 ? (
            <span className="text-emerald-700 font-semibold">
              {isM2 ? '↑' : '↓'} {improvement.toFixed(1)}% улучшение
            </span>
          ) : (
            <span className="text-red-700">Нет улучшения</span>
          )}
        </div>
      </div>

      {/* M1 Metrics */}
      {!isM2 && (
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          <div className="bg-white rounded-lg p-4 border-2 border-emerald-200">
            <div className="text-xs text-slate-600 uppercase">Лучший CER</div>
            <div className="text-2xl font-bold text-emerald-700">
              {(bestCER * 100).toFixed(2)}%
            </div>
            <div className="text-xs text-slate-500 mt-1">Эпоха {bestCEREpoch}</div>
          </div>

          <div className="bg-white rounded-lg p-4 border-2 border-purple-200">
            <div className="text-xs text-slate-600 uppercase">Лучшее точное совпадение</div>
            <div className="text-2xl font-bold text-purple-700">
              {(bestExact * 100).toFixed(1)}%
            </div>
            <div className="text-xs text-slate-500 mt-1">Эпоха {bestExactEpoch}</div>
          </div>

          <div className="bg-white rounded-lg p-4 border-2 border-indigo-200">
            <div className="text-xs text-slate-600 uppercase">Лучшая валидность %</div>
            <div className="text-2xl font-bold text-indigo-700">
              {(bestValid * 100).toFixed(1)}%
            </div>
            <div className="text-xs text-slate-500 mt-1">
              {bestValid === 1 ? '✓ Идеально' : 'Проверка грамматики'}
            </div>
          </div>
        </div>
      )}

      {/* M2 Metrics */}
      {isM2 && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-white rounded-lg p-4 border-2 border-emerald-200">
            <div className="text-xs text-slate-600 uppercase">Лучшая точн. символов</div>
            <div className="text-2xl font-bold text-emerald-700">
              {bestCharAcc.toFixed(1)}%
            </div>
            <div className="text-xs text-slate-500 mt-1">Эпоха {bestCharAccEpoch}</div>
          </div>

          <div className="bg-white rounded-lg p-4 border-2 border-blue-200">
            <div className="text-xs text-slate-600 uppercase">Лучшая точн. последов.</div>
            <div className="text-2xl font-bold text-blue-700">
              {bestSeqAcc.toFixed(1)}%
            </div>
            <div className="text-xs text-slate-500 mt-1">Эпоха {bestSeqAccEpoch}</div>
          </div>

          <div className="bg-white rounded-lg p-4 border-2 border-purple-200">
            <div className="text-xs text-slate-600 uppercase">Лучшая точн. сегментации</div>
            <div className="text-2xl font-bold text-purple-700">
              {bestSegAcc.toFixed(1)}%
            </div>
            <div className="text-xs text-slate-500 mt-1">Эпоха {bestSegAccEpoch}</div>
          </div>

          <div className="bg-white rounded-lg p-4 border-2 border-indigo-200">
            <div className="text-xs text-slate-600 uppercase">Точн. симв. (обуч.)</div>
            <div className="text-2xl font-bold text-indigo-700">
              {bestTrainCharAcc.toFixed(1)}%
            </div>
            <div className="text-xs text-slate-500 mt-1">
              Последняя эпоха
            </div>
          </div>
        </div>
      )}

      {/* Quality Assessment */}
      <div className="mt-4 p-3 bg-white rounded-lg">
        <div className="text-sm font-medium text-slate-700 mb-2">Оценка качества:</div>
        <div className="flex flex-wrap gap-2">
          {/* M1 Quality Assessment */}
          {!isM2 && (
            <>
              {bestCER < 0.05 && (
                <span className="px-3 py-1 bg-emerald-100 text-emerald-800 rounded-full text-xs">
                  ✓ Отличный CER (&lt;5%)
                </span>
              )}
              {bestExact > 0.60 && (
                <span className="px-3 py-1 bg-purple-100 text-purple-800 rounded-full text-xs">
                  ✓ Хорошее точное совпадение (&gt;60%)
                </span>
              )}
              {bestCER >= 0.05 && bestCER < 0.15 && (
                <span className="px-3 py-1 bg-amber-100 text-amber-800 rounded-full text-xs">
                  ⚠ Средний CER (5-15%)
                </span>
              )}
              {bestCER >= 0.15 && (
                <span className="px-3 py-1 bg-red-100 text-red-800 rounded-full text-xs">
                  ✗ Плохой CER (&gt;15%) - Нужно больше обучения или данных
                </span>
              )}
            </>
          )}

          {/* M2 Quality Assessment */}
          {isM2 && (
            <>
              {bestCharAcc >= 98 && (
                <span className="px-3 py-1 bg-emerald-100 text-emerald-800 rounded-full text-xs">
                  ✓ Отличная классификация символов (&gt;98%)
                </span>
              )}
              {bestSeqAcc >= 70 && (
                <span className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-xs">
                  ✓ Хорошая точность последовательности (&gt;70%)
                </span>
              )}
              {bestSegAcc >= 70 && (
                <span className="px-3 py-1 bg-purple-100 text-purple-800 rounded-full text-xs">
                  ✓ Хорошая сегментация (&gt;70%)
                </span>
              )}
              {bestSeqAcc >= 50 && bestSeqAcc < 70 && (
                <span className="px-3 py-1 bg-amber-100 text-amber-800 rounded-full text-xs">
                  ⚠ Средняя точн. посл. (50-70%) - Проблемы сегментации
                </span>
              )}
              {bestSeqAcc < 50 && (
                <span className="px-3 py-1 bg-red-100 text-red-800 rounded-full text-xs">
                  ✗ Плохая точн. посл. (&lt;50%) - Нужна лучшая сегментация
                </span>
              )}
              {bestCharAcc < 95 && (
                <span className="px-3 py-1 bg-red-100 text-red-800 rounded-full text-xs">
                  ✗ Плохая точн. симв. (&lt;95%) - Классификатору нужно больше обучения
                </span>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
