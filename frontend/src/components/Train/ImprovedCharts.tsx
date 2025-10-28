import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Brush,
  ReferenceLine,
} from 'recharts';
import type { MetricEvent } from '../../types/training';

interface ImprovedChartsProps {
  metrics: MetricEvent[];
}

export function ImprovedCharts({ metrics }: ImprovedChartsProps) {
  if (metrics.length === 0) {
    return (
      <div className="text-center text-slate-400 py-12">
        Метрики пока отсутствуют. Обучение скоро начнется...
      </div>
    );
  }

  // Detect model type from metrics
  const isM2 = metrics.some((m) => m.train_char_acc !== undefined);

  // Calculate best metrics
  const bestCER = Math.min(...metrics.map((m) => m.cer || 1));
  const bestExact = Math.max(...metrics.map((m) => m.exact || 0));

  // M2 metrics
  const bestCharAcc = Math.max(...metrics.map((m) => m.val_char_acc || 0));
  const bestSeqAcc = Math.max(...metrics.map((m) => m.val_seq_acc || 0));
  const bestSegAcc = Math.max(...metrics.map((m) => m.val_seg_acc || 0));

  return (
    <div className="space-y-6">
      {/* Loss Curves with Enhanced Features */}
      <div className="bg-white rounded-xl shadow p-6">
        <h2 className="text-lg font-semibold text-slate-700 mb-4">Кривые потерь</h2>
        <ResponsiveContainer width="100%" height={350}>
          <LineChart data={metrics} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis
              dataKey="epoch"
              label={{ value: 'Эпоха', position: 'insideBottom', offset: -5 }}
              stroke="#64748b"
            />
            <YAxis
              label={{ value: 'Потери', angle: -90, position: 'insideLeft' }}
              stroke="#64748b"
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#ffffff',
                border: '1px solid #e2e8f0',
                borderRadius: '8px',
              }}
            />
            <Legend wrapperStyle={{ paddingTop: '10px' }} />
            <Brush dataKey="epoch" height={30} stroke="#8b5cf6" />
            <Line
              type="monotone"
              dataKey="train_loss"
              stroke="#f59e0b"
              strokeWidth={2}
              name="Потери (обуч.)"
              dot={false}
              activeDot={{ r: 6 }}
            />
            <Line
              type="monotone"
              dataKey="val_loss"
              stroke="#3b82f6"
              strokeWidth={2}
              name="Потери (вал.)"
              dot={false}
              activeDot={{ r: 6 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* M1 Accuracy Metrics (CER, WER, Exact Match) */}
      {!isM2 && (
        <div className="bg-white rounded-xl shadow p-6">
          <h2 className="text-lg font-semibold text-slate-700 mb-4">
            M1 Метрики точности (Лучший: CER={(bestCER * 100).toFixed(2)}%, Точное={(bestExact * 100).toFixed(1)}%)
          </h2>
        <ResponsiveContainer width="100%" height={350}>
          <LineChart data={metrics} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis
              dataKey="epoch"
              label={{ value: 'Эпоха', position: 'insideBottom', offset: -5 }}
              stroke="#64748b"
            />
            <YAxis
              label={{ value: 'Процент ошибок', angle: -90, position: 'insideLeft' }}
              domain={[0, 1]}
              tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
              stroke="#64748b"
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#ffffff',
                border: '1px solid #e2e8f0',
                borderRadius: '8px',
              }}
              formatter={(value: number) => `${(value * 100).toFixed(2)}%`}
            />
            <Legend wrapperStyle={{ paddingTop: '10px' }} />
            <Brush dataKey="epoch" height={30} stroke="#8b5cf6" />

            {/* Reference line at 5% (good CER threshold) */}
            <ReferenceLine y={0.05} stroke="#10b981" strokeDasharray="3 3" label="Хорошо (5%)" />

            <Line
              type="monotone"
              dataKey="cer"
              stroke="#10b981"
              strokeWidth={2}
              name="CER (ошибка символов)"
              dot={false}
              activeDot={{ r: 6 }}
            />
            <Line
              type="monotone"
              dataKey="exact"
              stroke="#8b5cf6"
              strokeWidth={2}
              name="Точное совпадение %"
              dot={false}
              activeDot={{ r: 6 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      )}

      {/* M2 Accuracy Metrics (Character, Sequence, Segmentation) */}
      {isM2 && (
        <div className="bg-white rounded-xl shadow p-6">
          <h2 className="text-lg font-semibold text-slate-700 mb-4">
            M2 Метрики точности (Лучший: Симв={bestCharAcc.toFixed(1)}%, Посл={bestSeqAcc.toFixed(1)}%, Сегм={bestSegAcc.toFixed(1)}%)
          </h2>
          <ResponsiveContainer width="100%" height={350}>
            <LineChart data={metrics} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis
                dataKey="epoch"
                label={{ value: 'Эпоха', position: 'insideBottom', offset: -5 }}
                stroke="#64748b"
              />
              <YAxis
                label={{ value: 'Точность (%)', angle: -90, position: 'insideLeft' }}
                domain={[0, 100]}
                stroke="#64748b"
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#ffffff',
                  border: '1px solid #e2e8f0',
                  borderRadius: '8px',
                }}
                formatter={(value: number) => `${value.toFixed(2)}%`}
              />
              <Legend wrapperStyle={{ paddingTop: '10px' }} />
              <Brush dataKey="epoch" height={30} stroke="#8b5cf6" />

              {/* Reference line at 95% (good threshold) */}
              <ReferenceLine y={95} stroke="#10b981" strokeDasharray="3 3" label="Отлично (95%)" />
              <ReferenceLine y={80} stroke="#f59e0b" strokeDasharray="3 3" label="Хорошо (80%)" />

              <Line
                type="monotone"
                dataKey="train_char_acc"
                stroke="#f59e0b"
                strokeWidth={2}
                name="Точн. симв. (обуч.)"
                dot={false}
                activeDot={{ r: 6 }}
              />
              <Line
                type="monotone"
                dataKey="val_char_acc"
                stroke="#10b981"
                strokeWidth={2}
                name="Точн. симв. (вал.)"
                dot={false}
                activeDot={{ r: 6 }}
              />
              <Line
                type="monotone"
                dataKey="val_seq_acc"
                stroke="#3b82f6"
                strokeWidth={2}
                name="Точн. посл. (вал.)"
                dot={false}
                activeDot={{ r: 6 }}
              />
              <Line
                type="monotone"
                dataKey="val_seg_acc"
                stroke="#8b5cf6"
                strokeWidth={2}
                name="Точн. сегм. (вал.)"
                dot={false}
                activeDot={{ r: 6 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Training Speed (if available) */}
      {metrics.some((m) => m.epoch_time_sec) && (
        <div className="bg-white rounded-xl shadow p-6">
          <h2 className="text-lg font-semibold text-slate-700 mb-4">Скорость обучения</h2>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={metrics} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis
                dataKey="epoch"
                label={{ value: 'Эпоха', position: 'insideBottom', offset: -5 }}
                stroke="#64748b"
              />
              <YAxis
                label={{ value: 'Время (секунды)', angle: -90, position: 'insideLeft' }}
                stroke="#64748b"
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#ffffff',
                  border: '1px solid #e2e8f0',
                  borderRadius: '8px',
                }}
                formatter={(value: number) => `${value.toFixed(1)}с`}
              />
              <Line
                type="monotone"
                dataKey="epoch_time_sec"
                stroke="#06b6d4"
                strokeWidth={2}
                name="Длительность эпохи"
                dot={false}
                activeDot={{ r: 6 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
