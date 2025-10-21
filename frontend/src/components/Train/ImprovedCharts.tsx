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
        No metrics yet. Training will start soon...
      </div>
    );
  }

  // Calculate best metrics
  const bestCER = Math.min(...metrics.map((m) => m.cer || 1));
  const bestWER = Math.min(...metrics.map((m) => m.wer || 1));
  const bestExact = Math.max(...metrics.map((m) => m.exact || 0));

  return (
    <div className="space-y-6">
      {/* Loss Curves with Enhanced Features */}
      <div className="bg-white rounded-xl shadow p-6">
        <h2 className="text-lg font-semibold text-slate-700 mb-4">Loss Curves</h2>
        <ResponsiveContainer width="100%" height={350}>
          <LineChart data={metrics} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis
              dataKey="epoch"
              label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }}
              stroke="#64748b"
            />
            <YAxis
              label={{ value: 'Loss', angle: -90, position: 'insideLeft' }}
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
              name="Train Loss"
              dot={false}
              activeDot={{ r: 6 }}
            />
            <Line
              type="monotone"
              dataKey="val_loss"
              stroke="#3b82f6"
              strokeWidth={2}
              name="Val Loss"
              dot={false}
              activeDot={{ r: 6 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Accuracy Metrics (CER, WER, Exact Match) */}
      <div className="bg-white rounded-xl shadow p-6">
        <h2 className="text-lg font-semibold text-slate-700 mb-4">
          Accuracy Metrics (Best: CER={(bestCER * 100).toFixed(2)}%, Exact={(bestExact * 100).toFixed(1)}%)
        </h2>
        <ResponsiveContainer width="100%" height={350}>
          <LineChart data={metrics} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis
              dataKey="epoch"
              label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }}
              stroke="#64748b"
            />
            <YAxis
              label={{ value: 'Error Rate (%)', angle: -90, position: 'insideLeft' }}
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
            <ReferenceLine y={0.05} stroke="#10b981" strokeDasharray="3 3" label="Good (5%)" />

            <Line
              type="monotone"
              dataKey="cer"
              stroke="#10b981"
              strokeWidth={2}
              name="CER (Character Error Rate)"
              dot={false}
              activeDot={{ r: 6 }}
            />
            <Line
              type="monotone"
              dataKey="wer"
              stroke="#3b82f6"
              strokeWidth={2}
              name="WER (Word Error Rate)"
              dot={false}
              activeDot={{ r: 6 }}
            />
            <Line
              type="monotone"
              dataKey="exact"
              stroke="#8b5cf6"
              strokeWidth={2}
              name="Exact Match %"
              dot={false}
              activeDot={{ r: 6 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Learning Rate Schedule */}
      <div className="bg-white rounded-xl shadow p-6">
        <h2 className="text-lg font-semibold text-slate-700 mb-4">Learning Rate Schedule</h2>
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={metrics} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis
              dataKey="epoch"
              label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }}
              stroke="#64748b"
            />
            <YAxis
              label={{ value: 'Learning Rate', angle: -90, position: 'insideLeft' }}
              stroke="#64748b"
              scale="log"
              domain={['auto', 'auto']}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#ffffff',
                border: '1px solid #e2e8f0',
                borderRadius: '8px',
              }}
              formatter={(value: number) => value.toExponential(2)}
            />
            <Line
              type="monotone"
              dataKey="lr"
              stroke="#f59e0b"
              strokeWidth={2}
              name="Learning Rate"
              dot={false}
              activeDot={{ r: 6 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Training Speed (if available) */}
      {metrics.some((m) => m.epoch_time_sec) && (
        <div className="bg-white rounded-xl shadow p-6">
          <h2 className="text-lg font-semibold text-slate-700 mb-4">Training Speed</h2>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={metrics} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis
                dataKey="epoch"
                label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }}
                stroke="#64748b"
              />
              <YAxis
                label={{ value: 'Time (seconds)', angle: -90, position: 'insideLeft' }}
                stroke="#64748b"
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#ffffff',
                  border: '1px solid #e2e8f0',
                  borderRadius: '8px',
                }}
                formatter={(value: number) => `${value.toFixed(1)}s`}
              />
              <Line
                type="monotone"
                dataKey="epoch_time_sec"
                stroke="#06b6d4"
                strokeWidth={2}
                name="Epoch Duration"
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
