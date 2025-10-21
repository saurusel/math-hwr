import { useState } from 'react';
import type { SamplePredEvent } from '../../types/training';

interface SampleGalleryProps {
  samples: SamplePredEvent[];
}

export function SampleGallery({ samples }: SampleGalleryProps) {
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);

  if (samples.length === 0) {
    return (
      <div className="bg-white rounded-xl shadow p-6">
        <h2 className="text-lg font-semibold text-slate-700 mb-4">Sample Predictions</h2>
        <div className="text-center text-slate-400 py-12">
          No sample predictions yet. They will appear during training...
        </div>
      </div>
    );
  }

  // Get latest samples
  const latestSample = samples[samples.length - 1];
  const displaySamples = latestSample.items.slice(0, 10); // Show up to 10 samples

  return (
    <div className="bg-white rounded-xl shadow p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-slate-700">
          Sample Predictions (Epoch {latestSample.epoch})
        </h2>
        <div className="text-sm text-slate-600">
          Showing {displaySamples.length} of {latestSample.items.length} samples
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {displaySamples.map((sample, idx) => {
          const isMatch = sample.target === sample.pred;
          const isExpanded = expandedIdx === idx;

          return (
            <div
              key={idx}
              className={`border-2 rounded-lg p-4 transition-all ${
                isMatch
                  ? 'border-emerald-200 bg-emerald-50'
                  : 'border-red-200 bg-red-50'
              } ${isExpanded ? 'col-span-1 md:col-span-2' : ''}`}
            >
              {/* Header with Match Status */}
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-mono text-slate-500">{sample.id}</span>
                  {isMatch ? (
                    <span className="px-2 py-0.5 bg-emerald-600 text-white rounded text-xs font-semibold">
                      ✓ Match
                    </span>
                  ) : (
                    <span className="px-2 py-0.5 bg-red-600 text-white rounded text-xs font-semibold">
                      ✗ Error
                    </span>
                  )}
                </div>
                <button
                  className="text-xs text-slate-600 hover:text-slate-900"
                  onClick={() => setExpandedIdx(isExpanded ? null : idx)}
                >
                  {isExpanded ? '▼ Collapse' : '▶ Expand'}
                </button>
              </div>

              {/* Image if available */}
              {sample.image_b64 && (
                <div className="mb-3 bg-white p-2 rounded border border-slate-200">
                  <img
                    src={sample.image_b64}
                    alt={sample.id}
                    className="w-full h-auto"
                    style={{ imageRendering: 'pixelated', maxHeight: isExpanded ? 'none' : '64px' }}
                  />
                </div>
              )}

              {/* Predictions */}
              <div className="space-y-2 font-mono text-sm">
                <div className="flex items-start gap-2">
                  <span className="text-slate-600 font-semibold min-w-[60px]">Target:</span>
                  <span className="text-slate-900 break-all">{sample.target}</span>
                </div>
                <div className="flex items-start gap-2">
                  <span className="text-slate-600 font-semibold min-w-[60px]">Pred:</span>
                  <span
                    className={`break-all ${
                      isMatch ? 'text-emerald-700 font-semibold' : 'text-red-700'
                    }`}
                  >
                    {sample.pred}
                  </span>
                </div>
              </div>

              {/* Character-level diff if error */}
              {!isMatch && isExpanded && (
                <div className="mt-3 p-3 bg-white rounded border border-slate-200">
                  <div className="text-xs text-slate-600 mb-2">Character Alignment:</div>
                  <div className="font-mono text-xs space-y-1">
                    <div className="flex flex-wrap gap-1">
                      <span className="text-slate-500">Tgt:</span>
                      {sample.target.split(' ').map((char, i) => (
                        <span
                          key={`t-${i}`}
                          className="px-1 bg-slate-100 border border-slate-300 rounded"
                        >
                          {char}
                        </span>
                      ))}
                    </div>
                    <div className="flex flex-wrap gap-1">
                      <span className="text-slate-500">Prd:</span>
                      {sample.pred.split(' ').map((char, i) => (
                        <span
                          key={`p-${i}`}
                          className={`px-1 border rounded ${
                            sample.target.split(' ')[i] === char
                              ? 'bg-emerald-100 border-emerald-300'
                              : 'bg-red-100 border-red-300'
                          }`}
                        >
                          {char}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Statistics */}
      <div className="mt-4 p-3 bg-slate-50 rounded-lg">
        <div className="text-sm text-slate-700">
          <strong>Accuracy:</strong>{' '}
          {latestSample.items.length > 0
            ? `${((latestSample.items.filter((s) => s.target === s.pred).length / latestSample.items.length) * 100).toFixed(1)}%`
            : 'N/A'}{' '}
          ({latestSample.items.filter((s) => s.target === s.pred).length} / {latestSample.items.length} correct)
        </div>
      </div>
    </div>
  );
}
