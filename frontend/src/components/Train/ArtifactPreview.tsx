import { useEffect, useState } from 'react';
import { getArtifacts } from '../../api/experiments';

interface ArtifactPreviewProps {
  runId: string;
  epoch?: number;
}

interface ValidationPrediction {
  id: string;
  target: string;
  pred: string;
  image_path?: string;
}

export function ArtifactPreview({ runId, epoch }: ArtifactPreviewProps) {
  const [artifacts, setArtifacts] = useState<string[]>([]);
  const [predictions, setPredictions] = useState<ValidationPrediction[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    getArtifacts(runId, epoch)
      .then((arts) => {
        setArtifacts(arts);
        // TODO: Fetch actual predictions from artifact files
        // For MVP, just show placeholder
        setPredictions([]);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [runId, epoch]);

  if (loading) {
    return (
      <div className="bg-white rounded-xl shadow p-6 text-center text-slate-400">
        Loading artifacts...
      </div>
    );
  }

  if (artifacts.length === 0) {
    return (
      <div className="bg-white rounded-xl shadow p-6 text-center text-slate-400">
        No artifacts available
      </div>
    );
  }

  return (
    <div className="bg-white rounded-xl shadow p-6 space-y-4">
      <h3 className="text-lg font-semibold text-slate-800">
        Validation Predictions {epoch !== undefined && `(Epoch ${epoch})`}
      </h3>

      {predictions.length > 0 ? (
        <div className="space-y-3">
          {predictions.map((pred) => (
            <div key={pred.id} className="border border-slate-200 rounded-lg p-4">
              <div className="flex items-start gap-4">
                {pred.image_path && (
                  <div className="flex-shrink-0">
                    <img
                      src={pred.image_path}
                      alt={pred.id}
                      className="h-16 border border-slate-200 rounded"
                    />
                  </div>
                )}
                <div className="flex-1 space-y-2">
                  <div className="text-xs text-slate-500">{pred.id}</div>
                  <div className="font-mono text-sm">
                    <div>
                      <span className="text-slate-600">Target:</span> {pred.target}
                    </div>
                    <div>
                      <span className="text-slate-600">Pred:</span> {pred.pred}
                    </div>
                  </div>
                  {pred.target === pred.pred && (
                    <span className="inline-block px-2 py-0.5 bg-emerald-100 text-emerald-700 rounded text-xs">
                      ✓ Exact Match
                    </span>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="text-center text-slate-400 py-8">
          <p>Artifact files found: {artifacts.length}</p>
          <p className="text-sm mt-2">
            (Prediction parsing will be implemented when backend provides JSONL artifacts)
          </p>
          <div className="mt-4 text-left">
            <p className="text-sm font-semibold text-slate-600">Available artifacts:</p>
            <ul className="text-xs text-slate-500 mt-2 space-y-1">
              {artifacts.slice(0, 5).map((art, i) => (
                <li key={i} className="font-mono">
                  {art}
                </li>
              ))}
              {artifacts.length > 5 && (
                <li className="text-slate-400">... and {artifacts.length - 5} more</li>
              )}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}
