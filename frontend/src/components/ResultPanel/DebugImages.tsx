interface DebugImagesProps {
  preprocessedB64?: string;
}

export function DebugImages({ preprocessedB64 }: DebugImagesProps) {
  if (!preprocessedB64) return null;

  return (
    <div className="bg-white rounded-xl shadow p-4">
      <label className="text-sm font-medium text-slate-700 block mb-2">
        Preprocessed Image:
      </label>
      <div className="bg-slate-50 p-2 rounded border border-slate-200">
        <img
          src={preprocessedB64}
          alt="Preprocessed"
          className="max-w-full h-auto"
          style={{ imageRendering: 'pixelated' }}
        />
      </div>
    </div>
  );
}
