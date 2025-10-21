import { useEffect } from 'react';

export type ToastType = 'success' | 'error' | 'info' | 'warning';

interface ToastProps {
  message: string;
  type: ToastType;
  onClose: () => void;
  duration?: number;
}

export function Toast({ message, type, onClose, duration = 5000 }: ToastProps) {
  useEffect(() => {
    if (duration > 0) {
      const timer = setTimeout(onClose, duration);
      return () => clearTimeout(timer);
    }
  }, [duration, onClose]);

  const colors = {
    success: 'bg-emerald-50 border-emerald-300 text-emerald-800',
    error: 'bg-red-50 border-red-300 text-red-800',
    info: 'bg-blue-50 border-blue-300 text-blue-800',
    warning: 'bg-amber-50 border-amber-300 text-amber-800',
  };

  const icons = {
    success: '✓',
    error: '✕',
    info: 'ℹ',
    warning: '⚠',
  };

  return (
    <div
      className={`fixed top-4 right-4 max-w-md p-4 rounded-lg border shadow-lg z-50 ${colors[type]} animate-slide-in`}
    >
      <div className="flex items-start gap-3">
        <span className="text-xl">{icons[type]}</span>
        <div className="flex-1 break-words">{message}</div>
        <button
          className="text-xl leading-none opacity-50 hover:opacity-100"
          onClick={onClose}
        >
          ×
        </button>
      </div>
    </div>
  );
}
