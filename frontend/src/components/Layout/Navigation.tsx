import { Link, useLocation } from 'react-router-dom';

export function Navigation() {
  const location = useLocation();

  const isActive = (path: string) => {
    if (path === '/') {
      return location.pathname === '/';
    }
    return location.pathname.startsWith(path);
  };

  const linkClass = (path: string) =>
    `px-4 py-2 rounded-lg transition-colors ${
      isActive(path)
        ? 'bg-slate-900 text-white'
        : 'text-slate-700 hover:bg-slate-100'
    }`;

  return (
    <nav className="bg-white border-b border-slate-200 shadow-sm">
      <div className="max-w-7xl mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center gap-8">
            <Link to="/" className="text-xl font-bold text-slate-800">
              Math HWR
            </Link>
            <div className="flex gap-2">
              <Link to="/" className={linkClass('/')}>
                🎨 Распознавание
              </Link>
              <Link to="/train/experiments" className={linkClass('/train')}>
                🧪 Обучение
              </Link>
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
}
