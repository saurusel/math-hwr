import { useTranslation } from '../../i18n/useTranslation';

export function LanguageSwitcher() {
  const { locale, setLocale } = useTranslation();

  return (
    <div className="flex items-center gap-2">
      <button
        className={`px-3 py-1 rounded text-sm ${
          locale === 'en'
            ? 'bg-slate-900 text-white'
            : 'text-slate-600 hover:bg-slate-100'
        }`}
        onClick={() => setLocale('en')}
      >
        EN
      </button>
      <button
        className={`px-3 py-1 rounded text-sm ${
          locale === 'ru'
            ? 'bg-slate-900 text-white'
            : 'text-slate-600 hover:bg-slate-100'
        }`}
        onClick={() => setLocale('ru')}
      >
        RU
      </button>
    </div>
  );
}
