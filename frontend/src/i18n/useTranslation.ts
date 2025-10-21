import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { translations, Locale, TranslationKey } from './translations';

interface I18nState {
  locale: Locale;
  setLocale: (locale: Locale) => void;
  t: (key: TranslationKey) => string;
}

export const useI18n = create<I18nState>()(
  persist(
    (set, get) => ({
      locale: 'en',
      setLocale: (locale) => set({ locale }),
      t: (key) => {
        const { locale } = get();
        return translations[locale][key] || key;
      },
    }),
    {
      name: 'math-hwr-locale',
    }
  )
);

export function useTranslation() {
  const { locale, setLocale, t } = useI18n();
  return { locale, setLocale, t };
}
