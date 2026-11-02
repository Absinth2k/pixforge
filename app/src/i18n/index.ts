import i18n from 'i18next'
import { initReactI18next } from 'react-i18next'
import de from './de.json'
import en from './en.json'

export const LANG_KEY = 'pixforge.lang'

function storedLang(): string | null {
  try {
    return localStorage.getItem(LANG_KEY)
  } catch {
    return null
  }
}

/** Chosen in the about dialog, else the Windows language. */
const lang =
  storedLang() ?? ((navigator.language || 'de').toLowerCase().startsWith('de') ? 'de' : 'en')

void i18n.use(initReactI18next).init({
  resources: { de: { translation: de }, en: { translation: en } },
  lng: lang,
  fallbackLng: 'de',
  interpolation: { escapeValue: false }
})

document.documentElement.lang = lang

export function setLanguage(l: 'de' | 'en'): void {
  void i18n.changeLanguage(l)
  document.documentElement.lang = l
  try {
    localStorage.setItem(LANG_KEY, l)
  } catch {
    /* private mode: only for this session */
  }
}

export default i18n
