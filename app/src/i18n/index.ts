import i18n from 'i18next'
import { initReactI18next } from 'react-i18next'
import de from './de.json'
import en from './en.json'

const lang = (navigator.language || 'de').toLowerCase().startsWith('de') ? 'de' : 'en'

void i18n.use(initReactI18next).init({
  resources: { de: { translation: de }, en: { translation: en } },
  lng: lang,
  fallbackLng: 'de',
  interpolation: { escapeValue: false }
})

document.documentElement.lang = lang
export default i18n
