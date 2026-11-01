import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useStore } from '@/state/store'
import { Logo, nf } from './ui'

export function TitleBar() {
  const { t, i18n } = useTranslation()
  const img = useStore((s) => s.images.find((i) => i.id === s.selectedId))
  const [maximized, setMaximized] = useState(false)

  useEffect(() => {
    void window.pixforge.window.isMaximized().then(setMaximized)
    return window.pixforge.window.onMaximizedChange(setMaximized)
  }, [])

  const btn =
    'no-drag grid h-9 w-[46px] place-items-center text-text2 transition-colors duration-100 hover:bg-hover'
  return (
    <div
      className="drag flex h-9 shrink-0 items-center gap-2.5 border-b border-line bg-app pl-3.5"
      onDoubleClick={() => window.pixforge.window.toggleMaximize()}
    >
      <Logo />
      <span className="text-[12.5px] font-semibold tracking-[-0.01em]">{t('app.name')}</span>
      <span className="flex-1 truncate text-center text-[12px] text-text3">
        {img
          ? `${img.name} · ${nf(img.width, i18n.language)} × ${nf(img.height, i18n.language)}`
          : ''}
      </span>
      <div className="flex h-9">
        <button
          type="button"
          className={btn}
          aria-label={t('window.minimize')}
          onClick={() => window.pixforge.window.minimize()}
        >
          <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
            <path d="M1 6h10" stroke="currentColor" strokeWidth="1.2" />
          </svg>
        </button>
        <button
          type="button"
          className={btn}
          aria-label={maximized ? t('window.restore') : t('window.maximize')}
          onClick={() => window.pixforge.window.toggleMaximize()}
        >
          <svg
            width="12"
            height="12"
            viewBox="0 0 12 12"
            aria-hidden="true"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.2"
          >
            {maximized ? (
              <>
                <rect x="1.5" y="3.5" width="7" height="7" />
                <path d="M3.5 3.5V1.5h7v7h-2" />
              </>
            ) : (
              <rect x="1.5" y="1.5" width="9" height="9" />
            )}
          </svg>
        </button>
        <button
          type="button"
          className={btn + ' hover:!bg-[#C42B2B] hover:text-white'}
          aria-label={t('window.close')}
          onClick={() => window.pixforge.window.close()}
        >
          <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
            <path d="M1.5 1.5l9 9M10.5 1.5l-9 9" stroke="currentColor" strokeWidth="1.2" />
          </svg>
        </button>
      </div>
    </div>
  )
}
