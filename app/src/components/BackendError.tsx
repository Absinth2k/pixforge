import { AlertTriangle } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { useStore } from '@/state/store'

export function BackendError() {
  const { t } = useTranslation()
  const error = useStore((s) => s.backendError)
  const log = useStore((s) => s.backendLog)
  const connect = useStore((s) => s.connect)
  const set = useStore((s) => s.set)
  if (!error) return null
  return (
    <div role="alert" className="absolute inset-0 z-50 grid place-items-center bg-black/60 p-6">
      <div className="w-full max-w-[560px] rounded-dialog border border-line bg-panel p-6 shadow-2xl">
        <div className="flex items-center gap-3">
          <AlertTriangle className="text-danger" size={22} aria-hidden="true" />
          <h2 className="m-0 text-[17px] font-semibold">{t('errors.backendTitle')}</h2>
        </div>
        <p className="mt-3 text-[14px] text-text2">{error}</p>
        {log && (
          <pre className="mt-3 max-h-48 overflow-auto rounded-lg bg-canvas p-3 text-[11px] whitespace-pre-wrap text-text3 select-text">
            {log.slice(-2000)}
          </pre>
        )}
        <div className="mt-4 flex justify-end gap-2">
          <button
            type="button"
            className="h-9 rounded-btn border border-line px-3.5 text-text2 hover:bg-hover"
            onClick={() => window.pixforge.openLogs()}
          >
            {t('errors.logs')}
          </button>
          <button
            type="button"
            className="h-9 rounded-btn bg-accent px-4 font-semibold text-on-accent hover:bg-accent-hover"
            onClick={async () => {
              set({ backendError: null, api: null })
              await window.pixforge.restartBackend()
              await connect()
            }}
          >
            {t('errors.restart')}
          </button>
        </div>
      </div>
    </div>
  )
}
