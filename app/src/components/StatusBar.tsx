import { useTranslation } from 'react-i18next'
import { useStore } from '@/state/store'
import { formatDuration } from './ui'

const ACTIVE = new Set(['running', 'queued'])

export function StatusBar() {
  const { t, i18n } = useTranslation()
  const api = useStore((s) => s.api)
  const health = useStore((s) => s.health)
  const jobs = useStore((s) => s.jobs)
  const update = useStore((s) => s.update)
  const toast = useStore((s) => s.toast)
  const set = useStore((s) => s.set)

  const all = Object.values(jobs)
  const running = all.find((j) => j.status === 'running')
  // "image n of m" counts the jobs of the current run only (see store.batchIds)
  const batchIds = useStore((s) => s.batchIds)
  const batch = batchIds.map((id) => jobs[id]).filter(Boolean)
  const batchTotal = batch.length
  const batchDone = batch.filter((j) => !ACTIVE.has(j.status)).length
  const dev = health?.device
  const deviceText = !api
    ? t('status.starting')
    : dev
      ? dev.kind === 'gpu'
        ? `${dev.name} · DirectML · FP16${dev.vram_mb ? ` · ${(dev.vram_mb / 1024).toFixed(0)} GB` : ''}`
        : `${t('status.noGpu')} (${dev.name})`
      : ''

  return (
    <div className="flex h-[26px] shrink-0 items-center gap-4 overflow-hidden border-t border-line bg-app px-3.5 text-[11.5px] whitespace-nowrap text-text3">
      <span className="flex items-center gap-1.5">
        <span
          className={`h-[7px] w-[7px] rounded-full ${api ? (dev?.kind === 'gpu' ? 'bg-success' : 'bg-accent') : 'bg-text3'}`}
        />
        {deviceText}
      </span>
      {toast && (
        <span
          className={`flex min-w-0 items-center gap-2 ${toast.kind === 'error' ? 'text-danger' : 'text-text2'}`}
        >
          <span className="truncate">
            {toast.path ? t('status.exported', { name: toast.text }) : toast.text}
          </span>
          {toast.path && (
            <button
              type="button"
              className="text-accent-ink underline-offset-2 hover:underline"
              onClick={() => window.pixforge.showInFolder(toast.path!)}
            >
              {t('status.showInFolder')}
            </button>
          )}
          <button
            type="button"
            aria-label="OK"
            className="text-text3 hover:text-text"
            onClick={() => set({ toast: null })}
          >
            ×
          </button>
        </span>
      )}
      <span className="flex-1" />
      {running && (
        <>
          <span>
            {t('status.job', { n: batchDone + 1, total: batchTotal })}
            {`: ${running.name}${running.crop ? ` (${t('status.crop')})` : ''}`}
            {running.eta_s != null
              ? ` · ${t('status.remaining', { t: formatDuration(running.eta_s, i18n.language) })}`
              : ''}
          </span>
          <span className="h-1 w-[120px] overflow-hidden rounded-sm bg-track">
            <span
              className="block h-1 bg-accent transition-[width] duration-200"
              style={{ width: `${Math.round(running.progress * 100)}%` }}
            />
          </span>
        </>
      )}
      {update?.phase === 'ready' && (
        <button
          type="button"
          onClick={() => void window.pixforge.update.install()}
          className="rounded-btn border border-accent-line bg-accent-soft px-2 py-0.5 text-[11.5px] text-text"
        >
          {t('about.readyShort', { version: update.version })}
        </button>
      )}
      <span className="border-l border-line pl-3.5">{t('app.copyright')}</span>
    </div>
  )
}
