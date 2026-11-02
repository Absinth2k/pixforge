import { Download } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import type { DownloadState, ModelInfo } from '@/api/types'
import { useStore } from '@/state/store'
import { Logo } from './ui'

export function formatBytes(n: number, lang: string): string {
  const mb = n / 2 ** 20
  if (mb < 1) return `${Math.max(1, Math.round(n / 1024)).toLocaleString(lang)} KB`
  return mb >= 1000
    ? `${(mb / 1024).toLocaleString(lang, { maximumFractionDigits: 1 })} GB`
    : `${Math.round(mb).toLocaleString(lang)} MB`
}

/** Bytes still to fetch and progress of a set of models. */
export function downloadProgress(
  ids: string[],
  models: ModelInfo[],
  downloads: Record<string, DownloadState>
): { total: number; done: number; error: string | null; active: boolean } {
  let total = 0
  let done = 0
  let error: string | null = null
  let active = false
  for (const id of ids) {
    const m = models.find((x) => x.id === id)
    const d = downloads[id]
    const size = d?.total ?? m?.needed?.size ?? 0
    total += size
    done += d ? (d.state === 'done' ? size : d.done) : 0
    if (d?.state === 'error') error = d.error
    if (d && (d.state === 'queued' || d.state === 'running')) active = true
  }
  return { total, done, error, active }
}

/** First start: the default models are downloaded before anything else can happen. */
export function ModelSetup() {
  const { t, i18n } = useTranslation()
  const health = useStore((s) => s.health)
  const models = useStore((s) => s.models)
  const downloads = useStore((s) => s.downloads)
  const downloadModels = useStore((s) => s.downloadModels)
  const missing = health?.missing_defaults ?? []
  if (!missing.length) return null
  const lang = i18n.language
  const p = downloadProgress(missing, models, downloads)
  const pct = p.total ? Math.floor((100 * p.done) / p.total) : 0
  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="setup-title"
      data-testid="model-setup"
      className="absolute inset-0 z-50 grid place-items-center bg-canvas/95 p-6"
    >
      <div className="w-full max-w-[520px] rounded-dialog border border-line bg-panel p-7 shadow-2xl">
        <div className="flex items-center gap-3">
          <Logo size={30} />
          <h2 id="setup-title" className="m-0 text-[18px] font-semibold">
            {t('models.setupTitle')}
          </h2>
        </div>
        <p className="mt-3 text-[14px] leading-relaxed text-text2">
          {t('models.setupText', { size: formatBytes(p.total, lang) })}
        </p>
        <ul className="mt-4 space-y-2.5 p-0">
          {missing.map((id) => {
            const m = models.find((x) => x.id === id)
            const d = downloads[id]
            const size = d?.total ?? m?.needed?.size ?? 0
            const frac = d ? (d.state === 'done' ? 1 : size ? d.done / size : 0) : 0
            return (
              <li key={id} className="list-none">
                <div className="flex justify-between text-[12.5px] text-text2">
                  <span>{m?.name[lang === 'de' ? 'de' : 'en'] ?? id}</span>
                  <span className="text-text3">{formatBytes(size, lang)}</span>
                </div>
                <div className="mt-1 h-1.5 overflow-hidden rounded-sm bg-track">
                  <div
                    className="h-1.5 bg-accent transition-[width] duration-200"
                    style={{ width: `${Math.round(frac * 100)}%` }}
                  />
                </div>
              </li>
            )
          })}
        </ul>
        <div className="mt-5 flex items-center justify-between gap-3">
          <span className="text-[13px] text-text2" role="status">
            {p.error ? (
              <span className="text-danger">
                {t('models.setupError')} {p.error}
              </span>
            ) : (
              t('models.downloading', { pct })
            )}
          </span>
          {p.error && !p.active && (
            <button
              type="button"
              onClick={() => void downloadModels(missing)}
              className="flex h-9 items-center gap-2 rounded-btn bg-accent px-4 font-semibold text-on-accent hover:bg-accent-hover"
            >
              <Download size={15} aria-hidden="true" />
              {t('models.retry')}
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
