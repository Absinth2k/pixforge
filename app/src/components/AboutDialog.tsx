import { useEffect } from 'react'
import { X } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { setLanguage } from '@/i18n'
import { useStore } from '@/state/store'
import { Logo, Segmented, Switch } from './ui'

const REPO = 'https://github.com/Absinth2k/pixforge'

/** Version, automatic updates, device, licences. Opened from the title bar. */
export function AboutDialog() {
  const { t, i18n } = useTranslation()
  const open = useStore((s) => s.aboutOpen)
  const update = useStore((s) => s.update)
  const health = useStore((s) => s.health)
  const set = useStore((s) => s.set)

  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && set({ aboutOpen: false })
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, set])

  if (!open) return null
  const u = update
  const status = !u
    ? ''
    : u.phase === 'dev'
      ? t('about.dev')
      : u.phase === 'disabled'
        ? t('about.off')
        : u.phase === 'checking'
          ? t('about.checking')
          : u.phase === 'none'
            ? t('about.upToDate')
            : u.phase === 'downloading'
              ? t('about.downloading', { version: u.version, pct: u.percent ?? 0 })
              : u.phase === 'ready'
                ? t('about.ready', { version: u.version })
                : u.phase === 'error'
                  ? t('about.error', { msg: u.error ?? '' })
                  : ''
  const link = (url: string, label: string) => (
    <button
      type="button"
      className="text-accent-ink underline-offset-2 hover:underline"
      onClick={() => window.pixforge.openExternal(url)}
    >
      {label}
    </button>
  )
  return (
    <div
      className="absolute inset-0 z-50 grid place-items-center bg-black/50 p-6"
      onPointerDown={(e) => e.target === e.currentTarget && set({ aboutOpen: false })}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="about-title"
        className="relative w-full max-w-[460px] rounded-dialog border border-line bg-panel p-6 shadow-2xl"
      >
        <button
          type="button"
          aria-label={t('about.close')}
          onClick={() => set({ aboutOpen: false })}
          className="absolute top-3 right-3 grid h-8 w-8 place-items-center rounded-btn text-text2 hover:bg-hover"
        >
          <X size={16} aria-hidden="true" />
        </button>
        <div className="flex items-center gap-3">
          <Logo size={34} />
          <div>
            <h2 id="about-title" className="m-0 text-[18px] font-semibold">
              Pixforge
            </h2>
            <p className="m-0 text-[12.5px] text-text3" data-testid="about-version">
              {t('about.version', { version: u?.current || '–' })}
            </p>
          </div>
        </div>
        <p className="mt-4 text-[13px] leading-relaxed text-text2">{t('about.text')}</p>

        <div className="mt-4 flex items-center justify-between gap-3">
          <span className="text-[13px] font-medium">{t('about.language')}</span>
          <Segmented<'de' | 'en'>
            label={t('about.language')}
            value={i18n.language === 'en' ? 'en' : 'de'}
            onChange={(l) => setLanguage(l)}
            options={[
              { value: 'de', label: 'Deutsch' },
              { value: 'en', label: 'English' }
            ]}
          />
        </div>

        <div className="mt-4 rounded-tile border border-line p-3.5">
          <div className="flex items-center justify-between gap-3">
            <span className="text-[13px] font-medium">{t('about.autoUpdate')}</span>
            <Switch
              checked={!!u?.enabled}
              label={t('about.autoUpdate')}
              onChange={(on) => void window.pixforge.update.setEnabled(on)}
            />
          </div>
          <div className="mt-2 flex items-center justify-between gap-3">
            <span className="text-[12px] text-text3" role="status">
              {status}
            </span>
            {u?.phase === 'ready' ? (
              <button
                type="button"
                onClick={() => void window.pixforge.update.install()}
                className="h-8 rounded-btn bg-accent px-3 text-[12.5px] font-semibold text-on-accent hover:bg-accent-hover"
              >
                {t('about.restart')}
              </button>
            ) : (
              u?.enabled &&
              u.phase !== 'dev' && (
                <button
                  type="button"
                  disabled={u.phase === 'checking' || u.phase === 'downloading'}
                  onClick={() => void window.pixforge.update.check()}
                  className="h-8 rounded-btn border border-line px-3 text-[12.5px] text-text2 hover:bg-hover disabled:opacity-50"
                >
                  {t('about.check')}
                </button>
              )
            )}
          </div>
        </div>

        <dl className="mt-4 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-[12.5px]">
          <dt className="text-text3">{t('about.device')}</dt>
          <dd className="m-0 text-text2">
            {health ? `${health.device.name} (${health.device.provider})` : '–'}
          </dd>
          <dt className="text-text3">{t('about.privacy')}</dt>
          <dd className="m-0 text-text2">{t('about.privacyText')}</dd>
        </dl>

        <p className="mt-4 mb-0 flex flex-wrap gap-x-4 gap-y-1 text-[12.5px]">
          {link(REPO, t('about.source'))}
          {link(`${REPO}/blob/main/THIRD_PARTY.md`, t('about.licenses'))}
          <button
            type="button"
            className="text-accent-ink underline-offset-2 hover:underline"
            onClick={() => window.pixforge.openLogs()}
          >
            {t('errors.logs')}
          </button>
        </p>
        <p className="mt-4 mb-0 text-[11.5px] text-text3">{t('app.copyright')} · Apache-2.0</p>
      </div>
    </div>
  )
}
