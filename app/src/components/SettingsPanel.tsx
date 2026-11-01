import { useEffect, useState } from 'react'
import { Check, Crop, MoveRight, RotateCcw } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import type { Estimate, ImageInfo, Settings } from '@/api/types'
import { useStore } from '@/state/store'
import { formatDuration, Note, nf, Section, Segmented, Slider, Switch } from './ui'

interface ModelChoice {
  family: string
  id: (factor: number) => string
  name: { de: string; en: string }
  desc: { de: string; en: string }
  meta: string
}

const MODELS: ModelChoice[] = [
  {
    family: 'photo',
    id: () => 'photo',
    name: { de: 'Fotos', en: 'Photos' },
    desc: {
      de: 'Echte Fotos, auch verrauscht oder komprimiert',
      en: 'Real photos, also noisy or compressed'
    },
    meta: 'Standard · 4×'
  },
  {
    family: 'photo-sharp',
    id: () => 'photo-sharp',
    name: { de: 'Fotos, schärfer', en: 'Photos, sharper' },
    desc: {
      de: 'Knackiger, für Landschaft und Architektur',
      en: 'Crisper, for landscape and architecture'
    },
    meta: '4×'
  },
  {
    family: 'clean',
    id: () => 'clean-x4',
    name: { de: 'Saubere Bilder', en: 'Clean images' },
    desc: { de: 'Scans, Grafiken, rauschfreie Fotos', en: 'Scans, graphics, noise-free photos' },
    meta: '2× · 3× · 4×'
  },
  {
    family: 'max',
    id: () => 'max-x4',
    name: { de: 'Maximale Qualität', en: 'Maximum quality' },
    desc: {
      de: 'Größtes Modell, etwa doppelt so langsam',
      en: 'Largest model, about twice as slow'
    },
    meta: '2× · 4×'
  }
]

const familyOf = (model: string): string =>
  model.startsWith('photo-sharp')
    ? 'photo-sharp'
    : model.startsWith('photo')
      ? 'photo'
      : model.split('-x')[0]

type ScaleChoice = 1 | 2 | 4 | 6 | 'free'

function scaleChoice(s: Settings): ScaleChoice {
  if (s.target_width || s.target_height) return 'free'
  return ([1, 2, 4, 6] as const).find((v) => v === s.scale) ?? 'free'
}

export function SettingsPanel({ img }: { img: ImageInfo }) {
  const { t, i18n } = useTranslation()
  const lang = i18n.language === 'de' ? 'de' : 'en'
  const api = useStore((s) => s.api)
  const settings = useStore((s) => s.settings[img.id])
  const health = useStore((s) => s.health)
  const models = useStore((s) => s.models)
  const update = useStore((s) => s.updateSettings)
  const reset = useStore((s) => s.resetSettings)
  const images = useStore((s) => s.images)
  const setAll = useStore((s) => s.set)
  const [est, setEst] = useState<Estimate | null>(null)

  const settingsKey = JSON.stringify(settings)
  useEffect(() => {
    if (!api || !settings) return
    const ctrl = new AbortController()
    const timer = setTimeout(() => {
      api
        .estimate(img.id, settings, ctrl.signal)
        .then(setEst)
        .catch(() => undefined)
    }, 120)
    return () => {
      clearTimeout(timer)
      ctrl.abort()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [api, img.id, settingsKey])

  if (!settings) return null
  const denoiseTask = settings.task === 'denoise'
  const family = familyOf(settings.model)
  const choice = scaleChoice(settings)
  const sug = family.startsWith('photo') && !denoiseTask ? img.denoise.photo : img.denoise.clean
  const dnAuto = settings.denoise.mode === 'auto' && settings.denoise.strength === null
  const dnValue = settings.denoise.mode === 'off' ? 0 : (settings.denoise.strength ?? sug.strength)
  const faceModels = models.filter((m) => m.kind === 'face')
  const facesEnabled = settings.faces.enabled ?? img.faces.length > 0
  const gpu = health?.device.kind === 'gpu'

  const set = (fn: (s: Settings) => void) =>
    update(img.id, (s) => {
      fn(s)
      return s
    })

  const pickScale = (v: ScaleChoice) =>
    set((s) => {
      if (v === 'free') {
        s.target_width = Math.round(img.width * (s.scale || 4))
        s.target_height = null
      } else {
        s.scale = v
        s.target_width = null
        s.target_height = null
      }
    })

  let note: string | null = null
  if (denoiseTask) note = t('panel.noteDenoise')
  else if (choice === 1) note = t('panel.note1x')

  return (
    <aside
      aria-label={t('panel.title')}
      className="flex h-full max-w-full min-w-[300px] flex-[1_1_340px] flex-col overflow-y-auto border-l border-line bg-panel"
    >
      <div className="flex items-center justify-between gap-2 px-[18px] pt-3.5 pb-3">
        <h2 className="m-0 text-[14px] font-semibold tracking-[-0.005em]">{t('panel.title')}</h2>
        <button
          type="button"
          onClick={() => reset(img.id)}
          className="flex h-7 items-center gap-1.5 rounded-btn px-2 text-[12px] text-text2 hover:bg-hover"
        >
          <RotateCcw size={13} aria-hidden="true" />
          {t('panel.reset')}
        </button>
      </div>

      <Section title={t('panel.model')}>
        {denoiseTask && (
          <p className="-mt-1 mb-2.5 text-[11.5px] text-text3">{t('panel.modelNotNeeded')}</p>
        )}
        <div className="grid grid-cols-2 gap-2">
          {MODELS.map((m) => {
            const selected = !denoiseTask && family === m.family
            return (
              <button
                key={m.family}
                type="button"
                aria-pressed={selected}
                onClick={() =>
                  set((s) => {
                    s.task = 'upscale'
                    s.model = m.id(s.scale)
                  })
                }
                className={`flex min-h-[84px] flex-col items-start gap-[3px] rounded-tile border px-[11px] py-2.5 text-left transition-colors duration-150 ${
                  selected
                    ? 'border-accent-line bg-accent-soft'
                    : 'border-line bg-raised hover:border-line-strong'
                } ${denoiseTask ? 'opacity-45' : ''}`}
              >
                <span className="flex w-full items-center justify-between gap-1.5">
                  <span className="text-[12.5px] font-semibold text-text">{m.name[lang]}</span>
                  {selected && (
                    <Check
                      size={14}
                      strokeWidth={2.4}
                      className="text-accent-ink"
                      aria-hidden="true"
                    />
                  )}
                </span>
                <span className="text-[11.5px] leading-[1.35] text-text3">{m.desc[lang]}</span>
                <span className="text-[11px] text-text3">{m.meta}</span>
              </button>
            )
          })}
        </div>
      </Section>

      <Section title={t('panel.task')}>
        <Segmented<'upscale' | 'denoise'>
          stretch
          label={t('panel.task')}
          value={settings.task}
          onChange={(v) => set((s) => void (s.task = v))}
          options={[
            { value: 'upscale', label: t('panel.upscale') },
            { value: 'denoise', label: t('panel.denoiseOnly') }
          ]}
        />
        {!denoiseTask && (
          <div className="mt-2">
            <Segmented<string>
              stretch
              label="Faktor"
              value={String(choice)}
              onChange={(v) => pickScale(v === 'free' ? 'free' : (Number(v) as 1 | 2 | 4 | 6))}
              options={[
                { value: '1', label: '1×', title: t('panel.note1x') },
                { value: '2', label: '2×' },
                { value: '4', label: '4×' },
                { value: '6', label: '6×' },
                { value: 'free', label: t('panel.free') }
              ]}
            />
          </div>
        )}
        {!denoiseTask && choice === 'free' && (
          <div className="mt-2 grid grid-cols-2 gap-2">
            {(['target_width', 'target_height'] as const).map((k) => (
              <label key={k} className="flex flex-col gap-1 text-[11.5px] text-text3">
                {k === 'target_width' ? t('panel.width') : t('panel.height')}
                <input
                  type="number"
                  min={1}
                  max={65000}
                  value={settings[k] ?? ''}
                  placeholder={String(k === 'target_width' ? img.width * 2 : img.height * 2)}
                  onChange={(e) =>
                    set((s) => {
                      const v = e.target.value ? Math.max(1, Number(e.target.value)) : null
                      s[k] = v
                      // one dimension at a time keeps the aspect ratio
                      if (v) s[k === 'target_width' ? 'target_height' : 'target_width'] = null
                    })
                  }
                  className="h-8 rounded-btn border border-line bg-raised px-2 text-[13px] text-text"
                />
              </label>
            ))}
          </div>
        )}
        <div className="mt-3 flex items-baseline justify-between gap-2">
          <span className="text-[12px] text-text3">
            {nf(est?.in_w ?? img.width, lang)} × {nf(est?.in_h ?? img.height, lang)}
          </span>
          <MoveRight size={16} className="text-text3" aria-hidden="true" />
          <span className="text-[14px] font-semibold">
            {est ? `${nf(est.out_w, lang)} × ${nf(est.out_h, lang)}` : '…'}
          </span>
          <span className="text-[12px] text-text3">
            {est ? `${est.megapixels.toLocaleString(lang)} MP` : ''}
          </span>
        </div>
        {est && (
          <p className="mt-1.5 mb-0 text-[11.5px] text-text3">
            {t(gpu ? 'panel.etaGpu' : 'panel.etaCpu', { t: formatDuration(est.seconds, lang) })}
            {choice === 1 && !denoiseTask ? ' (wie 4×)' : ''}
          </p>
        )}
        {note && <Note>{note}</Note>}
        {est?.warnings.map((w) => (
          <Note key={w}>{w}</Note>
        ))}
        {settings.crop && (
          <div className="mt-2.5 flex items-center gap-2.5 rounded-lg border border-line bg-raised px-2.5 py-[9px] text-[12px]">
            <Crop size={15} className="shrink-0 text-text2" aria-hidden="true" />
            <span className="min-w-0 flex-1 text-text2">
              {t('crop.card', {
                w: nf(settings.crop[2], lang),
                h: nf(settings.crop[3], lang)
              })}
            </span>
            <div className="flex shrink-0 flex-col gap-1">
              <button
                type="button"
                onClick={() => setAll({ cropEditing: true })}
                className="h-[26px] rounded-btn border border-line px-2 text-[12px] text-text hover:bg-hover"
              >
                {t('crop.edit')}
              </button>
              <button
                type="button"
                onClick={() => {
                  set((s) => void (s.crop = null))
                  setAll({ cropEditing: false })
                }}
                className="h-[26px] rounded-btn border border-line px-2 text-[12px] text-text hover:bg-hover"
              >
                {t('crop.remove')}
              </button>
            </div>
          </div>
        )}
      </Section>

      <section className="border-b border-line px-[18px] py-[14px]">
        <Slider
          id="sl-denoise"
          label={t('panel.noise')}
          value={dnValue}
          display={dnValue === 0 ? t('panel.off') : String(Math.round(dnValue))}
          badge={
            dnAuto ? (
              <span className="rounded-full bg-accent-soft px-[7px] py-px text-[11px] font-semibold text-accent-ink">
                {t('panel.auto')}
              </span>
            ) : (
              <button
                type="button"
                className="rounded-full border border-line px-[7px] py-px text-[11px] text-text3 hover:bg-hover"
                onClick={() =>
                  set((s) => void (s.denoise = { ...s.denoise, mode: 'auto', strength: null }))
                }
              >
                {t('panel.auto')}
              </button>
            )
          }
          onChange={(v) =>
            set(
              (s) => void (s.denoise = { ...s.denoise, mode: v === 0 ? 'off' : 'on', strength: v })
            )
          }
        />
        {dnAuto && (
          <p className="m-0 text-[11.5px] text-text3">
            {lang === 'de' ? sug.message : sug.message_en}.
          </p>
        )}
      </section>

      <Section title={t('panel.fine')}>
        <Slider
          id="sl-sharpen"
          label={t('panel.sharpen')}
          value={settings.sharpen}
          onChange={(v) => set((s) => void (s.sharpen = v))}
        />
        <Slider
          id="sl-detail"
          label={t('panel.detail')}
          value={settings.detail_mix}
          disabled={denoiseTask}
          onChange={(v) => set((s) => void (s.detail_mix = v))}
        />
        <Slider
          id="sl-grain"
          label={t('panel.grain')}
          value={settings.grain}
          onChange={(v) => set((s) => void (s.grain = v))}
        />
        <Slider
          id="sl-jpeg"
          label={t('panel.jpeg')}
          value={settings.jpeg_strength}
          onChange={(v) => set((s) => void (s.jpeg_strength = v))}
        />
      </Section>

      <section className="border-b border-line px-[18px] py-[14px]">
        <div className="flex items-center justify-between gap-2">
          <h3 className="m-0 text-[12px] font-semibold text-text2">{t('panel.faces')}</h3>
          <Switch
            label={t('panel.faces')}
            checked={facesEnabled}
            onChange={(v) => set((s) => void (s.faces = { ...s.faces, enabled: v }))}
          />
        </div>
        <p className="mt-1 mb-2.5 text-[11.5px] text-text3">
          {img.faces.length
            ? t('panel.facesFound', { count: img.faces.length }) +
              (settings.faces.exclude.length
                ? ' · ' + t('panel.facesExcluded', { count: settings.faces.exclude.length })
                : '')
            : t('panel.noFaces')}
        </p>
        <Slider
          id="sl-face"
          label={t('panel.strength')}
          value={settings.faces.strength}
          disabled={!facesEnabled}
          onChange={(v) => set((s) => void (s.faces = { ...s.faces, strength: v }))}
        />
        <div className="mt-2 flex items-center justify-between gap-2">
          <label htmlFor="face-model" className="text-[12.5px]">
            {t('panel.faceModel')}
          </label>
          <select
            id="face-model"
            value={settings.faces.model}
            disabled={!facesEnabled}
            onChange={(e) => set((s) => void (s.faces = { ...s.faces, model: e.target.value }))}
            className="h-7 rounded-btn border border-line bg-raised px-2 text-[12px] text-text"
          >
            {faceModels.map((m) => (
              <option key={m.id} value={m.id} disabled={!m.installed.length}>
                {m.name[lang].replace(/^Gesichter \(|^Faces \(|\)$/g, '')}
                {m.noncommercial ? ` (${t('panel.nonCommercial')})` : ''}
              </option>
            ))}
          </select>
        </div>
        <label className="mt-2.5 flex items-center gap-2 text-[12.5px]">
          <input
            type="checkbox"
            checked={settings.faces.only_largest}
            disabled={!facesEnabled}
            onChange={(e) =>
              set((s) => void (s.faces = { ...s.faces, only_largest: e.target.checked }))
            }
            className="m-0 h-[15px] w-[15px] accent-[var(--accent)]"
          />
          {t('panel.onlyLargest')}
        </label>
      </section>

      <div className="mt-auto flex gap-2 px-[18px] pt-3.5 pb-[18px]">
        <button
          type="button"
          disabled={images.length < 2}
          onClick={() => {
            const st = useStore.getState().settings
            const next = { ...st }
            for (const i of images)
              next[i.id] = {
                ...structuredClone(settings),
                faces: {
                  ...settings.faces,
                  exclude: [],
                  enabled: i.faces.length > 0 && facesEnabled
                }
              }
            setAll({ settings: next })
          }}
          className="h-8 flex-1 rounded-btn border border-line bg-transparent font-medium text-text2 hover:bg-hover disabled:opacity-40"
        >
          {t('panel.applyAll')}
        </button>
      </div>
    </aside>
  )
}
