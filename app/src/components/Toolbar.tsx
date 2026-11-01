import { Crop, Download, FolderOpen, FolderPlus, Moon, Sun } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { useStore, type ViewMode } from '@/state/store'
import { Segmented } from './ui'

export type ZoomPreset = 'fit' | 1 | 2

export function Toolbar({
  zoomPreset,
  onZoom,
  onOpen,
  onOpenFolder
}: {
  zoomPreset: ZoomPreset | null
  onZoom: (z: ZoomPreset) => void
  onOpen: () => void
  onOpenFolder: () => void
}) {
  const { t } = useTranslation()
  const viewMode = useStore((s) => s.viewMode)
  const selectedId = useStore((s) => s.selectedId)
  const theme = useStore((s) => s.theme)
  const set = useStore((s) => s.set)
  const setTheme = useStore((s) => s.setTheme)
  const exportImages = useStore((s) => s.exportImages)
  const cropEditing = useStore((s) => s.cropEditing)
  const crop = useStore((s) => (s.selectedId ? s.settings[s.selectedId]?.crop : null))
  const updateSettings = useStore((s) => s.updateSettings)
  const image = useStore((s) => s.images.find((i) => i.id === s.selectedId))

  const toggleCrop = () => {
    if (!image) return
    if (cropEditing) return set({ cropEditing: false })
    if (!crop) {
      // start with the centre 70 % of the image
      const w = Math.round(image.width * 0.7)
      const h = Math.round(image.height * 0.7)
      updateSettings(image.id, (s) => {
        s.crop = [Math.round((image.width - w) / 2), Math.round((image.height - h) / 2), w, h]
        return s
      })
    }
    set({ cropEditing: true })
  }

  const iconBtn =
    'grid h-8 w-8 shrink-0 place-items-center rounded-btn border border-line bg-raised text-text2 transition-colors duration-100 hover:bg-hover hover:text-text'
  const secondary =
    'flex h-8 items-center gap-[7px] rounded-btn border border-line bg-raised px-3 font-medium text-text transition-colors duration-100 hover:bg-hover disabled:opacity-40'
  return (
    <div className="flex min-h-[52px] shrink-0 flex-wrap items-center gap-x-4 gap-y-2 border-b border-line bg-app px-3.5 py-2">
      <div className="flex gap-1.5">
        <button type="button" className={secondary} onClick={onOpen} title="Strg+O">
          <FolderOpen size={16} strokeWidth={1.75} aria-hidden="true" />
          {t('toolbar.open')}
        </button>
        <button
          type="button"
          className={iconBtn}
          onClick={onOpenFolder}
          aria-label={t('toolbar.addFolder')}
          title={t('toolbar.addFolder')}
        >
          <FolderPlus size={16} strokeWidth={1.75} aria-hidden="true" />
        </button>
      </div>
      <button
        type="button"
        disabled={!image}
        aria-pressed={cropEditing || !!crop}
        onClick={toggleCrop}
        title="C"
        className={
          cropEditing || crop
            ? 'flex h-8 items-center gap-[7px] rounded-btn border border-accent-line bg-accent-soft px-3 font-medium text-accent-ink disabled:opacity-40'
            : secondary
        }
      >
        <Crop size={16} strokeWidth={1.75} aria-hidden="true" />
        {t('crop.button')}
      </button>
      <div className="h-[22px] w-px bg-line" />
      <Segmented<ViewMode>
        label={t('toolbar.view')}
        value={viewMode}
        onChange={(v) => set({ viewMode: v })}
        options={[
          { value: 'single', label: t('toolbar.single') },
          { value: 'split', label: t('toolbar.split') },
          { value: 'side', label: t('toolbar.side') }
        ]}
      />
      <Segmented<string>
        label={t('toolbar.zoom')}
        value={zoomPreset === null ? null : String(zoomPreset)}
        onChange={(v) => onZoom(v === 'fit' ? 'fit' : (Number(v) as 1 | 2))}
        options={[
          { value: 'fit', label: t('toolbar.fit'), title: '0' },
          { value: '1', label: '100 %', title: '1' },
          { value: '2', label: '200 %', title: '2' }
        ]}
      />
      <div className="flex-1" />
      <button
        type="button"
        className={iconBtn}
        aria-label={t('toolbar.theme')}
        title={t('toolbar.theme')}
        onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
      >
        {theme === 'dark' ? (
          <Sun size={16} strokeWidth={1.75} aria-hidden="true" />
        ) : (
          <Moon size={16} strokeWidth={1.75} aria-hidden="true" />
        )}
      </button>
      <button
        type="button"
        disabled={!selectedId}
        onClick={() => selectedId && void exportImages([selectedId])}
        title="Strg+E"
        className="flex h-8 items-center gap-2 rounded-btn bg-accent px-3.5 font-semibold text-on-accent transition-colors duration-100 hover:bg-accent-hover disabled:opacity-40"
      >
        <Download size={16} strokeWidth={2} aria-hidden="true" />
        {t('toolbar.export')}
      </button>
    </div>
  )
}
