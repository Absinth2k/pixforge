import { Check, Pause, Play, ScanFace, X } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { latestJob, useStore } from '@/state/store'

export function BatchBar() {
  const { t } = useTranslation()
  const api = useStore((s) => s.api)
  const images = useStore((s) => s.images)
  const selectedId = useStore((s) => s.selectedId)
  const settings = useStore((s) => s.settings)
  const jobs = useStore((s) => s.jobs)
  const paused = useStore((s) => s.queuePaused)
  const select = useStore((s) => s.select)
  const remove = useStore((s) => s.removeImage)
  const exportImages = useStore((s) => s.exportImages)

  const done = images.filter((i) => latestJob(jobs, i.id)?.status === 'done').length

  return (
    <div className="flex h-[112px] shrink-0 flex-col border-t border-line bg-app">
      <div className="flex items-center gap-3 px-3.5 pt-[7px]">
        <span className="text-[12px] font-semibold text-text2">{t('batch.title')}</span>
        <span className="text-[12px] text-text3">
          {t('batch.summary', { count: images.length })}
          {done ? ` · ${t('batch.done', { count: done })}` : ''}
        </span>
        <span className="flex-1" />
        <button
          type="button"
          onClick={() => void api?.pause(!paused)}
          className="flex h-[26px] items-center gap-1.5 rounded-btn border border-line bg-raised px-2.5 text-[12px] text-text hover:bg-hover"
        >
          {paused ? <Play size={13} aria-hidden="true" /> : <Pause size={13} aria-hidden="true" />}
          {paused ? t('batch.resume') : t('batch.pause')}
        </button>
        <button
          type="button"
          disabled={!images.length}
          onClick={() => void exportImages(images.map((i) => i.id))}
          className="h-[26px] rounded-btn bg-accent px-3 text-[12px] font-semibold text-on-accent hover:bg-accent-hover disabled:opacity-40"
        >
          {t('batch.processAll')}
        </button>
      </div>
      <div
        className="flex gap-2.5 overflow-x-auto px-3.5 py-2"
        role="listbox"
        aria-label={t('batch.title')}
      >
        {images.map((img) => {
          const job = latestJob(jobs, img.id)
          const selected = img.id === selectedId
          const facesOn =
            (settings[img.id]?.faces.enabled ?? img.faces.length > 0) && img.faces.length > 0
          const status =
            job?.status === 'done'
              ? t('batch.statusDone')
              : job?.status === 'running'
                ? t('batch.statusRunning')
                : job?.status === 'queued'
                  ? t('batch.statusQueued')
                  : job?.status === 'error'
                    ? t('batch.statusError')
                    : ''
          return (
            <div key={img.id} className="group relative shrink-0">
              <button
                type="button"
                role="option"
                aria-selected={selected}
                aria-label={`${img.name}${status ? ', ' + status : ''}`}
                onClick={() => select(img.id)}
                className={`relative h-16 w-[104px] overflow-hidden rounded-lg bg-raised p-0 ${
                  selected ? 'border-2 border-accent' : 'border border-line'
                }`}
              >
                {api && (
                  <img
                    src={api.thumbUrl(img.id)}
                    alt=""
                    draggable={false}
                    className="absolute inset-0 h-full w-full object-cover"
                  />
                )}
                {facesOn && (
                  <span
                    title={t('batch.facesOn')}
                    className="absolute top-[5px] left-[5px] grid h-[18px] w-[18px] place-items-center rounded-[5px] bg-[rgba(14,14,16,.78)] text-[#F2A541]"
                  >
                    <ScanFace size={12} strokeWidth={2.2} aria-hidden="true" />
                  </span>
                )}
                {job?.status === 'done' && (
                  <span className="absolute top-[5px] right-[5px] grid h-[18px] w-[18px] place-items-center rounded-full bg-success text-[#0E0E10]">
                    <Check size={11} strokeWidth={3.2} aria-hidden="true" />
                  </span>
                )}
                {job?.status === 'error' && (
                  <span
                    className="absolute top-[5px] right-[5px] grid h-[18px] w-[18px] place-items-center rounded-full bg-danger text-white"
                    title={job.error ?? ''}
                  >
                    !
                  </span>
                )}
                {(job?.status === 'running' || job?.status === 'queued') && (
                  <span className="absolute right-0 bottom-0 left-0 h-[3px] bg-black/50">
                    <span
                      className="block h-[3px] bg-[#F2A541] transition-[width] duration-200"
                      style={{ width: `${Math.round((job.progress || 0) * 100)}%` }}
                    />
                  </span>
                )}
                <span className="absolute right-0 bottom-[3px] left-0 truncate px-1.5 text-left text-[10.5px] text-[#EDEDF0] [text-shadow:0_1px_2px_rgba(0,0,0,.8)]">
                  {img.name}
                </span>
              </button>
              <button
                type="button"
                aria-label={`${t('batch.remove')}: ${img.name}`}
                onClick={() => void remove(img.id)}
                className="absolute -top-1.5 -right-1.5 hidden h-5 w-5 place-items-center rounded-full border border-line bg-panel text-text2 group-focus-within:grid group-hover:grid hover:text-text"
              >
                <X size={12} aria-hidden="true" />
              </button>
            </div>
          )
        })}
      </div>
    </div>
  )
}
