import { FolderOpen, ImageIcon } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { Logo } from './ui'

export function EmptyState({
  onOpen,
  onOpenFolder,
  dragging
}: {
  onOpen: () => void
  onOpenFolder: () => void
  dragging: boolean
}) {
  const { t } = useTranslation()
  return (
    <main className="flex min-h-0 flex-1 flex-col items-center justify-center gap-7 bg-canvas px-6 py-10">
      <div
        className={`flex min-h-[380px] w-full max-w-[760px] flex-col items-center justify-center gap-[18px] rounded-[20px] border-[1.5px] border-dashed px-8 py-12 text-center transition-colors duration-150 ${
          dragging ? 'border-accent bg-accent-soft' : 'border-line-strong bg-app'
        }`}
      >
        <Logo size={88} />
        <h1 className="mt-1 mb-0 text-[28px] font-[650] tracking-[-0.025em]">
          {dragging ? t('empty.drop') : t('empty.title')}
        </h1>
        <p className="m-0 max-w-[460px] text-[15px] text-text2">{t('empty.text')}</p>
        <div className="mt-1.5 flex flex-wrap justify-center gap-2.5">
          <button
            type="button"
            onClick={onOpen}
            className="flex h-11 items-center gap-2 rounded-[9px] bg-accent px-5 text-[15px] font-[650] text-on-accent hover:bg-accent-hover"
          >
            <ImageIcon size={18} strokeWidth={2} aria-hidden="true" />
            {t('empty.pick')}
          </button>
          <button
            type="button"
            onClick={onOpenFolder}
            className="flex h-11 items-center gap-2 rounded-[9px] border border-line bg-raised px-[18px] text-[15px] font-medium text-text hover:bg-hover"
          >
            <FolderOpen size={18} strokeWidth={1.8} aria-hidden="true" />
            {t('empty.folder')}
          </button>
        </div>
        <p className="mt-1 mb-0 text-[12.5px] text-text3">{t('empty.formats')}</p>
      </div>
    </main>
  )
}
