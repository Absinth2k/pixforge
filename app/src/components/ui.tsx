import type { ReactNode } from 'react'

export function Logo({ size = 18, framed = true }: { size?: number; framed?: boolean }) {
  return (
    <svg width={size} height={size} viewBox="0 0 64 64" aria-hidden="true">
      {framed && (
        <rect x="0.5" y="0.5" width="63" height="63" rx="13.5" fill="#141416" stroke="#2B2B30" />
      )}
      <path d="M32 12H24V16H16V24H12V40H16V48H24V52H32A20 20 0 0 0 32 12Z" fill="#F2A541" />
      <rect x="13" y="6" width="5" height="5" fill="#F2A541" opacity="0.7" />
    </svg>
  )
}

export interface SegOption<T extends string | number> {
  value: T
  label: ReactNode
  title?: string
  disabled?: boolean
}

export function Segmented<T extends string | number>({
  options,
  value,
  onChange,
  label,
  stretch = false,
  dark = false
}: {
  options: SegOption<T>[]
  value: T | null
  onChange: (v: T) => void
  label: string
  stretch?: boolean
  dark?: boolean
}) {
  return (
    <div
      role="group"
      aria-label={label}
      className={
        (stretch ? 'grid ' : 'flex ') +
        (dark
          ? 'gap-0.5 rounded-lg border border-white/10 bg-[rgba(14,14,16,.8)] p-[3px] backdrop-blur'
          : 'gap-0.5 rounded-lg border border-line bg-raised p-[3px]')
      }
      style={
        stretch ? { gridTemplateColumns: `repeat(${options.length}, minmax(0, 1fr))` } : undefined
      }
    >
      {options.map((o) => {
        const active = o.value === value
        const cls = dark
          ? active
            ? 'bg-white/15 font-semibold text-white'
            : 'font-medium text-[#B9B9C1] hover:bg-white/10'
          : active
            ? 'bg-hover font-semibold text-text shadow-[0_1px_2px_rgba(0,0,0,.25)]'
            : 'font-medium text-text2 hover:bg-hover'
        return (
          <button
            key={String(o.value)}
            type="button"
            title={o.title}
            disabled={o.disabled}
            aria-pressed={active}
            onClick={() => onChange(o.value)}
            className={`h-[26px] whitespace-nowrap rounded-[5px] px-[11px] text-[12.5px] transition-colors duration-100 disabled:cursor-not-allowed disabled:opacity-40 ${cls}`}
          >
            {o.label}
          </button>
        )
      })}
    </div>
  )
}

export function Slider({
  id,
  label,
  value,
  onChange,
  display,
  disabled,
  badge
}: {
  id: string
  label: string
  value: number
  onChange: (v: number) => void
  display?: string
  disabled?: boolean
  badge?: ReactNode
}) {
  return (
    <div className="mb-2">
      <div className="flex items-center gap-2 text-[12.5px]">
        <label htmlFor={id} className="text-text">
          {label}
        </label>
        {badge}
        <span className="flex-1" />
        <span className="text-text2">{display ?? Math.round(value)}</span>
      </div>
      <input
        id={id}
        className="pf-range"
        type="range"
        min={0}
        max={100}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{ ['--p' as string]: `${value}%` }}
      />
    </div>
  )
}

export function Switch({
  checked,
  onChange,
  label,
  large = false
}: {
  checked: boolean
  onChange: (v: boolean) => void
  label: string
  large?: boolean
}) {
  const w = large ? 44 : 34
  const h = large ? 26 : 20
  const k = h - 4
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      onClick={() => onChange(!checked)}
      className="relative shrink-0 rounded-full transition-colors duration-150"
      style={{ width: w, height: h, background: checked ? 'var(--accent)' : 'var(--track)' }}
    >
      <span
        className="absolute top-[2px] rounded-full bg-white shadow-[0_1px_2px_rgba(0,0,0,.35)] transition-[left] duration-150"
        style={{ width: k, height: k, left: checked ? w - k - 2 : 2 }}
      />
    </button>
  )
}

export function Section({
  title,
  children,
  extra
}: {
  title: string
  children: ReactNode
  extra?: ReactNode
}) {
  return (
    <section className="border-b border-line px-[18px] py-[14px]">
      <div className="mb-2.5 flex items-center gap-2">
        <h3 className="m-0 text-[12px] font-semibold text-text2">{title}</h3>
        <span className="flex-1" />
        {extra}
      </div>
      {children}
    </section>
  )
}

export function Note({ children }: { children: ReactNode }) {
  return (
    <div className="mt-2.5 flex gap-2 rounded-lg border border-accent-line bg-accent-soft px-2.5 py-[9px] text-[12px] leading-[1.45] text-text">
      <svg
        width="15"
        height="15"
        viewBox="0 0 24 24"
        fill="none"
        stroke="var(--accent-ink)"
        strokeWidth="2"
        strokeLinecap="round"
        aria-hidden="true"
        className="mt-px shrink-0"
      >
        <circle cx="12" cy="12" r="9" />
        <path d="M12 11v5M12 8h.01" />
      </svg>
      <span>{children}</span>
    </div>
  )
}

export function formatDuration(sec: number, lang = 'de'): string {
  if (!isFinite(sec) || sec < 0) return '–'
  if (sec < 1) return lang === 'de' ? '< 1 s' : '< 1 s'
  if (sec < 60) return `${Math.round(sec)} s`
  const m = Math.floor(sec / 60)
  const s = Math.round(sec % 60)
  if (m < 60) return `${m}:${String(s).padStart(2, '0')} min`
  return `${Math.floor(m / 60)} h ${m % 60} min`
}

export const nf = (n: number, lang = 'de'): string =>
  n.toLocaleString(lang === 'de' ? 'de-DE' : 'en-US')
