import { useRef } from 'react'
import type { PointerEvent as ReactPointerEvent } from 'react'
import type { Region } from '@/api/types'

type Handle = 'move' | 'n' | 's' | 'e' | 'w' | 'ne' | 'nw' | 'se' | 'sw'

const MIN_SIZE = 16 // source px

const HANDLES: { h: Exclude<Handle, 'move'>; style: React.CSSProperties; cursor: string }[] = [
  { h: 'nw', style: { left: -6, top: -6 }, cursor: 'nwse-resize' },
  { h: 'ne', style: { right: -6, top: -6 }, cursor: 'nesw-resize' },
  { h: 'sw', style: { left: -6, bottom: -6 }, cursor: 'nesw-resize' },
  { h: 'se', style: { right: -6, bottom: -6 }, cursor: 'nwse-resize' },
  { h: 'n', style: { left: 'calc(50% - 6px)', top: -6 }, cursor: 'ns-resize' },
  { h: 's', style: { left: 'calc(50% - 6px)', bottom: -6 }, cursor: 'ns-resize' },
  { h: 'w', style: { left: -6, top: 'calc(50% - 6px)' }, cursor: 'ew-resize' },
  { h: 'e', style: { right: -6, top: 'calc(50% - 6px)' }, cursor: 'ew-resize' }
]

/** Clamp a rectangle into the image, keeping its size where possible. */
export function clampRect(r: Region, imgW: number, imgH: number): Region {
  const w = Math.min(Math.max(MIN_SIZE, r.w), imgW)
  const h = Math.min(Math.max(MIN_SIZE, r.h), imgH)
  return {
    x: Math.min(Math.max(0, r.x), imgW - w),
    y: Math.min(Math.max(0, r.y), imgH - h),
    w,
    h
  }
}

/** Fit an aspect ratio (w/h) into the rectangle around its centre, inside the image. */
export function applyAspect(r: Region, aspect: number, imgW: number, imgH: number): Region {
  let w = r.w
  let h = w / aspect
  if (h > r.h) {
    h = r.h
    w = h * aspect
  }
  const maxW = Math.min(imgW, imgH * aspect)
  if (w > maxW) {
    w = maxW
    h = w / aspect
  }
  return clampRect({ x: r.x + (r.w - w) / 2, y: r.y + (r.h - h) / 2, w, h }, imgW, imgH)
}

function resize(
  start: Region,
  h: Handle,
  dx: number,
  dy: number,
  aspect: number | null,
  W: number,
  H: number
): Region {
  if (h === 'move') return clampRect({ ...start, x: start.x + dx, y: start.y + dy }, W, H)
  let x0 = start.x
  let y0 = start.y
  let x1 = start.x + start.w
  let y1 = start.y + start.h
  if (h.includes('w')) x0 = Math.min(x1 - MIN_SIZE, Math.max(0, x0 + dx))
  if (h.includes('e')) x1 = Math.max(x0 + MIN_SIZE, Math.min(W, x1 + dx))
  if (h.includes('n')) y0 = Math.min(y1 - MIN_SIZE, Math.max(0, y0 + dy))
  if (h.includes('s')) y1 = Math.max(y0 + MIN_SIZE, Math.min(H, y1 + dy))
  if (aspect) {
    let w = x1 - x0
    let hh = y1 - y0
    if (h === 'n' || h === 's') w = hh * aspect
    else hh = w / aspect
    // keep the edge/corner opposite to the dragged handle in place
    if (h === 'n' || h === 's') {
      const cx = (start.x * 2 + start.w) / 2
      x0 = cx - w / 2
    } else if (h === 'e' || h === 'w') {
      const cy = (start.y * 2 + start.h) / 2
      y0 = cy - hh / 2
    } else {
      if (h.includes('w')) x0 = x1 - w
      if (h.includes('n')) y0 = y1 - hh
    }
    if (w > W || hh > H) {
      const k = Math.min(W / w, H / hh)
      w *= k
      hh *= k
    }
    return clampRect({ x: x0, y: y0, w, h: hh }, W, H)
  }
  return clampRect({ x: x0, y: y0, w: x1 - x0, h: y1 - y0 }, W, H)
}

export function CropOverlay({
  crop,
  view,
  s,
  editing,
  aspect,
  imgW,
  imgH,
  label,
  onChange
}: {
  crop: Region
  view: Region
  s: number
  editing: boolean
  aspect: number | null
  imgW: number
  imgH: number
  label: string
  onChange: (r: Region) => void
}) {
  const drag = useRef<{ h: Handle; x: number; y: number; start: Region } | null>(null)

  const start = (e: ReactPointerEvent<HTMLElement>) => {
    if (!editing || e.button !== 0) return
    e.stopPropagation()
    e.currentTarget.setPointerCapture(e.pointerId)
    const h = (e.currentTarget.dataset.handle ?? 'move') as Handle
    drag.current = { h, x: e.clientX, y: e.clientY, start: crop }
  }
  const move = (e: ReactPointerEvent<HTMLElement>) => {
    const d = drag.current
    if (!d) return
    e.stopPropagation()
    onChange(resize(d.start, d.h, (e.clientX - d.x) / s, (e.clientY - d.y) / s, aspect, imgW, imgH))
  }
  const end = () => {
    drag.current = null
  }

  return (
    <div
      className="absolute"
      style={{
        left: (crop.x - view.x) * s,
        top: (crop.y - view.y) * s,
        width: crop.w * s,
        height: crop.h * s,
        boxShadow: `0 0 0 9999px rgba(8,8,10,${editing ? 0.6 : 0.5})`,
        border: editing ? '1.5px solid #FFFFFF' : '1.5px dashed var(--accent)',
        pointerEvents: editing ? 'auto' : 'none',
        cursor: editing ? 'move' : 'default',
        boxSizing: 'border-box'
      }}
      data-handle="move"
      onPointerDown={start}
      onPointerMove={move}
      onPointerUp={end}
      onPointerCancel={end}
      onDoubleClick={(e) => e.stopPropagation()}
    >
      {editing && (
        <>
          {[33.333, 66.666].map((p) => (
            <span
              key={'v' + p}
              className="pointer-events-none absolute inset-y-0 w-px bg-white/35"
              style={{ left: `${p}%` }}
            />
          ))}
          {[33.333, 66.666].map((p) => (
            <span
              key={'h' + p}
              className="pointer-events-none absolute inset-x-0 h-px bg-white/35"
              style={{ top: `${p}%` }}
            />
          ))}
          {HANDLES.map(({ h, style, cursor }) => (
            <span
              key={h}
              data-handle={h}
              onPointerDown={start}
              className="absolute h-3 w-3 rounded-[2px] bg-white shadow-[0_0_0_1px_rgba(0,0,0,.4)]"
              style={{ ...style, cursor }}
            />
          ))}
        </>
      )}
      <span className="pointer-events-none absolute top-[calc(100%+8px)] left-0 rounded-[5px] bg-[rgba(14,14,16,.85)] px-[9px] py-1 text-[11.5px] font-medium whitespace-nowrap text-[#EDEDF0]">
        {label}
      </span>
    </div>
  )
}
