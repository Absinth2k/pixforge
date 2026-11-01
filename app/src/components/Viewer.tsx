import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import type { PointerEvent as ReactPointerEvent, WheelEvent as ReactWheelEvent } from 'react'
import { ScanFace } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { ApiError } from '@/api/client'
import type { ImageInfo, Region, Settings, Stage } from '@/api/types'
import { useStore, type ViewState } from '@/state/store'
import { Segmented } from './ui'

const PREVIEW_DEBOUNCE_MS = 300
const ORIGINAL_DEBOUNCE_MS = 40
const MAX_OUT = 4096
const MIN_ZOOM = 0.02
const MAX_ZOOM = 8

/** Output factor exactly like the backend's target_size (crop ignored in the preview). */
export function outputFactor(img: ImageInfo, s: Settings): number {
  if (s.task === 'denoise') return 1
  if (s.target_width && s.target_height)
    return Math.min(s.target_width / img.width, s.target_height / img.height)
  if (s.target_width) return s.target_width / img.width
  if (s.target_height) return s.target_height / img.height
  return s.scale
}

interface Layer {
  url: string
  region: Region
}

interface Geometry {
  paneW: number
  paneH: number
  s: number // screen px per source px
  z: number // screen px per output px
  fitZ: number
  view: Region // visible source region (may extend past the image)
}

function computeGeometry(
  img: ImageInfo,
  factor: number,
  paneW: number,
  paneH: number,
  vs: ViewState | undefined
): Geometry {
  const fitZ = Math.min(paneW / (img.width * factor), paneH / (img.height * factor))
  const z = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, vs?.zoom ?? fitZ))
  const s = z * factor
  const w = paneW / s
  const h = paneH / s
  const clampC = (c: number | undefined, size: number, view: number): number => {
    if (size <= view) return size / 2
    const v = c ?? size / 2
    return Math.min(size - view / 2, Math.max(view / 2, v))
  }
  const cx = clampC(vs?.cx, img.width, w)
  const cy = clampC(vs?.cy, img.height, h)
  return { paneW, paneH, s, z, fitZ, view: { x: cx - w / 2, y: cy - h / 2, w, h } }
}

function visibleRegion(img: ImageInfo, g: Geometry): Region {
  const x0 = Math.max(0, g.view.x)
  const y0 = Math.max(0, g.view.y)
  const x1 = Math.min(img.width, g.view.x + g.view.w)
  const y1 = Math.min(img.height, g.view.y + g.view.h)
  return { x: x0, y: y0, w: Math.max(1, x1 - x0), h: Math.max(1, y1 - y0) }
}

function outSize(r: Region, g: Geometry): [number, number] {
  const dpr = window.devicePixelRatio || 1
  let ow = r.w * g.s * dpr
  let oh = r.h * g.s * dpr
  const k = Math.min(1, MAX_OUT / Math.max(ow, oh))
  ow = Math.max(1, Math.round(ow * k))
  oh = Math.max(1, Math.round(oh * k))
  return [ow, oh]
}

function layerStyle(l: Layer, g: Geometry): React.CSSProperties {
  return {
    position: 'absolute',
    left: (l.region.x - g.view.x) * g.s,
    top: (l.region.y - g.view.y) * g.s,
    width: l.region.w * g.s,
    height: l.region.h * g.s,
    imageRendering: g.z >= 2 ? 'pixelated' : 'auto'
  }
}

export function Viewer({
  img,
  onZoomPreset
}: {
  img: ImageInfo
  onZoomPreset: (z: 'fit' | 1 | 2 | null) => void
}) {
  const { t, i18n } = useTranslation()
  const api = useStore((s) => s.api)
  const settings = useStore((s) => s.settings[img.id])
  const viewMode = useStore((s) => s.viewMode)
  const stage = useStore((s) => s.stage)
  const split = useStore((s) => s.split)
  const showFaces = useStore((s) => s.showFaces)
  const vs = useStore((s) => s.view[img.id])
  const setView = useStore((s) => s.setView)
  const set = useStore((s) => s.set)
  const updateSettings = useStore((s) => s.updateSettings)

  const boxRef = useRef<HTMLDivElement>(null)
  const [box, setBox] = useState({ w: 800, h: 600 })
  const [original, setOriginal] = useState<Layer | null>(null)
  const [result, setResult] = useState<Layer | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useLayoutEffect(() => {
    const el = boxRef.current
    if (!el) return
    const ro = new ResizeObserver(([e]) =>
      setBox({ w: e.contentRect.width, h: e.contentRect.height })
    )
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const side = viewMode === 'side'
  const gap = side ? 2 : 0
  const paneW = side ? (box.w - gap) / 2 : box.w
  const factor = settings ? outputFactor(img, settings) : 4
  const g = useMemo(
    () => computeGeometry(img, factor, paneW, box.h, vs),
    [img, factor, paneW, box.h, vs]
  )
  const region = useMemo(() => visibleRegion(img, g), [img, g])

  useEffect(() => {
    const z = vs?.zoom ?? null
    onZoomPreset(
      z === null ? 'fit' : Math.abs(z - 1) < 1e-6 ? 1 : Math.abs(z - 2) < 1e-6 ? 2 : null
    )
  }, [vs?.zoom, onZoomPreset])

  // Reset layers when the image changes.
  useEffect(() => {
    setOriginal(null)
    setResult(null)
    setError(null)
  }, [img.id])

  // Original layer: cheap, nearly immediate.
  const regionKey = `${region.x.toFixed(1)},${region.y.toFixed(1)},${region.w.toFixed(1)},${region.h.toFixed(1)}`
  const [ow, oh] = outSize(region, g)
  useEffect(() => {
    if (!api) return
    const timer = setTimeout(() => {
      const url = api.viewUrl(img.id, region, ow, oh)
      const pre = new Image()
      pre.onload = () => setOriginal({ url, region })
      pre.src = url
    }, ORIGINAL_DEBOUNCE_MS)
    return () => clearTimeout(timer)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [api, img.id, regionKey, ow, oh])

  // Result layer: debounced, cancellable, keeps the previous image until the new one arrives.
  const effectiveStage: Stage = viewMode === 'single' || stage !== 'original' ? stage : 'result'
  const settingsKey = JSON.stringify(settings)
  useEffect(() => {
    if (!api || !settings || effectiveStage === 'original') return
    const ctrl = new AbortController()
    setBusy(true)
    const timer = setTimeout(async () => {
      try {
        const res = await api.preview(img.id, settings, region, ow, oh, effectiveStage, ctrl.signal)
        if (!res) return
        setResult((old) => {
          if (old) URL.revokeObjectURL(old.url)
          return { url: res.url, region }
        })
        setError(null)
        setBusy(false)
      } catch (e) {
        if ((e as Error).name === 'AbortError') return
        setError(e instanceof ApiError ? e.message : String(e))
        setBusy(false)
      }
    }, PREVIEW_DEBOUNCE_MS)
    return () => {
      clearTimeout(timer)
      ctrl.abort()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [api, img.id, regionKey, ow, oh, settingsKey, effectiveStage])

  // ---- interaction -----------------------------------------------------------------
  const commit = useCallback(
    (z: number | null, cx: number, cy: number) => setView(img.id, { zoom: z, cx, cy }),
    [img.id, setView]
  )
  const dragRef = useRef<{ x: number; y: number; cx: number; cy: number; moved: boolean } | null>(
    null
  )
  const splitDrag = useRef(false)

  const onPointerDown = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return
    ;(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId)
    dragRef.current = {
      x: e.clientX,
      y: e.clientY,
      cx: g.view.x + g.view.w / 2,
      cy: g.view.y + g.view.h / 2,
      moved: false
    }
  }
  const onPointerMove = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (splitDrag.current) {
      const rect = boxRef.current!.getBoundingClientRect()
      set({ split: Math.min(100, Math.max(0, ((e.clientX - rect.left) / rect.width) * 100)) })
      return
    }
    const d = dragRef.current
    if (!d) return
    const dx = e.clientX - d.x
    const dy = e.clientY - d.y
    if (!d.moved && Math.hypot(dx, dy) < 3) return
    d.moved = true
    commit(g.z, d.cx - dx / g.s, d.cy - dy / g.s)
  }
  const onPointerUp = () => {
    dragRef.current = null
    splitDrag.current = false
  }
  const onWheel = (e: ReactWheelEvent<HTMLDivElement>) => {
    const rect = boxRef.current!.getBoundingClientRect()
    let px = e.clientX - rect.left
    if (side && px > paneW) px -= paneW + gap
    const py = e.clientY - rect.top
    const sx = g.view.x + px / g.s
    const sy = g.view.y + py / g.s
    const z2 = Math.min(
      MAX_ZOOM,
      Math.max(Math.min(g.fitZ, 1), g.z * (e.deltaY < 0 ? 1.2 : 1 / 1.2))
    )
    const s2 = z2 * factor
    commit(z2, sx - px / s2 + paneW / s2 / 2, sy - py / s2 + box.h / s2 / 2)
  }
  const onDoubleClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = boxRef.current!.getBoundingClientRect()
    if (vs?.zoom === undefined || vs.zoom === null) {
      const sx = g.view.x + (e.clientX - rect.left) / g.s
      const sy = g.view.y + (e.clientY - rect.top) / g.s
      commit(1, sx, sy)
    } else commit(null, img.width / 2, img.height / 2)
  }

  // ---- rendering helpers -------------------------------------------------------------
  const splitX = (split / 100) * paneW
  const resultClip = (l: Layer): string => {
    if (viewMode !== 'split') return 'none'
    const left = (l.region.x - g.view.x) * g.s
    return `inset(0 0 0 ${Math.max(0, splitX - left)}px)`
  }
  const facesOn = showFaces && settings?.faces.enabled !== false && img.faces.length > 0
  const excluded = new Set(settings?.faces.exclude ?? [])
  const toggleFace = (i: number) =>
    updateSettings(img.id, (s) => {
      const ex = new Set(s.faces.exclude)
      if (ex.has(i)) ex.delete(i)
      else ex.add(i)
      s.faces.exclude = [...ex].sort((a, b) => a - b)
      return s
    })
  const resultLabel =
    effectiveStage === 'denoised'
      ? t('preview.denoised')
      : effectiveStage === 'original'
        ? t('preview.original')
        : `Pixforge · ${settings?.task === 'denoise' ? t('panel.denoiseOnly') : `${Math.round(factor * 100) / 100}×`}`
  // Labels sit inside the visible image, not in the empty canvas around it.
  const labelInset = {
    left: Math.max(12, -g.view.x * g.s + 12),
    top: Math.max(12, -g.view.y * g.s + 12),
    right: Math.max(12, paneW - (img.width - g.view.x) * g.s + 12)
  }
  const zoomText =
    vs?.zoom == null
      ? `${t('toolbar.fit')} · ${Math.round(g.z * 100)} %`
      : `${Math.round(g.z * 100)} %`

  const pane = (which: 'left' | 'right' | 'single') => {
    const showOriginal =
      which === 'left' ||
      viewMode === 'split' ||
      (which === 'single' && effectiveStage === 'original')
    const showResult = which !== 'left' && effectiveStage !== 'original'
    return (
      <div
        className="absolute inset-y-0 overflow-hidden"
        style={{ left: which === 'right' ? paneW + gap : 0, width: paneW }}
      >
        {showOriginal && original && (
          <img
            src={original.url}
            alt={t('preview.original')}
            draggable={false}
            style={layerStyle(original, g)}
          />
        )}
        {showResult && result && (
          <img
            src={result.url}
            alt={resultLabel}
            draggable={false}
            style={{ ...layerStyle(result, g), clipPath: resultClip(result) }}
          />
        )}
        {facesOn &&
          img.faces.map((f, i) => {
            const [x, y, w, h] = f.box
            const ex = excluded.has(i)
            return (
              <button
                key={i}
                type="button"
                title={t('preview.facesHint')}
                aria-label={`${t('preview.faces')} ${i + 1}${ex ? ' (aus)' : ''}`}
                aria-pressed={!ex}
                onPointerDown={(e) => e.stopPropagation()}
                onClick={() => toggleFace(i)}
                className="absolute rounded-[4px]"
                style={{
                  left: (x - g.view.x) * g.s,
                  top: (y - g.view.y) * g.s,
                  width: w * g.s,
                  height: h * g.s,
                  border: ex ? '1.5px dashed rgba(255,255,255,.6)' : '1.5px solid var(--accent)',
                  boxShadow: ex ? 'none' : '0 0 0 1px rgba(0,0,0,.35)',
                  background: 'transparent'
                }}
              />
            )
          })}
        {(which === 'left' ||
          (which === 'single' && effectiveStage === 'original') ||
          viewMode === 'split') && (
          <span
            className="pointer-events-none absolute rounded-[5px] bg-[rgba(14,14,16,.72)] px-[9px] py-1 text-[11.5px] font-semibold text-[#EDEDF0]"
            style={{ left: labelInset.left, top: labelInset.top }}
          >
            {t('preview.original')}
          </span>
        )}
        {showResult && (
          <span
            className="pointer-events-none absolute rounded-[5px] bg-[rgba(14,14,16,.72)] px-[9px] py-1 text-[11.5px] font-semibold text-[#EDEDF0]"
            style={{ right: labelInset.right, top: labelInset.top }}
          >
            {resultLabel}
          </span>
        )}
      </div>
    )
  }

  return (
    <div className="relative min-h-0 min-w-0 flex-[999_1_560px] bg-canvas p-0">
      <div
        ref={boxRef}
        className="absolute inset-0 cursor-grab overflow-hidden active:cursor-grabbing"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onWheel={onWheel}
        onDoubleClick={onDoubleClick}
      >
        {side ? (
          <>
            {pane('left')}
            {pane('right')}
          </>
        ) : (
          pane('single')
        )}
        {viewMode === 'split' && (
          <div
            className="absolute inset-y-0 z-10 w-6 -translate-x-1/2 cursor-ew-resize"
            style={{ left: splitX }}
            onPointerDown={(e) => {
              e.stopPropagation()
              ;(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId)
              splitDrag.current = true
            }}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
            role="slider"
            tabIndex={0}
            aria-label={t('preview.divider')}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Math.round(split)}
            onKeyDown={(e) => {
              if (e.key === 'ArrowLeft') set({ split: Math.max(0, split - 2) })
              if (e.key === 'ArrowRight') set({ split: Math.min(100, split + 2) })
            }}
          >
            <div className="absolute inset-y-0 left-1/2 w-0.5 -translate-x-1/2 bg-white shadow-[0_0_0_1px_rgba(0,0,0,.25)]" />
            <div className="absolute top-1/2 left-1/2 grid h-[34px] w-[34px] -translate-x-1/2 -translate-y-1/2 place-items-center rounded-full bg-white shadow-[0_2px_10px_rgba(0,0,0,.45)]">
              <svg
                width="18"
                height="18"
                viewBox="0 0 24 24"
                fill="none"
                stroke="#141416"
                strokeWidth="2.2"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden="true"
              >
                <path d="M9 6l-6 6 6 6M15 6l6 6-6 6" />
              </svg>
            </div>
          </div>
        )}
      </div>

      <div className="absolute bottom-3 left-3 z-20">
        <Segmented<Stage>
          dark
          label={t('preview.stage')}
          value={stage}
          onChange={(v) => set({ stage: v })}
          options={[
            { value: 'original', label: t('preview.original') },
            { value: 'denoised', label: t('preview.denoised') },
            { value: 'result', label: t('preview.result') }
          ]}
        />
      </div>
      <div className="absolute right-3 bottom-3 z-20 flex items-center gap-2">
        {(busy || error) && (
          <span
            role="status"
            className={`flex items-center gap-2 rounded-btn border px-2.5 py-1.5 text-[12px] backdrop-blur ${
              error
                ? 'border-danger/60 bg-[rgba(14,14,16,.85)] text-[#F26D6D]'
                : 'border-white/10 bg-[rgba(14,14,16,.78)] text-[#C9C9D0]'
            }`}
          >
            {!error && (
              <span className="pf-spin inline-block h-3 w-3 rounded-full border-2 border-[#F2A541] border-t-transparent" />
            )}
            {error ? t('preview.error', { msg: error }) : t('preview.computing')}
          </span>
        )}
        {img.faces.length > 0 && (
          <button
            type="button"
            aria-pressed={showFaces}
            onClick={() => set({ showFaces: !showFaces })}
            className={`flex h-[30px] items-center gap-1.5 rounded-btn border bg-[rgba(14,14,16,.78)] px-2.5 text-[12px] font-medium backdrop-blur ${
              showFaces ? 'border-[#F2A541] text-[#F2A541]' : 'border-white/10 text-[#C9C9D0]'
            }`}
            title="F"
          >
            <ScanFace size={15} strokeWidth={1.8} aria-hidden="true" />
            {t('preview.faces')}
          </button>
        )}
        <span
          className="rounded-btn border border-white/10 bg-[rgba(14,14,16,.78)] px-2.5 py-1.5 text-[12px] text-[#C9C9D0]"
          lang={i18n.language}
        >
          {zoomText}
        </span>
      </div>
    </div>
  )
}
