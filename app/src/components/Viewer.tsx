import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import type { PointerEvent as ReactPointerEvent, WheelEvent as ReactWheelEvent } from 'react'
import { Download, RotateCw, ScanFace } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { ApiError } from '@/api/client'
import type { ImageInfo, Region, Settings, Stage, TileSession } from '@/api/types'
import { modelSettingsKey, postOf, requiredModels } from '@/api/types'
import { tileBus, useStore } from '@/state/store'
import { applyAspect, clampRect, CropOverlay } from './CropOverlay'
import { downloadProgress, formatBytes } from './ModelSetup'
import { TileCanvas, tileKey, tilesIn } from './TileCanvas'
import { nf, Segmented } from './ui'

const PREVIEW_DEBOUNCE_MS = 280
const TILES_DEBOUNCE_MS = 300
const ORIGINAL_DEBOUNCE_MS = 60
const ZOOM_ANIM_MS = 150
const MAX_OUT = 4096
const BASE_MAX = 2048

interface Tiles {
  key: string
  sess: TileSession
  done: Set<string>
}
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
  cx: number
  cy: number
  view: Region // visible source region (may extend past the image)
}

function geometry(
  img: ImageInfo,
  factor: number,
  paneW: number,
  paneH: number,
  zoom: number | null | undefined,
  cx?: number,
  cy?: number
): Geometry {
  const fitZ = Math.min(paneW / (img.width * factor), paneH / (img.height * factor))
  const z = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, zoom ?? fitZ))
  const s = z * factor
  const w = paneW / s
  const h = paneH / s
  const clampC = (c: number | undefined, size: number, view: number): number => {
    if (size <= view) return size / 2
    const v = c ?? size / 2
    return Math.min(size - view / 2, Math.max(view / 2, v))
  }
  const ccx = clampC(cx, img.width, w)
  const ccy = clampC(cy, img.height, h)
  return {
    paneW,
    paneH,
    s,
    z,
    fitZ,
    cx: ccx,
    cy: ccy,
    view: { x: ccx - w / 2, y: ccy - h / 2, w, h }
  }
}

function intersect(a: Region, b: Region): Region | null {
  const x0 = Math.max(a.x, b.x)
  const y0 = Math.max(a.y, b.y)
  const x1 = Math.min(a.x + a.w, b.x + b.w)
  const y1 = Math.min(a.y + a.h, b.y + b.h)
  return x1 - x0 >= 1 && y1 - y0 >= 1 ? { x: x0, y: y0, w: x1 - x0, h: y1 - y0 } : null
}

const ASPECTS: { key: string; ratio: number | null | 'original' }[] = [
  { key: 'free', ratio: null },
  { key: 'original', ratio: 'original' },
  { key: '1:1', ratio: 1 },
  { key: '4:3', ratio: 4 / 3 },
  { key: '3:2', ratio: 3 / 2 },
  { key: '16:9', ratio: 16 / 9 }
]

function visibleRegion(img: ImageInfo, g: Geometry): Region {
  const x0 = Math.max(0, g.view.x)
  const y0 = Math.max(0, g.view.y)
  const x1 = Math.min(img.width, g.view.x + g.view.w)
  const y1 = Math.min(img.height, g.view.y + g.view.h)
  return { x: x0, y: y0, w: Math.max(1, x1 - x0), h: Math.max(1, y1 - y0) }
}

function outSize(r: Region, s: number, cap = MAX_OUT, useDpr = true): [number, number] {
  const dpr = useDpr ? window.devicePixelRatio || 1 : 1
  let ow = r.w * s * dpr
  let oh = r.h * s * dpr
  const k = Math.min(1, cap / Math.max(ow, oh))
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
    maxWidth: 'none',
    imageRendering: g.z >= 2 ? 'pixelated' : 'auto'
  }
}

const easeOut = (t: number): number => 1 - Math.pow(1 - t, 3)

/** Replace a layer and free the old object URL. */
function useBlobLayer(): [Layer | null, (l: Layer | null) => void] {
  const [layer, setLayer] = useState<Layer | null>(null)
  const replace = useCallback((l: Layer | null) => {
    setLayer((old) => {
      if (old && old.url.startsWith('blob:') && old.url !== l?.url) URL.revokeObjectURL(old.url)
      return l
    })
  }, [])
  return [layer, replace]
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
  const cropEditing = useStore((s) => s.cropEditing)
  const loadingModel = useStore((s) => s.loadingModel)
  const models = useStore((s) => s.models)
  const downloads = useStore((s) => s.downloads)
  const modelsRev = useStore((s) => s.modelsRev)
  const downloadModels = useStore((s) => s.downloadModels)
  const [aspectKey, setAspectKey] = useState('free')
  const [portrait, setPortrait] = useState(false)

  const boxRef = useRef<HTMLDivElement>(null)
  const [box, setBox] = useState({ w: 800, h: 600 })
  const [baseOriginal, setBaseOriginal] = useState<Layer | null>(null)
  const [original, setOriginal] = useState<Layer | null>(null)
  const [result, setResult] = useBlobLayer()
  const [tiles, setTiles] = useState<Tiles | null>(null)
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

  // Target view (what the store says) – network requests are based on this.
  const target = useMemo(
    () => geometry(img, factor, paneW, box.h, vs?.zoom, vs?.cx, vs?.cy),
    [img, factor, paneW, box.h, vs]
  )

  // Displayed view – animated towards the target so zooming glides instead of jumping.
  const [disp, setDisp] = useState({ z: target.z, cx: target.cx, cy: target.cy })
  const dispRef = useRef(disp)
  useLayoutEffect(() => {
    dispRef.current = disp
  }, [disp])
  const skipAnim = useRef(true)
  useEffect(() => {
    const to = { z: target.z, cx: target.cx, cy: target.cy }
    if (skipAnim.current) {
      skipAnim.current = false
      setDisp(to)
      return
    }
    const from = { ...dispRef.current }
    if (
      Math.abs(from.z - to.z) < 1e-9 &&
      Math.abs(from.cx - to.cx) < 0.01 &&
      Math.abs(from.cy - to.cy) < 0.01
    ) {
      return
    }
    let raf = 0
    const t0 = performance.now()
    const step = (now: number) => {
      const k = easeOut(Math.min(1, (now - t0) / ZOOM_ANIM_MS))
      setDisp({
        z: Math.exp(Math.log(from.z) + (Math.log(to.z) - Math.log(from.z)) * k),
        cx: from.cx + (to.cx - from.cx) * k,
        cy: from.cy + (to.cy - from.cy) * k
      })
      if (k < 1) raf = requestAnimationFrame(step)
    }
    raf = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf)
  }, [target.z, target.cx, target.cy])

  const g = useMemo(
    () => geometry(img, factor, paneW, box.h, disp.z, disp.cx, disp.cy),
    [img, factor, paneW, box.h, disp]
  )
  const crop: Region | null = useMemo(() => {
    const c = settings?.crop
    return c ? { x: c[0], y: c[1], w: c[2], h: c[3] } : null
  }, [settings?.crop])
  const visible = useMemo(() => visibleRegion(img, target), [img, target])
  // With a crop only its visible part is computed: faster previews, same as the export.
  const region = useMemo(() => (crop ? intersect(visible, crop) : visible), [crop, visible])
  const area = crop ?? { x: 0, y: 0, w: img.width, h: img.height }

  useEffect(() => {
    const z = vs?.zoom ?? null
    onZoomPreset(
      z === null ? 'fit' : Math.abs(z - 1) < 1e-6 ? 1 : Math.abs(z - 2) < 1e-6 ? 2 : null
    )
  }, [vs?.zoom, onZoomPreset])

  // New image: drop all layers, jump (no animation) to its view.
  useEffect(() => {
    setOriginal(null)
    setBaseOriginal(null)
    setResult(null)
    setTiles(null)
    setError(null)
    skipAnim.current = true
  }, [img.id, setResult])

  // Base original: the whole image at screen-like resolution, loaded once per image.
  useEffect(() => {
    if (!api) return
    const full = { x: 0, y: 0, w: img.width, h: img.height }
    const k = Math.min(1, BASE_MAX / Math.max(img.width, img.height))
    const url = api.viewUrl(img.id, full, Math.round(img.width * k), Math.round(img.height * k))
    const pre = new Image()
    pre.onload = () => setBaseOriginal({ url, region: full })
    pre.src = url
  }, [api, img.id, img.width, img.height])

  // Sharp original for the visible region.
  const keyOf = (r: Region | null) =>
    r ? `${r.x.toFixed(1)},${r.y.toFixed(1)},${r.w.toFixed(1)},${r.h.toFixed(1)}` : 'none'
  const visibleKey = keyOf(visible)
  const [vw, vh] = outSize(visible, target.s)
  useEffect(() => {
    if (!api) return
    const timer = setTimeout(() => {
      const url = api.viewUrl(img.id, visible, vw, vh)
      const pre = new Image()
      pre.onload = () => setOriginal({ url, region: visible })
      pre.src = url
    }, ORIGINAL_DEBOUNCE_MS)
    return () => clearTimeout(timer)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [api, img.id, visibleKey, vw, vh])

  const regionKey = keyOf(region)
  // Zoomed out the quick preview is only a stand-in until the tiles arrive: CSS pixels
  // are enough (a 1.5x HiDPI screen would otherwise cost 2.25x the GPU time).
  const [ow, oh] = region ? outSize(region, target.s, MAX_OUT, target.z >= 1) : [1, 1]

  // Models these settings need that are not downloaded yet: no preview requests until they are.
  const missing = settings
    ? requiredModels(settings, factor, img.faces.length > 0).filter(
        (id) => models.find((m) => m.id === id)?.needed
      )
    : []
  const missingKey = missing.join(',')
  const missingDl = downloadProgress(missing, models, downloads)
  const missingNames = missing
    .map((id) => {
      const m = models.find((x) => x.id === id)
      return `„${m ? m.name[i18n.language === 'de' ? 'de' : 'en'] : id}“`
    })
    .join(', ')

  // ---- background tiles of the real result -----------------------------------------
  const effectiveStage: Stage = viewMode === 'single' || stage !== 'original' ? stage : 'result'
  const settingsKey = JSON.stringify(settings)
  const tileStage = effectiveStage === 'original' ? null : effectiveStage
  // The crop only limits which tiles are computed; their content stays valid.
  // Nor does the finishing: the backend applies it per tile request, without a model run.
  const sessKey = `${img.id}|${tileStage}|${settings ? modelSettingsKey(settings) : ''}`
  const cropKey = keyOf(crop)
  useEffect(() => {
    if (!api || !settings || !tileStage || missingKey) return
    const ctrl = new AbortController()
    const timer = setTimeout(async () => {
      try {
        const sess = await api.tiles(img.id, settings, tileStage, visible, ctrl.signal)
        setTiles((old) => {
          const done = new Set(sess.done.map(([i, j]) => tileKey(i, j)))
          if (old && old.sess.id === sess.id) for (const k of old.done) done.add(k)
          return { key: sessKey, sess, done }
        })
      } catch {
        /* the tiles improve the preview; the quick preview keeps working without them */
      }
    }, TILES_DEBOUNCE_MS)
    return () => {
      clearTimeout(timer)
      ctrl.abort()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [api, img.id, sessKey, cropKey, visibleKey, missingKey, modelsRev])

  const sid = tiles?.sess.id
  useEffect(() => {
    if (!sid) return
    const on = (e: Event) => {
      const ev = (e as CustomEvent<Record<string, unknown>>).detail
      if (ev.session !== sid) return
      setTiles((old) => {
        if (!old || old.sess.id !== sid) return old
        const counts = { done_count: Number(ev.done), total: Number(ev.total) }
        if (ev.type === 'tile') {
          const done = new Set(old.done)
          done.add(tileKey(Number(ev.i), Number(ev.j)))
          return { ...old, done, sess: { ...old.sess, ...counts } }
        }
        const state = ev.state as TileSession['state']
        const error = (ev.error as string | null) ?? null
        return { ...old, sess: { ...old.sess, ...counts, state, error } }
      })
    }
    tileBus.addEventListener('tile', on)
    return () => tileBus.removeEventListener('tile', on)
  }, [sid])

  // tiles of the current settings only: after a change the old ones are wrong
  const live = tiles && tiles.key === sessKey ? tiles : null
  const isDone = (t: [number, number]) => !!live && live.done.has(tileKey(t[0], t[1]))
  const regionTiles = live && region ? tilesIn(live.sess, region) : []
  const covered = regionTiles.length > 0 && regionTiles.every(isDone)
  const visTiles = live ? tilesIn(live.sess, region ?? visible) : []
  const pct = visTiles.length
    ? Math.floor((100 * visTiles.filter(isDone).length) / visTiles.length)
    : 100
  const refining = !!live && pct < 100

  // ---- quick preview of the visible region: debounced, cancellable, stale-while-loading
  useEffect(() => {
    if (!api || !settings || effectiveStage === 'original' || !region || missingKey) return
    if (covered) {
      setBusy(false) // the real result is already there
      return
    }
    const ctrl = new AbortController()
    setBusy(true)
    const timer = setTimeout(async () => {
      try {
        const res = await api.preview(img.id, settings, region, ow, oh, effectiveStage, ctrl.signal)
        if (!res) return
        setResult({ url: URL.createObjectURL(res.blob), region })
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
  }, [api, img.id, regionKey, ow, oh, settingsKey, effectiveStage, covered, missingKey, modelsRev])

  const ratioFor = (key: string): number | null => {
    const a = ASPECTS.find((x) => x.key === key)?.ratio ?? null
    let r = a === 'original' ? img.width / img.height : a
    if (r && r !== 1 && portrait !== r < 1) r = 1 / r
    return r
  }
  const setCrop = (r: Region) =>
    updateSettings(img.id, (s) => {
      s.crop = [Math.round(r.x), Math.round(r.y), Math.round(r.w), Math.round(r.h)]
      return s
    })
  const pickAspect = (key: string, flip = portrait) => {
    setAspectKey(key)
    setPortrait(flip)
    const a = ASPECTS.find((x) => x.key === key)?.ratio ?? null
    let r = a === 'original' ? img.width / img.height : a
    if (r && r !== 1 && flip !== r < 1) r = 1 / r
    if (r && crop) setCrop(applyAspect(crop, r, img.width, img.height))
  }
  const cropLabel = crop
    ? t('crop.label', {
        w: nf(Math.round(crop.w), i18n.language),
        h: nf(Math.round(crop.h), i18n.language),
        ow: nf(Math.round(crop.w * factor), i18n.language),
        oh: nf(Math.round(crop.h * factor), i18n.language)
      })
    : ''

  // ---- interaction -----------------------------------------------------------------
  const commit = useCallback(
    (z: number | null, cx: number, cy: number, animate = true) => {
      if (!animate) skipAnim.current = true
      setView(img.id, { zoom: z, cx, cy })
    },
    [img.id, setView]
  )
  const dragRef = useRef<{ x: number; y: number; cx: number; cy: number; moved: boolean } | null>(
    null
  )
  const splitDrag = useRef(false)

  const onPointerDown = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return
    ;(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId)
    dragRef.current = { x: e.clientX, y: e.clientY, cx: g.cx, cy: g.cy, moved: false }
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
    // panning follows the pointer directly, without animation
    commit(vs?.zoom ?? null, d.cx - dx / g.s, d.cy - dy / g.s, false)
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
    // the point under the cursor, in what the user currently sees
    const sx = g.view.x + px / g.s
    const sy = g.view.y + py / g.s
    // proportional to the wheel/touchpad delta: small touchpad moves zoom a little
    const delta = e.deltaMode === 1 ? e.deltaY * 40 : e.deltaY
    const k = Math.min(2, Math.max(0.5, Math.exp(-delta * 0.0018)))
    const minZ = Math.min(target.fitZ, 1)
    const z2 = Math.min(MAX_ZOOM, Math.max(minZ, target.z * k))
    if (z2 <= target.fitZ * 1.0001) {
      commit(null, img.width / 2, img.height / 2) // snap to "fit" instead of clamping jumps
      return
    }
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
        {showOriginal && baseOriginal && (
          <img
            src={baseOriginal.url}
            alt=""
            draggable={false}
            style={layerStyle(baseOriginal, g)}
          />
        )}
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
        {showResult && live && api && (
          <TileCanvas
            api={api}
            session={live.sess}
            done={live.done}
            draw={{ view: g.view, s: g.s, z: g.z }}
            fetchView={{ view: target.view, s: target.s, z: target.z }}
            width={paneW}
            height={box.h}
            clip={viewMode === 'split' ? `inset(0 0 0 ${Math.max(0, splitX)}px)` : 'none'}
            area={area}
            post={settings ? postOf(settings) : { sh: 0, dm: 0, gr: 0 }}
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
        {crop && (
          <CropOverlay
            crop={crop}
            view={g.view}
            s={g.s}
            editing={cropEditing}
            aspect={ratioFor(aspectKey)}
            imgW={img.width}
            imgH={img.height}
            label={cropLabel}
            onChange={(r) => setCrop(clampRect(r, img.width, img.height))}
          />
        )}
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

      {missing.length > 0 && effectiveStage !== 'original' && (
        <div
          role="alert"
          data-testid="missing-model"
          className="absolute top-1/2 left-1/2 z-30 w-[360px] max-w-[90%] -translate-x-1/2 -translate-y-1/2 rounded-dialog border border-line bg-panel p-5 shadow-2xl"
        >
          <h3 className="m-0 text-[15px] font-semibold">{t('models.missingTitle')}</h3>
          <p className="mt-2 text-[13px] leading-relaxed text-text2">
            {t('models.missingText', {
              names: missingNames,
              size: formatBytes(missingDl.total, i18n.language)
            })}
          </p>
          {missingDl.error && <p className="mt-2 text-[12.5px] text-danger">{missingDl.error}</p>}
          <div className="mt-3 flex justify-end">
            {missingDl.active ? (
              <span className="text-[13px] text-text2" role="status">
                {t('models.downloading', {
                  pct: missingDl.total ? Math.floor((100 * missingDl.done) / missingDl.total) : 0
                })}
              </span>
            ) : (
              <button
                type="button"
                onClick={() => void downloadModels(missing)}
                className="flex h-9 items-center gap-2 rounded-btn bg-accent px-4 font-semibold text-on-accent hover:bg-accent-hover"
              >
                <Download size={15} aria-hidden="true" />
                {missingDl.error ? t('models.retry') : t('models.download')}
              </button>
            )}
          </div>
        </div>
      )}
      {cropEditing && crop && (
        <div className="absolute top-3 left-1/2 z-30 flex -translate-x-1/2 items-center gap-2">
          <Segmented<string>
            dark
            label={t('crop.aspect')}
            value={aspectKey}
            onChange={(k) => pickAspect(k)}
            options={ASPECTS.map((a) => ({
              value: a.key,
              label:
                a.key === 'free'
                  ? t('crop.free')
                  : a.key === 'original'
                    ? t('crop.original')
                    : a.key
            }))}
          />
          <button
            type="button"
            aria-label={t('crop.orientation')}
            title={t('crop.orientation')}
            onClick={() => pickAspect(aspectKey, !portrait)}
            className="grid h-[34px] w-[34px] place-items-center rounded-lg border border-white/10 bg-[rgba(14,14,16,.82)] text-[#EDEDF0] backdrop-blur hover:bg-[rgba(40,40,46,.9)]"
          >
            <RotateCw size={15} aria-hidden="true" />
          </button>
          <button
            type="button"
            onClick={() => {
              updateSettings(img.id, (s) => ({ ...s, crop: null }))
              set({ cropEditing: false })
            }}
            className="h-[34px] rounded-lg border border-white/10 bg-[rgba(14,14,16,.82)] px-3 text-[12.5px] text-[#EDEDF0] backdrop-blur hover:bg-[rgba(40,40,46,.9)]"
          >
            {t('crop.remove')}
          </button>
          <button
            type="button"
            onClick={() => set({ cropEditing: false })}
            className="h-[34px] rounded-lg bg-accent px-3.5 text-[12.5px] font-semibold text-on-accent hover:bg-accent-hover"
          >
            {t('crop.done')}
          </button>
        </div>
      )}
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
        {(busy || error || loadingModel || refining) && (
          <span
            role="status"
            data-testid="preview-status"
            title={refining ? t('preview.refiningHint') : undefined}
            className={`flex items-center gap-2 rounded-btn border px-2.5 py-1.5 text-[12px] backdrop-blur ${
              error
                ? 'border-danger/60 bg-[rgba(14,14,16,.85)] text-[#F26D6D]'
                : 'border-white/10 bg-[rgba(14,14,16,.78)] text-[#C9C9D0]'
            }`}
          >
            {!error && (
              <span className="pf-spin inline-block h-3 w-3 rounded-full border-2 border-[#F2A541] border-t-transparent" />
            )}
            {error
              ? t('preview.error', { msg: error })
              : loadingModel
                ? t('preview.loadingModel')
                : refining && live
                  ? live.sess.state === 'paused'
                    ? t('preview.refiningPaused', { pct })
                    : live.sess.state === 'error'
                      ? t('preview.refiningError', { msg: live.sess.error ?? '' })
                      : t('preview.refining', { pct })
                  : t('preview.computing')}
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
          data-testid="zoom-readout"
        >
          {zoomText}
        </span>
      </div>
    </div>
  )
}
