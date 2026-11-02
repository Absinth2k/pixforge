import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import type { Api } from '@/api/client'
import type { Post, Region, TileSession } from '@/api/types'

/** What the viewer needs to know about a view (source px, screen px per source px). */
export interface TileView {
  view: Region
  s: number
  z: number
}

const MAX_PARALLEL = 6
/** decoded tile pixels kept in memory (~4 bytes each) */
const MAX_CACHED_PX = 80e6

interface Cached {
  bmp: ImageBitmap
  px: number
  used: number
}

export const tileKey = (i: number, j: number): string => `${i},${j}`

/** Tiles (i, j) of the session grid that intersect ``r`` (source px). */
export function tilesIn(sess: TileSession, r: Region): [number, number][] {
  const t = sess.tile
  const i0 = Math.max(0, Math.floor(r.x / t))
  const j0 = Math.max(0, Math.floor(r.y / t))
  const i1 = Math.min(sess.cols - 1, Math.ceil((r.x + r.w) / t) - 1)
  const j1 = Math.min(sess.rows - 1, Math.ceil((r.y + r.h) / t) - 1)
  const out: [number, number][] = []
  for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) out.push([i, j])
  return out
}

/** Pyramid level whose pixels match the screen: level k has factor / 2^k px per source px. */
function levelFor(sess: TileSession, devicePxPerSource: number): number {
  const k = Math.floor(Math.log2(sess.factor / Math.max(devicePxPerSource, 1e-6)))
  return Math.max(0, Math.min(sess.levels - 1, k))
}

/**
 * Draws the finished background tiles of the real result. Tiles are fetched at the pyramid
 * level that matches the target zoom and drawn with edges snapped to device pixels, so the
 * grid has no visible seams. While a level is still loading, any other level of the same tile
 * is drawn instead (scaled), so zooming never shows holes.
 */
export function TileCanvas({
  api,
  session,
  done,
  draw,
  fetchView,
  width,
  height,
  clip,
  area,
  post
}: {
  api: Api
  session: TileSession
  /** finishing applied by the backend per request (instant, no model run) */
  post: Post
  done: Set<string>
  /** displayed (possibly animating) view */
  draw: TileView
  /** target view of the zoom animation: the level to fetch is chosen from it */
  fetchView: TileView
  width: number
  height: number
  clip: string
  /** only this part (crop) is drawn */
  area: Region
}) {
  const ref = useRef<HTMLCanvasElement>(null)
  const cache = useRef(new Map<string, Cached>())
  /** last decoded cache key per tile: drawn while another finishing or level is loading */
  const lastOf = useRef(new Map<string, string>())
  const pk = `${post.sh}/${post.dm}/${post.gr}`
  const loading = useRef(new Set<string>())
  const [tick, setTick] = useState(0)
  const dpr = window.devicePixelRatio || 1
  const sid = session.id

  // ---- fetch the tiles of the target view at their level --------------------------------
  const level = levelFor(session, fetchView.s * dpr)
  const vis = tilesIn(session, fetchView.view)
  const wanted = vis.filter(([i, j]) => done.has(tileKey(i, j)))
  const wantedKey = wanted.map(([i, j]) => tileKey(i, j)).join(';')
  useEffect(() => {
    const c = cache.current
    for (const [i, j] of wanted) {
      if (loading.current.size >= MAX_PARALLEL) break
      const key = `${sid}:${pk}:${level}:${i},${j}`
      if (c.has(key) || loading.current.has(key)) continue
      loading.current.add(key)
      fetch(api.tileUrl(sid, i, j, level, post))
        .then((r) => (r.ok ? r.blob() : Promise.reject(new Error(String(r.status)))))
        .then((b) => createImageBitmap(b))
        .then((bmp) => {
          c.set(key, { bmp, px: bmp.width * bmp.height, used: performance.now() })
          lastOf.current.set(`${sid}:${i},${j}`, key)
          let total = 0
          for (const v of c.values()) total += v.px
          if (total > MAX_CACHED_PX) {
            const old = [...c.entries()].sort((a, b) => a[1].used - b[1].used)
            for (const [k, v] of old) {
              if (total <= MAX_CACHED_PX * 0.8) break
              v.bmp.close()
              c.delete(k)
              total -= v.px
            }
          }
        })
        .catch(() => undefined) // a tile of an evicted session: simply not drawn
        .finally(() => {
          loading.current.delete(key)
          setTick((n) => n + 1) // draw it, and fetch the next ones
        })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [api, sid, pk, level, wantedKey, tick])

  // Free decoded tiles when the component goes away (image closed).
  useEffect(() => {
    const c = cache.current
    return () => {
      for (const v of c.values()) v.bmp.close()
      c.clear()
    }
  }, [])

  // ---- draw --------------------------------------------------------------------------
  useLayoutEffect(() => {
    const cv = ref.current
    if (!cv) return
    const W = Math.max(1, Math.round(width * dpr))
    const H = Math.max(1, Math.round(height * dpr))
    if (cv.width !== W) cv.width = W
    if (cv.height !== H) cv.height = H
    const ctx = cv.getContext('2d')
    if (!ctx) return
    ctx.clearRect(0, 0, W, H)
    const k = draw.s * dpr // device px per source px
    const X = (x: number) => Math.round((x - draw.view.x) * k)
    const Y = (y: number) => Math.round((y - draw.view.y) * k)
    ctx.save()
    ctx.beginPath()
    ctx.rect(X(area.x), Y(area.y), X(area.x + area.w) - X(area.x), Y(area.y + area.h) - Y(area.y))
    ctx.clip()
    ctx.imageSmoothingEnabled = draw.z < 2
    ctx.imageSmoothingQuality = 'high'
    const want = levelFor(session, k)
    const now = performance.now()
    const t = session.tile
    for (const [i, j] of tilesIn(session, draw.view)) {
      if (!done.has(tileKey(i, j))) continue
      // the matching level, else the closest one that is already decoded, else whatever
      // was shown last (e.g. the previous finishing while the new one loads)
      let hit: Cached | undefined
      for (let d = 0; d < session.levels && !hit; d++) {
        hit =
          cache.current.get(`${sid}:${pk}:${want + d}:${i},${j}`) ??
          cache.current.get(`${sid}:${pk}:${want - d}:${i},${j}`)
      }
      if (!hit) {
        const last = lastOf.current.get(`${sid}:${i},${j}`)
        hit = last ? cache.current.get(last) : undefined
      }
      if (!hit) continue
      hit.used = now
      const x0 = i * t
      const y0 = j * t
      const x1 = Math.min(session.width, x0 + t)
      const y1 = Math.min(session.height, y0 + t)
      ctx.drawImage(hit.bmp, X(x0), Y(y0), X(x1) - X(x0), Y(y1) - Y(y0))
    }
    ctx.restore()
  })

  return (
    <canvas
      ref={ref}
      data-testid="tile-canvas"
      data-tick={tick}
      className="pointer-events-none absolute top-0 left-0"
      style={{ width, height, clipPath: clip }}
    />
  )
}
