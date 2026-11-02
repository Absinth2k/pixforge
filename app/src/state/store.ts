import { create } from 'zustand'
import { Api } from '@/api/client'
import {
  defaultSettings,
  type Health,
  type ImageInfo,
  type Job,
  type ModelInfo,
  type Settings,
  type Stage
} from '@/api/types'
import { applyTheme, type ThemeName } from '@/theme/theme'

export type ViewMode = 'single' | 'split' | 'side'

/** Background tile events ("tile", "tiles") from the backend, for the viewer. */
export const tileBus = new EventTarget()

/** Closes the event socket of the previous backend (after a restart). */
let closeEvents: (() => void) | null = null

/** A request failed because the backend is gone (closed, crashed, restarting). */
function connectionMessage(e: unknown): string {
  const msg = e instanceof Error ? e.message : String(e)
  return /failed to fetch|networkerror|load failed/i.test(msg)
    ? 'Keine Verbindung zum Rechenkern. Pixforge bitte neu starten, falls das so bleibt.'
    : msg
}

export interface ViewState {
  /** screen px per output px; null = fit */
  zoom: number | null
  /** centre of the view in source pixels */
  cx: number
  cy: number
}

interface State {
  api: Api | null
  backendError: string | null
  backendLog: string | null
  health: Health | null
  models: ModelInfo[]
  images: ImageInfo[]
  selectedId: string | null
  settings: Record<string, Settings>
  jobs: Record<string, Job>
  queuePaused: boolean
  viewMode: ViewMode
  stage: Stage
  split: number
  showFaces: boolean
  cropEditing: boolean
  view: Record<string, ViewState>
  theme: ThemeName
  toast: { text: string; path?: string; kind: 'info' | 'error' } | null
  pendingPaths: string[]
  loadingModel: string | null

  connect(): Promise<void>
  addPaths(paths: string[]): Promise<void>
  removeImage(id: string): Promise<void>
  select(id: string | null): void
  updateSettings(id: string, fn: (s: Settings) => Settings): void
  resetSettings(id: string): void
  setView(id: string, v: ViewState): void
  set(partial: Partial<State>): void
  exportImages(ids: string[]): Promise<void>
  setTheme(t: ThemeName): void
}

const THEME_KEY = 'pixforge.theme'

function storedTheme(): ThemeName {
  try {
    const t = localStorage.getItem(THEME_KEY)
    return t === 'light' ? 'light' : 'dark'
  } catch {
    return 'dark'
  }
}

export const useStore = create<State>((set, get) => ({
  api: null,
  backendError: null,
  backendLog: null,
  health: null,
  models: [],
  images: [],
  selectedId: null,
  settings: {},
  jobs: {},
  queuePaused: false,
  viewMode: 'split',
  stage: 'result',
  split: 50,
  showFaces: true,
  cropEditing: false,
  view: {},
  theme: storedTheme(),
  toast: null,
  pendingPaths: [],
  loadingModel: null,

  async connect() {
    const info = await window.pixforge.getBackend()
    if (!info.ok || !info.port || !info.token) {
      set({ backendError: info.error ?? 'unbekannt', backendLog: info.log ?? null })
      return
    }
    const api = new Api(info.port, info.token)
    set({ api, backendError: null, backendLog: null })
    let health: Health, models: ModelInfo[], jobs: { jobs: Job[]; paused: boolean }
    try {
      ;[health, models, jobs] = await Promise.all([api.health(), api.models(), api.jobs()])
    } catch (e) {
      set({ backendError: connectionMessage(e) })
      return
    }
    const byId: Record<string, Job> = {}
    for (const j of jobs.jobs) byId[j.id] = j
    set({ health, models, jobs: byId, queuePaused: jobs.paused })
    // After a restart of the backend it knows none of the open images: register them again
    // and carry settings, view and selection over to their new ids.
    const open = get().images
    if (open.length) {
      try {
        const res = await api.addImages(open.map((i) => i.path))
        const byPath = new Map(res.images.map((i) => [i.path.toLowerCase(), i]))
        set((s) => {
          const ids = new Map<string, string>()
          const images: ImageInfo[] = []
          for (const old of s.images) {
            const fresh = byPath.get(old.path.toLowerCase())
            if (!fresh) continue // file gone meanwhile
            ids.set(old.id, fresh.id)
            images.push(fresh)
          }
          const settings: Record<string, Settings> = {}
          const view: Record<string, ViewState> = {}
          for (const [o, n] of ids) {
            if (s.settings[o]) settings[n] = s.settings[o]
            if (s.view[o]) view[n] = s.view[o]
          }
          const sel = s.selectedId ? (ids.get(s.selectedId) ?? null) : null
          return { images, settings, view, selectedId: sel ?? images[0]?.id ?? null }
        })
      } catch (e) {
        set({ toast: { text: connectionMessage(e), kind: 'error' } })
      }
    }
    const queued = [...get().pendingPaths, ...(await window.pixforge.takePendingFiles())]
    set({ pendingPaths: [] })
    if (queued.length) void get().addPaths(queued)
    closeEvents?.()
    closeEvents = api.events((ev) => {
      if (ev.type === 'job') {
        const job = ev as unknown as Job
        set((s) => ({ jobs: { ...s.jobs, [job.id]: job } }))
        if (job.status === 'done' && job.output) {
          set({ toast: { text: job.output, path: job.output, kind: 'info' } })
        } else if (job.status === 'error') {
          set({ toast: { text: `${job.name}: ${job.error}`, kind: 'error' } })
        }
      } else if (ev.type === 'model') {
        set({ loadingModel: ev.loading ? String(ev.id) : null })
      } else if (ev.type === 'tile' || ev.type === 'tiles') {
        tileBus.dispatchEvent(new CustomEvent('tile', { detail: ev }))
      } else if (ev.type === 'queue' && typeof ev.paused === 'boolean') {
        set({ queuePaused: ev.paused })
      }
    })
  },

  async addPaths(paths) {
    const api = get().api
    if (!paths.length) return
    if (!api) {
      // backend not ready yet: keep them, connect() loads them afterwards
      set((s) => ({ pendingPaths: [...s.pendingPaths, ...paths] }))
      return
    }
    const known = new Set(get().images.map((i) => i.path.toLowerCase()))
    const fresh = paths.filter((p) => !known.has(p.toLowerCase()))
    if (!fresh.length) return
    let res: Awaited<ReturnType<Api['addImages']>>
    try {
      res = await api.addImages(fresh)
    } catch (e) {
      set({ toast: { text: connectionMessage(e), kind: 'error' } })
      return
    }
    set((s) => {
      const settings = { ...s.settings }
      for (const img of res.images) {
        const st = defaultSettings()
        // Faces: switch suggested on when faces were found (prompt: per-image toggle).
        st.faces.enabled = img.faces.length > 0
        settings[img.id] = st
      }
      return {
        images: [...s.images, ...res.images],
        settings,
        selectedId: s.selectedId ?? res.images[0]?.id ?? null
      }
    })
    if (res.errors.length) {
      set({ toast: { text: res.errors.map((e) => e.error).join('\n'), kind: 'error' } })
    }
  },

  async removeImage(id) {
    try {
      await get().api?.removeImage(id)
    } catch {
      /* gone on the backend side anyway: remove it from the list */
    }
    set((s) => {
      const images = s.images.filter((i) => i.id !== id)
      const idx = s.images.findIndex((i) => i.id === id)
      const next =
        s.selectedId === id ? (images[Math.min(idx, images.length - 1)]?.id ?? null) : s.selectedId
      return { images, selectedId: next }
    })
  },

  select(id) {
    set({ selectedId: id, cropEditing: false })
  },

  updateSettings(id, fn) {
    set((s) => ({
      settings: { ...s.settings, [id]: fn(structuredClone(s.settings[id] ?? defaultSettings())) }
    }))
  },

  resetSettings(id) {
    const img = get().images.find((i) => i.id === id)
    const st = defaultSettings()
    st.faces.enabled = !!img && img.faces.length > 0
    set((s) => ({ settings: { ...s.settings, [id]: st } }))
  },

  setView(id, v) {
    set((s) => ({ view: { ...s.view, [id]: v } }))
  },

  set(partial) {
    set(partial)
  },

  async exportImages(ids) {
    const { api, settings } = get()
    if (!api) return
    try {
      for (const id of ids) {
        const job = await api.submitJob(id, settings[id] ?? defaultSettings())
        set((s) => ({ jobs: { ...s.jobs, [job.id]: job } }))
      }
    } catch (e) {
      set({ toast: { text: connectionMessage(e), kind: 'error' } })
    }
  },

  setTheme(t) {
    applyTheme(t)
    try {
      localStorage.setItem(THEME_KEY, t)
    } catch {
      /* private mode etc. */
    }
    set({ theme: t })
  }
}))

/** Latest job per image (for thumbnails and the status bar). */
export function latestJob(jobs: Record<string, Job>, imageId: string): Job | undefined {
  let last: Job | undefined
  for (const j of Object.values(jobs)) if (j.image_id === imageId) last = j
  return last
}
