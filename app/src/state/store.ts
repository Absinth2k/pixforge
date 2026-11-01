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

  async connect() {
    const info = await window.pixforge.getBackend()
    if (!info.ok || !info.port || !info.token) {
      set({ backendError: info.error ?? 'unbekannt', backendLog: info.log ?? null })
      return
    }
    const api = new Api(info.port, info.token)
    set({ api, backendError: null, backendLog: null })
    const [health, models, jobs] = await Promise.all([api.health(), api.models(), api.jobs()])
    const byId: Record<string, Job> = {}
    for (const j of jobs.jobs) byId[j.id] = j
    set({ health, models, jobs: byId, queuePaused: jobs.paused })
    const queued = [...get().pendingPaths, ...(await window.pixforge.takePendingFiles())]
    set({ pendingPaths: [] })
    if (queued.length) void get().addPaths(queued)
    api.events((ev) => {
      if (ev.type === 'job') {
        const job = ev as unknown as Job
        set((s) => ({ jobs: { ...s.jobs, [job.id]: job } }))
        if (job.status === 'done' && job.output) {
          set({ toast: { text: job.output, path: job.output, kind: 'info' } })
        } else if (job.status === 'error') {
          set({ toast: { text: `${job.name}: ${job.error}`, kind: 'error' } })
        }
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
    const res = await api.addImages(fresh)
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
    await get().api?.removeImage(id)
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
    for (const id of ids) {
      const job = await api.submitJob(id, settings[id] ?? defaultSettings())
      set((s) => ({ jobs: { ...s.jobs, [job.id]: job } }))
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
