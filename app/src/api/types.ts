/** Shapes of the backend API (see backend/src/pixforge/server/app.py). */

export type Task = 'upscale' | 'denoise'
export type Stage = 'original' | 'denoised' | 'result'

export interface DenoiseSettings {
  mode: 'auto' | 'on' | 'off'
  strength: number | null
  model: string
  detail_protect: number
}

export interface FaceSettings {
  enabled: boolean | null
  model: string
  strength: number
  min_size: number
  only_largest: boolean
  exclude: number[]
}

export interface Settings {
  task: Task
  model: string
  scale: number
  target_width: number | null
  target_height: number | null
  denoise: DenoiseSettings
  jpeg_strength: number
  sharpen: number
  detail_mix: number
  grain: number
  faces: FaceSettings
  crop: [number, number, number, number] | null
}

export interface DenoiseSuggestion {
  strength: number
  reason: 'already_denoised' | 'clean' | 'model_handles' | 'noisy'
  message: string
  message_en: string
}

export interface FaceInfo {
  box: [number, number, number, number]
  landmarks: number[][]
  score: number
}

export interface ImageInfo {
  id: string
  path: string
  name: string
  width: number
  height: number
  bit_depth: number
  format: string
  has_alpha: boolean
  meta: Record<string, string | number | boolean>
  noise: { luma: number; chroma: number; grayscale: boolean }
  denoise: { photo: DenoiseSuggestion; clean: DenoiseSuggestion }
  faces: FaceInfo[]
  color_space: string
}

export interface DeviceInfo {
  name: string
  kind: 'gpu' | 'cpu'
  provider: string
  vram_mb: number | null
  discrete: boolean | null
  device_index: number | null
}

export interface Health {
  version: string
  device: DeviceInfo
  devices: DeviceInfo[]
  missing_defaults: string[]
}

export interface ModelInfo {
  id: string
  kind: 'upscale' | 'denoise' | 'jpeg' | 'face' | 'detect'
  name: { de: string; en: string }
  description: { de: string; en: string }
  scale: number
  license: string
  noncommercial: boolean
  default: boolean
  installed: string[]
  size: Record<string, number>
}

export interface Estimate {
  in_w: number
  in_h: number
  out_w: number
  out_h: number
  factor: number
  megapixels: number
  seconds: number
  warnings: string[]
}

export type JobStatus = 'queued' | 'running' | 'done' | 'error' | 'cancelled' | 'skipped'

export interface Job {
  id: string
  image_id: string
  name: string
  status: JobStatus
  stage: string
  progress: number
  eta_s: number | null
  output: string | null
  error: string | null
  warnings: string[]
  seconds: number | null
}

export interface Region {
  x: number
  y: number
  w: number
  h: number
}

export const defaultSettings = (): Settings => ({
  task: 'upscale',
  model: 'photo',
  scale: 4,
  target_width: null,
  target_height: null,
  denoise: { mode: 'auto', strength: null, model: 'denoise', detail_protect: 60 },
  jpeg_strength: 0,
  sharpen: 0,
  detail_mix: 0,
  grain: 0,
  faces: {
    enabled: null,
    model: 'face',
    strength: 80,
    min_size: 16,
    only_largest: false,
    exclude: []
  },
  crop: null
})
