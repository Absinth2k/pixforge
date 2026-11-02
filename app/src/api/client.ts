import type {
  Estimate,
  Health,
  ImageInfo,
  Job,
  ModelInfo,
  Region,
  Post,
  Settings,
  Stage,
  TileSession
} from './types'
import { wireSettings } from './types'

export type PreviewKind = 'exact' | 'approx' | 'original'

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string
  ) {
    super(message)
  }
}

export class Api {
  readonly base: string

  constructor(
    port: number,
    private token: string
  ) {
    this.base = `http://127.0.0.1:${port}`
  }

  private async req<T>(
    method: string,
    path: string,
    body?: unknown,
    signal?: AbortSignal
  ): Promise<T> {
    const res = await fetch(this.base + path, {
      method,
      signal,
      headers: { 'X-Pixforge-Token': this.token, 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body)
    })
    if (!res.ok) {
      let msg = res.statusText
      try {
        const data = await res.json()
        msg = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail)
      } catch {
        /* not JSON */
      }
      throw new ApiError(res.status, msg)
    }
    return (await res.json()) as T
  }

  health = (): Promise<Health> => this.req('GET', '/api/health')
  models = (): Promise<ModelInfo[]> => this.req('GET', '/api/models')
  addImages = (
    paths: string[]
  ): Promise<{ images: ImageInfo[]; errors: { path: string; error: string }[] }> =>
    this.req('POST', '/api/images', { paths })
  removeImage = (id: string): Promise<unknown> => this.req('DELETE', `/api/images/${id}`)
  estimate = (imageId: string, settings: Settings, signal?: AbortSignal): Promise<Estimate> =>
    this.req(
      'POST',
      '/api/estimate',
      { image_id: imageId, settings: wireSettings(settings) },
      signal
    )
  submitJob = (imageId: string, settings: Settings): Promise<Job> =>
    this.req('POST', '/api/jobs', { image_id: imageId, settings: wireSettings(settings) })
  jobs = (): Promise<{ jobs: Job[]; paused: boolean }> => this.req('GET', '/api/jobs')
  cancelJob = (id: string): Promise<unknown> => this.req('POST', `/api/jobs/${id}/cancel`)
  pause = (paused: boolean): Promise<unknown> =>
    this.req('POST', `/api/queue/pause?paused=${paused}`)

  /** Start or continue the background tiles; ``focus`` (visible region) goes first. */
  tiles = (
    imageId: string,
    settings: Settings,
    stage: Exclude<Stage, 'original'>,
    focus: Region | null,
    signal?: AbortSignal
  ): Promise<TileSession> =>
    this.req(
      'POST',
      '/api/tiles',
      { image_id: imageId, settings: wireSettings(settings), stage, focus },
      signal
    )

  tileUrl(sid: string, i: number, j: number, level: number, post?: Post): string {
    const q = new URLSearchParams({ t: this.token })
    if (post) {
      q.set('sh', String(post.sh))
      q.set('dm', String(post.dm))
      q.set('gr', String(post.gr))
    }
    return `${this.base}/api/tiles/${sid}/${i}/${j}/${level}?${q}`
  }

  downloadModels = (ids: string[]): Promise<unknown> =>
    this.req('POST', '/api/models/download', { ids })
  cancelDownload = (id: string): Promise<unknown> =>
    this.req('POST', `/api/models/${encodeURIComponent(id)}/cancel`)

  thumbUrl(id: string): string {
    return `${this.base}/api/images/${id}/thumb?t=${encodeURIComponent(this.token)}`
  }

  viewUrl(id: string, r: Region, ow: number, oh: number): string {
    const q = new URLSearchParams({
      x: r.x.toFixed(2),
      y: r.y.toFixed(2),
      w: r.w.toFixed(2),
      h: r.h.toFixed(2),
      ow: String(ow),
      oh: String(oh),
      t: this.token
    })
    return `${this.base}/api/images/${id}/view?${q}`
  }

  /** Processed region as an object URL, or null when superseded by a newer request. */
  async preview(
    imageId: string,
    settings: Settings,
    region: Region,
    outW: number,
    outH: number,
    stage: Stage,
    signal?: AbortSignal,
    channel: 'view' | 'base' = 'view'
  ): Promise<{ blob: Blob; ms: number; kind: PreviewKind } | null> {
    const res = await fetch(this.base + '/api/preview', {
      method: 'POST',
      signal,
      headers: { 'X-Pixforge-Token': this.token, 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image_id: imageId,
        settings: wireSettings(settings),
        region,
        out_w: outW,
        out_h: outH,
        stage,
        channel
      })
    })
    if (res.status === 204) return null
    if (!res.ok) {
      let msg = res.statusText
      try {
        msg = (await res.json()).detail ?? msg
      } catch {
        /* ignore */
      }
      throw new ApiError(res.status, msg)
    }
    return {
      blob: await res.blob(),
      ms: Number(res.headers.get('X-Preview-Ms') ?? 0),
      kind: (res.headers.get('X-Preview-Kind') as PreviewKind | null) ?? 'exact'
    }
  }

  /** WebSocket with automatic reconnect. Returns a function that closes it. */
  events(onEvent: (ev: Record<string, unknown>) => void): () => void {
    let ws: WebSocket | null = null
    let closed = false
    let retry = 500
    const connect = (): void => {
      ws = new WebSocket(
        `ws://127.0.0.1:${this.base.split(':').pop()}/api/events?t=${encodeURIComponent(this.token)}`
      )
      ws.onmessage = (m) => {
        retry = 500
        onEvent(JSON.parse(m.data as string))
      }
      ws.onclose = () => {
        if (!closed) setTimeout(connect, (retry = Math.min(retry * 2, 5000)))
      }
    }
    connect()
    return () => {
      closed = true
      ws?.close()
    }
  }
}
