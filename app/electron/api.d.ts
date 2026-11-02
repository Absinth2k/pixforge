/** API exposed by the preload script as `window.pixforge` (contextIsolation, sandboxed). */

export interface BackendInfo {
  ok: boolean
  port?: number
  token?: string
  error?: string
  log?: string
}

export interface UpdateState {
  phase: 'idle' | 'dev' | 'disabled' | 'checking' | 'none' | 'downloading' | 'ready' | 'error'
  enabled: boolean
  /** version of the running app */
  current: string
  version?: string
  percent?: number
  error?: string
}

export interface PixforgeBridge {
  platform: string
  versions: { app: string; electron: string; chrome: string }
  getBackend(): Promise<BackendInfo>
  restartBackend(): Promise<BackendInfo>
  onBackendExited(cb: (info: { code: number | null; log: string }) => void): () => void
  window: {
    minimize(): void
    toggleMaximize(): void
    close(): void
    isMaximized(): Promise<boolean>
    onMaximizedChange(cb: (maximized: boolean) => void): () => void
  }
  openImages(): Promise<string[]>
  openFolder(): Promise<string[]>
  expandPaths(paths: string[]): Promise<string[]>
  pathForFile(file: File): string
  showInFolder(path: string): void
  openLogs(): void
  onOpenFiles(cb: (paths: string[]) => void): () => void
  /** Files passed on the command line ("Open with"), handed out once. */
  takePendingFiles(): Promise<string[]>
  /** Opens an https:// link in the default browser. */
  openExternal(url: string): void
  update: {
    get(): Promise<UpdateState>
    setEnabled(on: boolean): Promise<UpdateState>
    check(): Promise<UpdateState>
    install(): Promise<void>
    onState(cb: (s: UpdateState) => void): () => void
  }
}

declare global {
  interface Window {
    pixforge: PixforgeBridge
  }
}
