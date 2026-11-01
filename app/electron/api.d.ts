/** API exposed by the preload script as `window.pixforge` (contextIsolation, sandboxed). */

export interface BackendInfo {
  ok: boolean
  port?: number
  token?: string
  error?: string
  log?: string
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
}

declare global {
  interface Window {
    pixforge: PixforgeBridge
  }
}
