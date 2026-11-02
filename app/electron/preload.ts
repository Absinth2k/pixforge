import { contextBridge, ipcRenderer, webUtils } from 'electron'
import type { PixforgeBridge } from './api'

function subscribe<T>(channel: string, cb: (value: T) => void): () => void {
  const handler = (_e: unknown, value: T): void => cb(value)
  ipcRenderer.on(channel, handler)
  return () => ipcRenderer.removeListener(channel, handler)
}

const bridge: PixforgeBridge = {
  platform: process.platform,
  versions: {
    app: process.env['npm_package_version'] ?? '',
    electron: process.versions.electron,
    chrome: process.versions.chrome
  },
  getBackend: () => ipcRenderer.invoke('backend:info'),
  restartBackend: () => ipcRenderer.invoke('backend:restart'),
  onBackendExited: (cb) => subscribe('backend:exited', cb),
  window: {
    minimize: () => ipcRenderer.send('win:minimize'),
    toggleMaximize: () => ipcRenderer.send('win:toggle-maximize'),
    close: () => ipcRenderer.send('win:close'),
    isMaximized: () => ipcRenderer.invoke('win:is-maximized'),
    onMaximizedChange: (cb) => subscribe('win:maximized', cb)
  },
  openImages: () => ipcRenderer.invoke('dialog:open-images'),
  openFolder: () => ipcRenderer.invoke('dialog:open-folder'),
  expandPaths: (paths) => ipcRenderer.invoke('fs:expand', paths),
  pathForFile: (file) => webUtils.getPathForFile(file),
  showInFolder: (path) => ipcRenderer.send('shell:show-item', path),
  openLogs: () => ipcRenderer.send('shell:open-logs'),
  onOpenFiles: (cb) => subscribe('app:open-files', cb),
  takePendingFiles: () => ipcRenderer.invoke('app:take-pending-files'),
  openExternal: (url) => ipcRenderer.send('shell:open-external', url),
  update: {
    get: () => ipcRenderer.invoke('update:get'),
    setEnabled: (on) => ipcRenderer.invoke('update:set-enabled', on),
    check: () => ipcRenderer.invoke('update:check'),
    install: () => ipcRenderer.invoke('update:install'),
    onState: (cb) => subscribe('update:state', cb)
  }
}

contextBridge.exposeInMainWorld('pixforge', bridge)
