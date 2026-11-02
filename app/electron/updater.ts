/**
 * Automatic updates from GitHub Releases (electron-updater), switchable in the info dialog.
 *
 * Only packaged builds look for updates: shortly after the start and then every 6 hours. An
 * update is downloaded in the background and installed when Pixforge is closed (or right
 * away via "Jetzt neu starten"). The switch is stored in userData/settings.json.
 */
import { app, BrowserWindow, ipcMain } from 'electron'
import { autoUpdater } from 'electron-updater'
import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'

export type UpdatePhase =
  'idle' | 'dev' | 'disabled' | 'checking' | 'none' | 'downloading' | 'ready' | 'error'

export interface UpdateState {
  phase: UpdatePhase
  enabled: boolean
  current: string
  version?: string
  percent?: number
  error?: string
}

const CHECK_EVERY_MS = 6 * 60 * 60 * 1000
const FIRST_CHECK_MS = 10_000

let state: UpdateState = { phase: 'idle', enabled: true, current: '' }
let timer: NodeJS.Timeout | null = null

function settingsFile(): string {
  return join(app.getPath('userData'), 'settings.json')
}

function readSettings(): Record<string, unknown> {
  try {
    if (existsSync(settingsFile())) return JSON.parse(readFileSync(settingsFile(), 'utf8'))
  } catch {
    /* broken file: defaults */
  }
  return {}
}

function writeSettings(patch: Record<string, unknown>): void {
  try {
    writeFileSync(settingsFile(), JSON.stringify({ ...readSettings(), ...patch }, null, 2))
  } catch {
    /* read-only profile: the switch only lasts for this session */
  }
}

function publish(patch: Partial<UpdateState>): void {
  state = { ...state, ...patch }
  for (const w of BrowserWindow.getAllWindows()) w.webContents.send('update:state', state)
}

async function check(): Promise<void> {
  if (!app.isPackaged || !state.enabled) return
  if (state.phase === 'downloading' || state.phase === 'ready') return
  try {
    await autoUpdater.checkForUpdates()
  } catch (e) {
    publish({ phase: 'error', error: e instanceof Error ? e.message : String(e) })
  }
}

function schedule(): void {
  if (timer) clearInterval(timer)
  timer = null
  if (!app.isPackaged || !state.enabled) return
  setTimeout(() => void check(), FIRST_CHECK_MS)
  timer = setInterval(() => void check(), CHECK_EVERY_MS)
}

export function initUpdater(): void {
  const enabled = readSettings()['autoUpdate'] !== false
  state = {
    phase: !app.isPackaged ? 'dev' : enabled ? 'idle' : 'disabled',
    enabled,
    current: app.getVersion()
  }

  ipcMain.handle('update:get', () => state)
  ipcMain.handle('update:set-enabled', (_e, on: boolean) => {
    writeSettings({ autoUpdate: on })
    publish({
      enabled: on,
      phase: !app.isPackaged ? 'dev' : on ? 'idle' : 'disabled'
    })
    schedule()
    return state
  })
  ipcMain.handle('update:check', async () => {
    await check()
    return state
  })
  ipcMain.handle('update:install', () => {
    if (state.phase === 'ready') autoUpdater.quitAndInstall()
  })

  if (!app.isPackaged) return
  autoUpdater.autoDownload = true
  autoUpdater.autoInstallOnAppQuit = true
  autoUpdater.on('checking-for-update', () => publish({ phase: 'checking', error: undefined }))
  autoUpdater.on('update-not-available', () => publish({ phase: 'none' }))
  autoUpdater.on('update-available', (info) =>
    publish({ phase: 'downloading', version: info.version, percent: 0 })
  )
  autoUpdater.on('download-progress', (p) =>
    publish({ phase: 'downloading', percent: Math.round(p.percent) })
  )
  autoUpdater.on('update-downloaded', (info) => publish({ phase: 'ready', version: info.version }))
  autoUpdater.on('error', (e) => publish({ phase: 'error', error: e.message }))
  schedule()
}
