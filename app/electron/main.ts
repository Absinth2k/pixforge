import { readdir, stat } from 'node:fs/promises'
import { extname, join } from 'node:path'
import { app, BrowserWindow, dialog, ipcMain, session, shell } from 'electron'
import { restartBackend, setExitHandler, startBackend, stopBackend } from './backend-process'
import { initUpdater } from './updater'

const IMAGE_EXT = new Set([
  '.jpg',
  '.jpeg',
  '.png',
  '.tif',
  '.tiff',
  '.webp',
  '.bmp',
  '.heic',
  '.heif',
  '.avif',
  '.hif',
  '.cr2',
  '.cr3',
  '.nef',
  '.nrw',
  '.arw',
  '.dng',
  '.raf',
  '.orf',
  '.rw2',
  '.pef',
  '.srw'
])
const DEV_URL = process.env['ELECTRON_RENDERER_URL']

let win: BrowserWindow | null = null
let pendingFiles: string[] = []

function filesFromArgv(argv: string[]): string[] {
  return argv.slice(1).filter((a) => !a.startsWith('-') && IMAGE_EXT.has(extname(a).toLowerCase()))
}

async function expandPaths(paths: string[]): Promise<string[]> {
  const out: string[] = []
  for (const p of paths) {
    try {
      const st = await stat(p)
      if (st.isDirectory()) {
        const names = (await readdir(p)).sort((a, b) => a.localeCompare(b, 'de'))
        for (const n of names) if (IMAGE_EXT.has(extname(n).toLowerCase())) out.push(join(p, n))
      } else if (IMAGE_EXT.has(extname(p).toLowerCase())) {
        out.push(p)
      }
    } catch {
      /* unreadable path: ignore */
    }
  }
  return out
}

function createWindow(): void {
  win = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 1024,
    minHeight: 680,
    frame: false,
    backgroundColor: '#141416',
    show: false,
    title: 'Pixforge',
    icon: join(__dirname, '../../build/icon.ico'),
    webPreferences: {
      preload: join(__dirname, '../preload/preload.js'),
      sandbox: true,
      contextIsolation: true,
      nodeIntegration: false,
      spellcheck: false
    }
  })
  win.once('ready-to-show', () => win?.show())
  win.on('maximize', () => win?.webContents.send('win:maximized', true))
  win.on('unmaximize', () => win?.webContents.send('win:maximized', false))
  // Never navigate away from the app; external links open in the browser.
  win.webContents.on('will-navigate', (e, url) => {
    if (!DEV_URL || !url.startsWith(DEV_URL)) e.preventDefault()
  })
  win.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith('https://')) void shell.openExternal(url)
    return { action: 'deny' }
  })
  if (DEV_URL) void win.loadURL(DEV_URL)
  else void win.loadFile(join(__dirname, '../renderer/index.html'))
}

function installCsp(): void {
  if (DEV_URL) return // Vite's dev server needs inline scripts for hot reload
  session.defaultSession.webRequest.onHeadersReceived((details, callback) => {
    callback({
      responseHeaders: {
        ...details.responseHeaders,
        'Content-Security-Policy': [
          "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; " +
            "img-src 'self' blob: data: http://127.0.0.1:*; font-src 'self' data:; " +
            "connect-src 'self' http://127.0.0.1:* ws://127.0.0.1:*"
        ]
      }
    })
  })
}

function registerIpc(): void {
  ipcMain.handle('backend:info', () => startBackend())
  ipcMain.handle('backend:restart', () => restartBackend())
  ipcMain.on('win:minimize', () => win?.minimize())
  ipcMain.on('win:toggle-maximize', () => (win?.isMaximized() ? win.unmaximize() : win?.maximize()))
  ipcMain.on('win:close', () => win?.close())
  ipcMain.handle('win:is-maximized', () => win?.isMaximized() ?? false)
  ipcMain.handle('dialog:open-images', async () => {
    if (!win) return []
    const res = await dialog.showOpenDialog(win, {
      title: 'Bilder öffnen',
      properties: ['openFile', 'multiSelections'],
      filters: [
        { name: 'Bilder', extensions: [...IMAGE_EXT].map((e) => e.slice(1)) },
        { name: 'Alle Dateien', extensions: ['*'] }
      ]
    })
    return res.canceled ? [] : res.filePaths
  })
  ipcMain.handle('dialog:open-folder', async () => {
    if (!win) return []
    const res = await dialog.showOpenDialog(win, {
      title: 'Ordner öffnen',
      properties: ['openDirectory']
    })
    return res.canceled ? [] : expandPaths(res.filePaths)
  })
  ipcMain.handle('fs:expand', (_e, paths: string[]) => expandPaths(paths))
  ipcMain.handle('app:take-pending-files', () => {
    const files = pendingFiles
    pendingFiles = []
    return files
  })
  ipcMain.on('shell:show-item', (_e, path: string) => shell.showItemInFolder(path))
  ipcMain.on('shell:open-external', (_e, url: string) => {
    if (typeof url === 'string' && url.startsWith('https://')) void shell.openExternal(url)
  })
  ipcMain.on('shell:open-logs', () => {
    void shell.openPath(join(app.getPath('home'), 'AppData', 'Local', 'Pixforge', 'logs'))
  })
}

// Tests run with their own profile, so they never hand their files to a running Pixforge
// (the single-instance lock is tied to the userData folder).
if (process.env['PIXFORGE_E2E_PROFILE'])
  app.setPath('userData', process.env['PIXFORGE_E2E_PROFILE'])

if (!app.requestSingleInstanceLock()) {
  app.quit()
} else {
  app.on('second-instance', (_e, argv) => {
    const files = filesFromArgv(argv)
    if (win) {
      if (win.isMinimized()) win.restore()
      win.focus()
      if (files.length) win.webContents.send('app:open-files', files)
    }
  })
  app.whenReady().then(() => {
    app.setAppUserModelId('de.andrestrahl.pixforge')
    pendingFiles = filesFromArgv(process.argv)
    installCsp()
    registerIpc()
    setExitHandler((code, log) => win?.webContents.send('backend:exited', { code, log }))
    void startBackend()
    createWindow()
    initUpdater()
  })
  app.on('window-all-closed', () => app.quit())
  app.on('before-quit', () => stopBackend())
}
