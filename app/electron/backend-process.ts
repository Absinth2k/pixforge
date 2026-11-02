/**
 * Starts and stops the Python backend (FastAPI on 127.0.0.1, random port, random token).
 *
 * Development: backend/.venv/Scripts/python.exe -m pixforge.server
 * Packaged:    resources/backend/pixforge-server.exe (PyInstaller folder build, stage 6)
 */
import { spawn, spawnSync, type ChildProcess } from 'node:child_process'
import { randomBytes } from 'node:crypto'
import { existsSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { app } from 'electron'
import type { BackendInfo } from './api'

// generous: on a fresh PC the antivirus scans the new backend files on the very first start
const READY_TIMEOUT_MS = 180_000

let child: ChildProcess | null = null
let current: Promise<BackendInfo> | null = null
let logTail = ''
let stopping = false
let onExit: ((code: number | null, log: string) => void) | null = null

function remember(chunk: string): void {
  logTail = (logTail + chunk).slice(-8000)
}

function command(): { exe: string; args: string[]; cwd: string } {
  if (app.isPackaged) {
    const dir = join(process.resourcesPath, 'backend')
    return { exe: join(dir, 'pixforge-server.exe'), args: [], cwd: dir }
  }
  const backendDir = resolve(app.getAppPath(), '..', 'backend')
  const python = join(backendDir, '.venv', 'Scripts', 'python.exe')
  return { exe: python, args: ['-m', 'pixforge.server'], cwd: backendDir }
}

export function setExitHandler(handler: (code: number | null, log: string) => void): void {
  onExit = handler
}

export function startBackend(): Promise<BackendInfo> {
  if (current) return current
  stopping = false
  logTail = ''
  const token = randomBytes(24).toString('base64url')
  const { exe, args, cwd } = command()
  current = new Promise<BackendInfo>((resolvePromise) => {
    if (!existsSync(exe)) {
      resolvePromise({
        ok: false,
        error: app.isPackaged
          ? 'Die Bildverarbeitung von Pixforge fehlt. Bitte Pixforge neu installieren.'
          : `Python-Umgebung nicht gefunden: ${exe}. Im Ordner backend "uv sync" ausführen.`
      })
      return
    }
    const proc = spawn(exe, args, {
      cwd,
      env: { ...process.env, PIXFORGE_TOKEN: token, PYTHONUNBUFFERED: '1', PYTHONUTF8: '1' },
      stdio: ['pipe', 'pipe', 'pipe'],
      windowsHide: true
    })
    child = proc
    let settled = false
    const timer = setTimeout(() => {
      if (!settled) {
        settled = true
        resolvePromise({
          ok: false,
          error: 'Die Bildverarbeitung startet nicht (Zeitüberschreitung).',
          log: logTail
        })
      }
    }, READY_TIMEOUT_MS)
    let buf = ''
    proc.stdout?.setEncoding('utf8')
    proc.stdout?.on('data', (chunk: string) => {
      remember(chunk)
      buf += chunk
      const m = /PIXFORGE_READY (\d+)/.exec(buf)
      if (m && !settled) {
        settled = true
        clearTimeout(timer)
        resolvePromise({ ok: true, port: Number(m[1]), token })
      }
    })
    proc.stderr?.setEncoding('utf8')
    proc.stderr?.on('data', remember)
    proc.on('error', (err) => {
      if (!settled) {
        settled = true
        clearTimeout(timer)
        resolvePromise({ ok: false, error: `Start fehlgeschlagen: ${err.message}`, log: logTail })
      }
    })
    proc.on('exit', (code) => {
      child = null
      current = null
      if (!settled) {
        settled = true
        clearTimeout(timer)
        resolvePromise({
          ok: false,
          error: 'Die Bildverarbeitung wurde beim Start beendet.',
          log: logTail
        })
      } else if (!stopping) {
        onExit?.(code, logTail)
      }
    })
  })
  return current
}

export function stopBackend(): void {
  stopping = true
  const proc = child
  if (!proc) return
  // Closing stdin makes the backend exit by itself; kill the tree if it does not.
  proc.stdin?.end()
  const pid = proc.pid
  setTimeout(() => {
    if (child === proc && pid) {
      spawnSync('taskkill', ['/pid', String(pid), '/T', '/F'], { windowsHide: true })
    }
  }, 3000).unref()
}

export async function restartBackend(): Promise<BackendInfo> {
  stopBackend()
  const old = child
  if (old?.pid) spawnSync('taskkill', ['/pid', String(old.pid), '/T', '/F'], { windowsHide: true })
  child = null
  current = null
  return startBackend()
}
