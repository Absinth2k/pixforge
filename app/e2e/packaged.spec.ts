/**
 * Smoke test of the packaged app (what the installer ships): frozen backend, no Python.
 *
 *   PIXFORGE_PACKAGED=dist/win-unpacked/Pixforge.exe npx playwright test e2e/packaged.spec.ts
 *
 * Optional PIXFORGE_MODELS=<empty folder> also tests the first-start model download.
 */
import { mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { spawn } from 'node:child_process'
import { chromium, expect, test } from '@playwright/test'
import { deflateSync } from 'node:zlib'

const EXE = process.env['PIXFORGE_PACKAGED']

/** A small PNG without any Python (the packaged app must not need it). */
function testPng(): string {
  const w = 320
  const h = 240
  const raw = Buffer.alloc((w * 3 + 1) * h)
  for (let y = 0; y < h; y++) {
    raw[y * (w * 3 + 1)] = 0
    for (let x = 0; x < w; x++) {
      const o = y * (w * 3 + 1) + 1 + x * 3
      raw[o] = (x * 255) / w
      raw[o + 1] = (y * 255) / h
      raw[o + 2] = ((x >> 4) + (y >> 4)) % 2 ? 200 : 60
    }
  }
  const crcTable = Array.from({ length: 256 }, (_, n) => {
    let c = n
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1
    return c >>> 0
  })
  const crc = (b: Buffer) => {
    let c = 0xffffffff
    for (const v of b) c = crcTable[(c ^ v) & 0xff] ^ (c >>> 8)
    return (c ^ 0xffffffff) >>> 0
  }
  const chunk = (type: string, data: Buffer) => {
    const len = Buffer.alloc(4)
    len.writeUInt32BE(data.length)
    const td = Buffer.concat([Buffer.from(type), data])
    const c = Buffer.alloc(4)
    c.writeUInt32BE(crc(td))
    return Buffer.concat([len, td, c])
  }
  const ihdr = Buffer.alloc(13)
  ihdr.writeUInt32BE(w, 0)
  ihdr.writeUInt32BE(h, 4)
  ihdr[8] = 8
  ihdr[9] = 2
  const png = Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr),
    chunk('IDAT', deflateSync(raw)),
    chunk('IEND', Buffer.alloc(0))
  ])
  const path = join(mkdtempSync(join(tmpdir(), 'pixforge-pkg-')), 'muster.png')
  writeFileSync(path, png)
  return path
}

test.skip(!EXE, 'set PIXFORGE_PACKAGED to the packaged Pixforge.exe')

test('packaged app: setup, about, preview', async () => {
  test.setTimeout(15 * 60_000)
  // Playwright cannot inject into a packaged Electron: start it with a DevTools port instead
  const profile = mkdtempSync(join(tmpdir(), 'pixforge-profile-'))
  const proc = spawn(resolve(EXE!), ['--remote-debugging-port=0'], {
    env: { ...process.env, LANG: 'de_DE', PIXFORGE_E2E_PROFILE: profile }
  })
  const port = await new Promise<number>((ok, fail) => {
    const t = setTimeout(() => fail(new Error('no DevTools port')), 120_000)
    proc.stderr.on('data', (d: Buffer) => {
      const m = /DevTools listening on ws:\/\/127\.0\.0\.1:(\d+)/.exec(d.toString())
      if (m) {
        clearTimeout(t)
        ok(Number(m[1]))
      }
    })
  })
  const browser = await chromium.connectOverCDP(`http://127.0.0.1:${port}`)
  try {
    const ctx = browser.contexts()[0]
    const page = ctx.pages()[0] ?? (await ctx.waitForEvent('page'))
    await page.setViewportSize({ width: 1440, height: 900 })
    // first start without models: the setup screen downloads them, then disappears
    if (process.env['PIXFORGE_MODELS']) {
      await expect(page.getByTestId('model-setup')).toBeVisible({ timeout: 120_000 })
      await page.screenshot({ path: 'test-results/packaged-setup.png' })
      await expect(page.getByTestId('model-setup')).toBeHidden({ timeout: 10 * 60_000 })
    }
    await expect(page.getByText(/DirectML|Prozessor|processor/)).toBeVisible({ timeout: 120_000 })
    await page.getByTestId('about-button').click()
    await expect(page.getByTestId('about-version')).toHaveText(
      new RegExp(process.env['PIXFORGE_EXPECT_VERSION'] ?? '\\d+\\.\\d+\\.\\d+')
    )
    if (process.env['PIXFORGE_CHECK_UPDATE']) {
      // the installed version must find itself as the newest release on GitHub
      await page.getByRole('button', { name: /Jetzt prüfen|Check now/ }).click()
      await expect(page.getByText(/Pixforge ist aktuell|Pixforge is up to date/)).toBeVisible({
        timeout: 60_000
      })
    }
    await page.screenshot({ path: 'test-results/packaged-about.png' })
    await page.keyboard.press('Escape')
    // drop a file the way Explorer's "open with" does: as a second instance
    spawn(resolve(EXE!), [testPng()], {
      env: { ...process.env, PIXFORGE_E2E_PROFILE: profile }
    })
    await expect(page.locator('img[alt^="Pixforge"]')).toBeVisible({ timeout: 120_000 })
    await expect(page.getByTestId('tile-canvas')).toBeAttached({ timeout: 120_000 })
    await page.screenshot({ path: 'test-results/packaged-main.png' })
  } finally {
    await browser.close().catch(() => undefined)
    proc.kill()
  }
})
