/**
 * Smoke test of the real app: Electron + Python backend + models.
 *
 *   npm run test:e2e
 *
 * PIXFORGE_SHOT_IMAGE=<path>  additionally loads that image and writes screenshots to
 *                             ../docs/images (used for the README).
 */
import { execFileSync } from 'node:child_process'
import { mkdirSync, mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import {
  _electron as electron,
  expect,
  test,
  type ElectronApplication,
  type Page
} from '@playwright/test'

const APP_DIR = resolve(__dirname, '..')
const PYTHON = resolve(APP_DIR, '..', 'backend', '.venv', 'Scripts', 'python.exe')

function syntheticImage(): string {
  const dir = mkdtempSync(join(tmpdir(), 'pixforge-e2e-'))
  const path = join(dir, 'testbild.png')
  // gradients + checker + soft noise, 480x320, generated with the backend's Pillow/numpy
  execFileSync(PYTHON, [
    '-c',
    [
      'import numpy as np, sys',
      'from PIL import Image',
      'yy, xx = np.mgrid[0:320, 0:480].astype(np.float32)',
      'r = 0.5 + 0.4*np.sin(xx/17)*np.cos(yy/23)',
      'g = (xx + yy) / 800',
      'b = ((xx//24 + yy//24) % 2) * 0.5 + 0.25',
      'img = np.stack([r, g, b], -1) + np.random.default_rng(0).normal(0, 0.02, (320, 480, 3))',
      'Image.fromarray((np.clip(img, 0, 1)*255).astype(np.uint8)).save(sys.argv[1])'
    ].join('\n'),
    path
  ])
  return path
}

async function launch(): Promise<{ app: ElectronApplication; page: Page }> {
  const app = await electron.launch({
    args: ['.'],
    cwd: APP_DIR,
    env: {
      ...process.env,
      LANG: 'de_DE',
      PIXFORGE_E2E_PROFILE: mkdtempSync(join(tmpdir(), 'pixforge-profile-'))
    }
  })
  const page = await app.firstWindow()
  await page.setViewportSize({ width: 1440, height: 900 })
  return { app, page }
}

async function openFiles(app: ElectronApplication, paths: string[]): Promise<void> {
  await app.evaluate(({ BrowserWindow }, p) => {
    BrowserWindow.getAllWindows()[0].webContents.send('app:open-files', p)
  }, paths)
}

test('start screen, load an image, preview renders', async () => {
  const { app, page } = await launch()
  try {
    await expect(
      page.getByRole('heading', { name: /Bilder hierher ziehen|Drop images here/ })
    ).toBeVisible()
    await expect(page.getByText(/DirectML|Prozessor|processor/)).toBeVisible()
    await openFiles(app, [syntheticImage()])
    await expect(page.getByRole('option', { name: /testbild\.png/ })).toBeVisible()
    const result = page.locator('img[alt^="Pixforge"]')
    await expect(result).toBeVisible()
    await expect(page.getByText(/Vorschau wird berechnet|Computing preview/)).toBeHidden()
    // the real result arrives as background tiles and replaces the quick preview
    await expect(page.getByTestId('tile-canvas')).toBeAttached({ timeout: 30_000 })
    await expect(page.getByTestId('preview-status')).toBeHidden({ timeout: 60_000 })
    // finishing on/off: no new model run, the tiles stay complete
    const fine = page.getByTestId('fine-toggle')
    await expect(fine).toHaveAttribute('aria-pressed', 'true')
    await fine.click()
    await expect(fine).toHaveAttribute('aria-pressed', 'false')
    await expect(page.getByTestId('preview-status')).toBeHidden({ timeout: 5_000 })
    await fine.click()
    // the estimate shows the 4x output size
    await expect(page.getByText('1.920 × 1.280')).toBeVisible()
    // denoise-only switches the task and the output size
    await page.getByRole('button', { name: /Nur entrauschen|Denoise only/ }).click()
    await expect(page.getByText('480 × 320').last()).toBeVisible()
    // export: visible feedback right after the click (the tiny test image may already be done)
    const exportBtn = page.getByRole('button', { name: /Exportieren|Export|Wartet|Exportiere/ })
    await exportBtn.click()
    await expect(
      page.getByText(/Bild wird verarbeitet|Image is being processed|Gespeichert|Saved/).first()
    ).toBeVisible()
  } finally {
    await app.close()
  }
})

test('smooth wheel zoom and crop tool', async () => {
  const { app, page } = await launch()
  try {
    await openFiles(app, [syntheticImage()])
    await expect(page.locator('img[alt^="Pixforge"]')).toBeVisible()
    const viewer = page.locator('img[alt="Original"]').first()
    const box = (await viewer.boundingBox())!
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2)
    for (let i = 0; i < 4; i++) await page.mouse.wheel(0, -120)
    await page.waitForTimeout(300)
    await expect(page.getByTestId('zoom-readout')).toHaveText(/^\d+ %$/) // left "fit" mode
    for (let i = 0; i < 12; i++) await page.mouse.wheel(0, 240)
    await expect(page.getByTestId('zoom-readout')).toHaveText(/^Einpassen · \d+ %$/) // snapped back

    await page.keyboard.press('c')
    await expect(page.getByRole('button', { name: 'Fertig' })).toBeVisible()
    await expect(page.getByText('Ausschnitt 336 × 224 px → 1.344 × 896 px').first()).toBeVisible()
    await page.getByRole('button', { name: 'Fertig' }).click()
    await expect(page.getByText(/Nur der Ausschnitt 336 × 224 px wird gerechnet/)).toBeVisible()
    await expect(page.getByText('1.344 × 896')).toBeVisible() // estimate uses the crop
    await page.getByRole('button', { name: 'Aufheben' }).click()
    await expect(page.getByText('1.920 × 1.280')).toBeVisible()
  } finally {
    await app.close()
  }
})

test('screenshots for the README', async () => {
  const shot = process.env.PIXFORGE_SHOT_IMAGE
  test.skip(!shot, 'set PIXFORGE_SHOT_IMAGE to an image path')
  const outDir = resolve(APP_DIR, '..', 'docs', 'images')
  mkdirSync(outDir, { recursive: true })
  const { app, page } = await launch()
  try {
    await expect(page.getByRole('heading', { name: /Bilder hierher ziehen/ })).toBeVisible()
    await page.waitForTimeout(800)
    await page.screenshot({ path: join(outDir, 'start.png') })
    await openFiles(app, [shot!])
    await expect(page.locator('img[alt^="Pixforge"]')).toBeVisible()
    await expect(page.getByText(/Vorschau wird berechnet/)).toBeHidden({ timeout: 90_000 })
    await page.waitForTimeout(600)
    await page.screenshot({ path: join(outDir, 'main.png') })
    await page.getByRole('button', { name: 'Hell/Dunkel umschalten' }).click()
    await page.waitForTimeout(300)
    await page.screenshot({ path: join(outDir, 'main-light.png') })
    await page.getByRole('button', { name: 'Hell/Dunkel umschalten' }).click()
    await page.keyboard.press('1')
    // the preview request is debounced: wait until it started, then until it finished
    await expect(page.getByText(/Vorschau wird berechnet/)).toBeVisible({ timeout: 5_000 })
    await expect(page.getByText(/Vorschau wird berechnet/)).toBeHidden({ timeout: 90_000 })
    await page.waitForTimeout(400)
    await page.screenshot({ path: join(outDir, 'main-100.png') })
    await page.keyboard.press('0')
    await page.keyboard.press('c')
    await page.getByRole('button', { name: '3:2' }).click()
    await expect(page.getByText(/Vorschau wird berechnet/)).toBeHidden({ timeout: 90_000 })
    await page.waitForTimeout(700)
    await page.screenshot({ path: join(outDir, 'crop.png') })
  } finally {
    await app.close()
  }
})
