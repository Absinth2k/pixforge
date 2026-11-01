import { useCallback, useEffect, useState } from 'react'
import { BackendError } from '@/components/BackendError'
import { BatchBar } from '@/components/BatchBar'
import { EmptyState } from '@/components/EmptyState'
import { SettingsPanel } from '@/components/SettingsPanel'
import { StatusBar } from '@/components/StatusBar'
import { TitleBar } from '@/components/TitleBar'
import { Toolbar, type ZoomPreset } from '@/components/Toolbar'
import { Viewer } from '@/components/Viewer'
import { useStore } from '@/state/store'
import { applyTheme } from '@/theme/theme'

export default function App() {
  const connect = useStore((s) => s.connect)
  const addPaths = useStore((s) => s.addPaths)
  const images = useStore((s) => s.images)
  const selectedId = useStore((s) => s.selectedId)
  const theme = useStore((s) => s.theme)
  const set = useStore((s) => s.set)
  const [zoomPreset, setZoomPreset] = useState<ZoomPreset | null>('fit')
  const [dragging, setDragging] = useState(false)
  const img = images.find((i) => i.id === selectedId) ?? null

  useEffect(() => applyTheme(theme), [theme])

  useEffect(() => {
    void connect()
    const off1 = window.pixforge.onBackendExited(({ log }) =>
      set({
        api: null,
        backendError: 'Die Bildverarbeitung wurde unerwartet beendet.',
        backendLog: log
      })
    )
    const off2 = window.pixforge.onOpenFiles((paths) => void addPaths(paths))
    return () => {
      off1()
      off2()
    }
  }, [connect, addPaths, set])

  const open = useCallback(async () => addPaths(await window.pixforge.openImages()), [addPaths])
  const openFolder = useCallback(
    async () => addPaths(await window.pixforge.openFolder()),
    [addPaths]
  )

  const zoomTo = useCallback((z: ZoomPreset) => {
    const st = useStore.getState()
    const cur = st.selectedId
    const image = st.images.find((i) => i.id === cur)
    if (!cur || !image) return
    const v = st.view[cur]
    st.setView(cur, {
      zoom: z === 'fit' ? null : z,
      cx: z === 'fit' ? image.width / 2 : (v?.cx ?? image.width / 2),
      cy: z === 'fit' ? image.height / 2 : (v?.cy ?? image.height / 2)
    })
  }, [])

  // Keyboard: Ctrl+O open, Ctrl+E export, 0/1/2 zoom, F faces, arrows switch images, Del remove.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return
      const st = useStore.getState()
      if (e.ctrlKey && e.key.toLowerCase() === 'o') {
        e.preventDefault()
        void open()
      } else if (e.ctrlKey && e.key.toLowerCase() === 'e' && st.selectedId) {
        e.preventDefault()
        void st.exportImages([st.selectedId])
      } else if (!e.ctrlKey && !e.altKey) {
        if (e.key === '0') zoomTo('fit')
        else if (e.key === '1') zoomTo(1)
        else if (e.key === '2') zoomTo(2)
        else if (e.key.toLowerCase() === 'f') st.set({ showFaces: !st.showFaces })
        else if (e.key === 'Escape' && st.cropEditing) st.set({ cropEditing: false })
        else if (e.key.toLowerCase() === 'c' && st.selectedId) {
          const image = st.images.find((i) => i.id === st.selectedId)
          if (image && !st.settings[image.id]?.crop) {
            const w = Math.round(image.width * 0.7)
            const h = Math.round(image.height * 0.7)
            st.updateSettings(image.id, (s) => {
              s.crop = [Math.round((image.width - w) / 2), Math.round((image.height - h) / 2), w, h]
              return s
            })
          }
          st.set({ cropEditing: !st.cropEditing })
        } else if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
          const idx = st.images.findIndex((i) => i.id === st.selectedId)
          const next = st.images[idx + (e.key === 'ArrowRight' ? 1 : -1)]
          if (next) st.select(next.id)
        } else if (e.key === 'Delete' && st.selectedId) void st.removeImage(st.selectedId)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, zoomTo])

  // Drag and drop of files and folders anywhere in the window.
  const onDrop = async (e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const paths = [...e.dataTransfer.files]
      .map((f) => window.pixforge.pathForFile(f))
      .filter(Boolean)
    await addPaths(await window.pixforge.expandPaths(paths))
  }

  return (
    <div
      className="relative flex h-full flex-col"
      onDragOver={(e) => {
        e.preventDefault()
        if (!dragging) setDragging(true)
      }}
      onDragLeave={(e) => {
        if (e.currentTarget === e.target || !e.currentTarget.contains(e.relatedTarget as Node))
          setDragging(false)
      }}
      onDrop={(e) => void onDrop(e)}
    >
      <TitleBar />
      <Toolbar
        zoomPreset={zoomPreset}
        onZoom={zoomTo}
        onOpen={() => void open()}
        onOpenFolder={() => void openFolder()}
      />
      {img ? (
        <div className="flex min-h-0 flex-1 flex-wrap">
          <Viewer img={img} onZoomPreset={setZoomPreset} />
          <SettingsPanel img={img} />
        </div>
      ) : (
        <EmptyState
          onOpen={() => void open()}
          onOpenFolder={() => void openFolder()}
          dragging={dragging}
        />
      )}
      {images.length > 0 && <BatchBar />}
      <StatusBar />
      {dragging && img && (
        <div className="pointer-events-none absolute inset-0 z-40 border-2 border-dashed border-accent bg-accent-soft" />
      )}
      <BackendError />
    </div>
  )
}
