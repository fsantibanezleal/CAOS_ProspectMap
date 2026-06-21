import { useEffect, useRef, useState } from 'react';

// a perceptually-uniform sequential colormap (viridis approximation via piecewise-linear control points). Honest:
// monotone luminance, no rainbow that fabricates structure.
const VIRIDIS: [number, number, number][] = [
  [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142], [38, 130, 142],
  [31, 158, 137], [53, 183, 121], [110, 206, 88], [181, 222, 43], [253, 231, 37],
];
function colormap(t: number): [number, number, number] {
  const x = Math.max(0, Math.min(1, t)) * (VIRIDIS.length - 1);
  const i = Math.floor(x);
  const f = x - i;
  const a = VIRIDIS[i];
  const b = VIRIDIS[Math.min(i + 1, VIRIDIS.length - 1)];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

/** The prospectivity raster: a scalar field over the nx*ny grid painted with a perceptually-uniform colormap, known
 * deposits overlaid as markers, OOD ("do-not-trust") cells dimmed/cross-hatched. Hover reads the cell value out. Pure
 * canvas + ImageData (the FragmentIQ SceneView pattern) — no GPU/map dependency for a teaching-scale grid. */
export function MapView({ nx, ny, field, range, deposits, ood, oodThreshold, height = 360, lang = 'en', valueLabel = 'P' }: {
  nx: number; ny: number; field: Float64Array; range: [number, number]; deposits: number[];
  ood?: Float64Array; oodThreshold?: number; height?: number; lang?: 'en' | 'es'; valueLabel?: string;
}) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hover, setHover] = useState<{ x: number; y: number; text: string } | null>(null);

  useEffect(() => {
    const wrap = wrapRef.current;
    const canvas = canvasRef.current;
    if (!wrap || !canvas) return;
    const dispW = wrap.clientWidth || 640;
    const scale = dispW / nx;
    const dispH = Math.round(ny * scale);
    const dpr = window.devicePixelRatio || 1;
    canvas.width = dispW * dpr;
    canvas.height = dispH * dpr;
    canvas.style.width = `${dispW}px`;
    canvas.style.height = `${dispH}px`;
    const ctx = canvas.getContext('2d')!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // paint the field into an offscreen ImageData at native resolution, then scale-draw (nearest-neighbour)
    const off = document.createElement('canvas');
    off.width = nx;
    off.height = ny;
    const octx = off.getContext('2d')!;
    const img = octx.createImageData(nx, ny);
    const [lo, hi] = range;
    const span = hi - lo || 1;
    for (let i = 0; i < nx * ny; i++) {
      const t = (field[i] - lo) / span;
      let [r, g, b] = colormap(t);
      // OOD: dim + cross-hatch the cells whose geology is outside the labelled training envelope
      if (ood && oodThreshold != null && ood[i] > oodThreshold) {
        const { x, y } = { x: i % nx, y: Math.floor(i / nx) };
        const hatch = (x + y) % 4 === 0;
        r = r * 0.35 + (hatch ? 60 : 30);
        g = g * 0.35 + (hatch ? 60 : 30);
        b = b * 0.35 + (hatch ? 60 : 30);
      }
      img.data[i * 4] = r;
      img.data[i * 4 + 1] = g;
      img.data[i * 4 + 2] = b;
      img.data[i * 4 + 3] = 255;
    }
    octx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, 0, 0, dispW, dispH);

    // deposits as markers (vector overlay)
    ctx.fillStyle = 'rgba(255,255,255,0.95)';
    ctx.strokeStyle = 'rgba(0,0,0,0.85)';
    ctx.lineWidth = 1;
    const rad = Math.max(2, scale * 0.7);
    for (const d of deposits) {
      const cx = (d % nx + 0.5) * scale;
      const cy = (Math.floor(d / nx) + 0.5) * scale;
      ctx.beginPath();
      ctx.arc(cx, cy, rad, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
  }, [nx, ny, field, range, deposits, ood, oodThreshold, height]);

  const onMove = (e: React.MouseEvent) => {
    const wrap = wrapRef.current;
    if (!wrap) return;
    const rect = wrap.getBoundingClientRect();
    const scale = rect.width / nx;
    const ix = Math.floor((e.clientX - rect.left) / scale);
    const iy = Math.floor((e.clientY - rect.top) / scale);
    if (ix < 0 || ix >= nx || iy < 0 || iy >= ny) { setHover(null); return; }
    const i = iy * nx + ix;
    const isOod = ood && oodThreshold != null && ood[i] > oodThreshold;
    setHover({
      x: e.clientX - rect.left, y: e.clientY - rect.top,
      text: `${valueLabel} ${field[i].toFixed(3)}${isOod ? (lang === 'es' ? ' · fuera de envolvente' : ' · out-of-envelope') : ''}`,
    });
  };

  return (
    <div className="fq-scene" ref={wrapRef} style={{ position: 'relative' }} onMouseMove={onMove} onMouseLeave={() => setHover(null)}>
      <canvas ref={canvasRef} style={{ display: 'block', borderRadius: 8, width: '100%' }} />
      {hover && <div className="heatmap-readout" style={{ left: Math.min(hover.x + 10, 9999), top: hover.y + 10 }}>{hover.text}</div>}
    </div>
  );
}
