import { useCallback } from 'react';
import type uPlot from 'uplot';
import { UPlotChart, themeColors } from './UPlotChart.tsx';

export interface Series {
  label: string;
  y: number[];
  color?: string;
  dash?: number[];
}

/** A small multi-series line chart (capture-rate / ROC) on a shared numeric x-axis, theme-aware, crosshair readout.
 * Optionally draws the 1:1 diagonal (the random baseline). */
export function CurveChart({ x, series, xLabel, yLabel, diagonal = false, height = 240 }: {
  x: number[]; series: Series[]; xLabel: string; yLabel: string; diagonal?: boolean; height?: number;
}) {
  const data = [x, ...series.map((s) => s.y), ...(diagonal ? [x] : [])] as unknown as uPlot.AlignedData;
  const build = useCallback((width: number, h: number): uPlot.Options => {
    const c = themeColors();
    const palette = [c.accent, c.good, c.warn, c.bad];
    return {
      width,
      height: h,
      scales: { x: { time: false, range: [0, 1] }, y: { range: [0, 1] } },
      axes: [
        { label: xLabel, stroke: c.subtle, grid: { stroke: c.border }, ticks: { stroke: c.border } },
        { label: yLabel, stroke: c.subtle, grid: { stroke: c.border }, ticks: { stroke: c.border } },
      ],
      series: [
        { label: xLabel },
        ...series.map((s, i) => ({
          label: s.label,
          stroke: s.color ?? palette[i % palette.length],
          width: 2,
          dash: s.dash,
        })),
        ...(diagonal ? [{ label: 'random', stroke: c.faint, width: 1, dash: [4, 4] }] : []),
      ],
    };
  }, [series, xLabel, yLabel, diagonal]);
  return <UPlotChart data={data} build={build} height={height} />;
}
