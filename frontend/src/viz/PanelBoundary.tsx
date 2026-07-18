import { Component, type ReactNode } from 'react';

// Per-panel error boundary: a crash inside one prospectivity view (e.g. a tool fed an edge-case layer stack)
// renders a small inline message INSTEAD of unmounting the whole App to a blank page. The tab bar stays usable so
// the user can switch away. Mirrors the RotorVitals PanelBoundary (the reference app); keyed by source + case so
// it resets when the datum changes.
export class PanelBoundary extends Component<{ children: ReactNode; lang?: 'en' | 'es' }, { error: Error | null }> {
  state: { error: Error | null } = { error: null };
  static getDerivedStateFromError(error: Error) { return { error }; }
  render() {
    if (this.state.error) {
      const es = this.props.lang === 'es';
      return (
        <div className="pf-plot" style={{ padding: '1rem', color: 'var(--color-fg-faint)' }}>
          <strong>{es ? 'Esta vista no aplica a este caso' : 'This view does not apply to this case'}</strong>
          <p className="hint" style={{ marginTop: '0.4rem' }}>
            {es
              ? 'No se pudo computar esta vista sobre el dato actual (p. ej. un caso sin las capas o el formato que la herramienta requiere). Seleccionar otra pestaña o caso.'
              : 'This view could not be computed on the current datum (e.g. a case lacking the layers or format the tool needs). Pick another tab or case.'}
          </p>
          <p className="hint" style={{ opacity: 0.6, fontFamily: 'ui-monospace, monospace', fontSize: '0.75rem' }}>{this.state.error.message}</p>
        </div>
      );
    }
    return this.props.children;
  }
}
