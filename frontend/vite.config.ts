import react from '@vitejs/plugin-react';
import { defineConfig } from 'vitest/config';

export default defineConfig({
  base: '/', // served at the custom domain root (prospectmap.fasl-work.com)
  plugins: [react()],
  test: { environment: 'node', globals: true },
});
