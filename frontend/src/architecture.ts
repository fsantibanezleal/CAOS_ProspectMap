// In-app Architecture / "How it works" modal config (ADR-0058) for ProspectMap.
// Passed to <AppShell config={{ ...config, architecture }}>. The ⓘ header button opens the modal. Each tab pairs one
// themed SVG (frontend/public/svg/tech/) with a bilingual ES/EN body.
import type { ArchitectureConfig } from '@fasl-work/caos-app-shell';

export const architecture: ArchitectureConfig = {
  tabs: [
    {
      id: 'app',
      en: 'The app',
      es: 'La app',
      svg: 'svg/tech/01-the-app.svg',
      body_en:
        'ProspectMap answers "where is the next deposit most likely to be?", it stacks open geophysical / geochemical ' +
        '/ structural evidence layers over a study-area grid and computes a posterior prospectivity map P(deposit | ' +
        'evidence) per cell by Weights of Evidence (a Bayesian log-odds update).\n\n' +
        'It is a real system, not a colourful heatmap. The WofE / logistic-regression engine (frontend/src/mpm/) ' +
        'recomputes the posterior LIVE in the browser when you toggle an evidence layer or switch the method ' +
        '(binarization is automatic at the maximizing-contrast threshold t*); the ' +
        'per-layer weights table, the capture / ROC curves and the conditional-independence test update with it. The ' +
        'whole point is that it exposes WofE’s failure modes HONESTLY, conditional-independence violation inflating ' +
        'the posterior, and random-CV vs spatial-CV inflating the AUC. The synthetic controls (C-NEGATIVE, C-CIVIOLATE, ' +
        'C-RECOVER, C-SATURATE) have known ground truth, so the method is verifiable; the white-box WofE is the authority.',
      body_es:
        'ProspectMap responde "¿dónde es más probable el próximo depósito?", apila capas de evidencia geofísica / ' +
        'geoquímica / estructural abiertas sobre una grilla del área de estudio y calcula un mapa posterior de ' +
        'prospectividad P(depósito | evidencia) por celda con Weights of Evidence (una actualización bayesiana de ' +
        'log-odds).\n\n' +
        'Es un sistema real, no un heatmap vistoso. El motor WofE / regresión logística (frontend/src/mpm/) recalcula ' +
        'el posterior EN VIVO en el navegador al activar/desactivar una capa de evidencia o cambiar el método (la ' +
        'binarización es automática en el umbral de contraste máximo t*); la tabla de ' +
        'pesos por capa, las curvas de captura / ROC y el test de independencia condicional se actualizan con él. Lo ' +
        'central es que expone HONESTAMENTE las fallas de WofE, la violación de independencia condicional que infla el ' +
        'posterior, y el CV aleatorio vs el CV espacial que infla el AUC. Los controles sintéticos (C-NEGATIVE, ' +
        'C-CIVIOLATE, C-RECOVER, C-SATURATE) tienen ground-truth conocido, así el método es verificable; el WofE ' +
        'de caja blanca es la autoridad.',
    },
    {
      id: 'lanes',
      en: 'Lanes, web / offline / compute',
      es: 'Carriles, web / offline / cómputo',
      svg: 'svg/tech/02-lanes.svg',
      body_en:
        'Three lanes. WEB (live, in the browser): the TypeScript WofE / CI / logistic-regression / validation engine ' +
        '(frontend/src/mpm/) recomputes the posterior raster on every control, and onnxruntime-web runs the learned ' +
        'classifier + the geology OOD autoencoder, no server. offline / COMPUTE (your machine, isolated .venv): the ' +
        'Python pipeline bakes the canonical case artifacts (the SAME TS engine via tsx) and the heavy lane (--retrain, ' +
        'torch) trains the two learned models and exports them to ONNX. REPLAY: the small committed artifacts in ' +
        'data/derived are overlaid into the SPA by copy-data.mjs and loaded live; the typed mirror ' +
        '(contract.types.ts) fails the build if the web and the pipeline shapes diverge.',
      body_es:
        'Tres carriles. WEB (en vivo, en el navegador): el motor TypeScript de WofE / CI / regresión logística / ' +
        'validación (frontend/src/mpm/) recalcula el raster posterior con cada control, y onnxruntime-web ejecuta el ' +
        'clasificador aprendido + el autoencoder OOD de geología, sin servidor. offline / CÓMPUTO (tu máquina, .venv ' +
        'aislado): el pipeline Python hornea los artefactos canónicos por caso (el MISMO motor TS vía tsx) y el carril ' +
        'pesado (--retrain, torch) entrena los dos modelos aprendidos y los exporta a ONNX. REPLAY: los artefactos ' +
        'pequeños versionados en data/derived se superponen al SPA con copy-data.mjs y se cargan en vivo; el espejo ' +
        'tipado (contract.types.ts) rompe el build si la web y el pipeline divergen.',
    },
    {
      id: 'web-flow',
      en: 'Web-app flow',
      es: 'Flujo de la web',
      svg: 'svg/tech/03-web-flow.svg',
      body_en:
        'The App page recomputes live: inputs (the case selector or your own evidence stack, plus the per-layer on/off ' +
        'toggles + the method chip; each layer binarizes automatically at t*) feed the TypeScript WofE engine and the onnxruntime-web ' +
        'models, which feed the interactive viz, the canvas prospectivity raster, the per-layer weights table, the ' +
        'capture / ROC curves and the CI readout, each reading values back on hover. The six sibling pages (App · ' +
        'Introduction · Methodology · Implementation · Experiments · Benchmark) are identical across every CAOS ' +
        'product. The build is gated by the contract-type mirror, the artifacts are overlaid by copy-data, vite builds ' +
        'the static output, and GitHub Pages serves it at prospectmap.fasl-work.com.',
      body_es:
        'La página App recalcula en vivo: las entradas (el selector de casos o tu propio stack de evidencia, más los ' +
        'toggles on/off por capa + el chip de método; cada capa se binariza automáticamente en t*) alimentan el motor WofE en ' +
        'TypeScript y los modelos onnxruntime-web, que alimentan la visualización interactiva, el raster de ' +
        'prospectividad en canvas, la tabla de pesos por capa, las curvas de captura / ROC y el lectura de CI, cada uno ' +
        'devolviendo valores al pasar el cursor. Las seis páginas hermanas (App · Introducción · Metodología · ' +
        'Implementación · Experimentos · Benchmark) son idénticas en todos los productos CAOS. El build lo controla el ' +
        'espejo de tipos del contrato, los artefactos los superpone copy-data, vite construye el estático y GitHub ' +
        'Pages lo sirve en prospectmap.fasl-work.com.',
    },
    {
      id: 'science',
      en: 'The science',
      es: 'La ciencia',
      svg: 'svg/tech/04-the-science.svg',
      body_en:
        'Weights of Evidence, step by step: ① binarize each evidence layer at a threshold (the data-driven choice is ' +
        'the maximizing-contrast threshold t* = argmax C(t)); ② per pattern compute W⁺ / W⁻, the contrast ' +
        'C = W⁺ − W⁻ and the studentized contrast C/s(C); ③ the posterior log-odds of a cell = the prior ' +
        'logit + Σ of the present/absent weights, under conditional independence; ④ the Agterberg-Cheng omnibus ' +
        'test (Σ posterior ≈ N(D) under CI) + the CI ratio surface when correlated layers double-count and ' +
        'inflate the posterior, then logistic regression (which fits the layers jointly, no CI needed) is the fix.\n\n' +
        'Honest validation is the spine: the prediction-rate capture curve (% deposits captured vs % area, under ' +
        'SPATIAL cross-validation) and capture@10% are the headline; random-CV is shown beside spatial-CV to expose ' +
        'the inflation. The white-box WofE is always on and transparent, the authority the learned classifier is ' +
        'measured against on the SAME spatial holdout, never a fabricated win.',
      body_es:
        'Weights of Evidence, paso a paso: ① binariza cada capa de evidencia en un umbral (la elección data-driven es ' +
        'el umbral de contraste máximo t* = argmax C(t)); ② por patrón calcula W⁺ / W⁻, el contraste ' +
        'C = W⁺ − W⁻ y el contraste estudentizado C/s(C); ③ el log-odds posterior de una celda = el logit ' +
        'previo + Σ de los pesos presente/ausente, bajo independencia condicional; ④ el test omnibus de ' +
        'Agterberg-Cheng (Σ posterior ≈ N(D) bajo CI) + el CI ratio afloran cuando capas correlacionadas ' +
        'doble-cuentan e inflan el posterior, entonces la regresión logística (que ajusta las capas en conjunto, sin ' +
        'CI) es el arreglo.\n\n' +
        'La validación honesta es la columna: la curva de captura prediction-rate (% depósitos capturados vs % área, ' +
        'bajo cross-validation ESPACIAL) y capture@10% son lo central; el CV aleatorio se muestra junto al espacial ' +
        'para exponer la inflación. El WofE de caja blanca está siempre activo y es transparente, la autoridad contra ' +
        'la que se mide el clasificador aprendido en el MISMO holdout espacial, nunca una victoria fabricada.',
    },
    {
      id: 'design',
      en: 'Data contracts / design',
      es: 'Contratos de datos / diseño',
      svg: 'svg/tech/05-data-contracts.svg',
      body_en:
        'Two validated data contracts bracket the pipeline. Contract 1 (ingestion) defines a valid case bundle, a ' +
        'co-registered evidence cube (grid + layers) + a presence-only deposit point pattern + a study-area mask, with ' +
        'guards (grid/cell positive, ≥ 1 deposit) and honesty flags (presence-only-tiny < 10 deposits ⇒ a black ' +
        'box overfits; single-layer; synthetic). Contract 2 (artifact) defines the output the web reads (the per-layer ' +
        'weights, the posterior summary, the CI diagnostics, the capture / ROC curves, the random-vs-spatial-CV gap, ' +
        'the model index), mirrored exactly by contract.types.ts. Between them the staged deterministic pipeline runs ' +
        'the lane gate (numpy-light by default, --retrain for the heavy torch lane) and writes a provenance manifest, ' +
        'so every result is reproducible and the web can never silently drift.',
      body_es:
        'Dos contratos de datos validados encierran el pipeline. El Contrato 1 (ingesta) define un bundle de caso ' +
        'válido, un cubo de evidencia co-registrado (grilla + capas) + un patrón de puntos de depósitos presence-only ' +
        '+ una máscara del área, con guardas (grilla/celda positivas, ≥ 1 depósito) y flags de honestidad ' +
        '(presence-only-tiny < 10 depósitos ⇒ una caja negra sobreajusta; capa única; sintético). El Contrato 2 ' +
        '(artefacto) define la salida que lee la web (los pesos por capa, el resumen del posterior, los diagnósticos de ' +
        'CI, las curvas de captura / ROC, el gap CV aleatorio-vs-espacial, el índice de modelos), espejado exactamente ' +
        'por contract.types.ts. Entre ambos, el pipeline por etapas y determinista corre el lane gate (numpy-light por ' +
        'defecto, --retrain para el carril pesado de torch) y escribe un manifest de procedencia, de modo que cada ' +
        'resultado es reproducible y la web nunca diverge en silencio.',
    },
  ],
};
