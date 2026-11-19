# Phase 5 visual and interaction review

These screenshots show the local Streamlit application running on 2026-11-19. Desktop captures use a 1440 × 1200 browser viewport and show the visible portion of each vertically scrolling page.

| Page | Actual capture |
| --- | --- |
| Research Overview | [Screenshot](../artifacts/phase5/overview-desktop.png) |
| Dataset Explorer | [Screenshot](../artifacts/phase5/dataset-desktop.png) |
| Model Laboratory | [Screenshot](../artifacts/phase5/laboratory-desktop.png) |
| Hospital Reliability | [Screenshot](../artifacts/phase5/reliability-desktop.png) |
| Calibration & Sensitivity | [Screenshot](../artifacts/phase5/calibration-desktop.png) |
| Prediction Sandbox | [Screenshot](../artifacts/phase5/sandbox-desktop.png) |
| Prediction Sandbox at 390 × 844 | [Screenshot](../artifacts/phase5/sandbox-narrow.png) |
| Experiment Registry | [Screenshot](../artifacts/phase5/registry-desktop.png) |
| Methodology & Reproducibility | [Screenshot](../artifacts/phase5/methodology-desktop.png) |

All eight pages were opened in a real browser, and their rendered screenshots inspected. Navy/slate surfaces, chart contrast, table boundaries, navigation groups, labels, legends and long text were reviewed. Main content had no horizontal overflow in the measured desktop views. Tables use native scrolling when their columns exceed the available width. Screenshots were taken during the final presentation review; subsequent documentation and Phase 2 audit details do not change the principal layouts shown.

Live interactions included navigation, SQLite initialization/import of 102 aggregate records, selection of Switzerland and the alternative cholesterol policy, and submission of Fictional A through the verified model. The sandbox returned an experimental score of 0.050 and historical class 0 at the locked 0.45 threshold. This is a UI demonstration on synthetic inputs, not new research or a clinical risk estimate.

The sandbox was inspected at widths of 501 and 390 CSS pixels. At 390 × 844, the sidebar collapses, the two columns stack, and the submitted output follows the form. The narrow capture is scrolled to show the last form field, submit button and result together.

Report-missing/schema errors, disabled-model submission, registry failures, selection changes, reset behavior and no-training/no-persistence boundaries are covered by Streamlit AppTest and service tests. This is not a comprehensive screen-reader, cross-browser or device certification. A dedicated accessibility pass is a proposed Phase 6 check.

In a local microbenchmark, checking and loading all six artifacts averaged about 7.5 ms across 100 loads. Browser headings appeared in roughly 0.3–2.5 seconds during the first navigation check. Those timings include browser automation overhead. The complete automated suite passed **69 tests**, including all 39 original research tests.

With file watching disabled, application code edits require a server restart. A direct visit to a non-root URL produced initial Streamlit health and host-configuration probe 404s, followed by a successful fallback to the root endpoints. Native sidebar navigation worked.

No public deployment, push, remote database provisioning or Phase 6 work occurred.
