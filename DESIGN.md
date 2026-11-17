# Design system

## Scene and theme

A student and examiner inspect segmentation details on a laptop in a bright classroom. Use a light, quiet workspace so uploaded photographs remain the strongest visual elements.

## Color strategy

Restrained tinted neutral surfaces with one rust accent for primary actions. Use OKLCH tokens: canvas `oklch(0.97 0.006 70)`, panel `oklch(0.99 0.004 70)`, ink `oklch(0.26 0.008 65)`, muted `oklch(0.47 0.008 65)`, border `oklch(0.85 0.009 70)`, action `oklch(0.48 0.11 40)`. Prediction overlays retain their separately labelled teal data color.

## Typography

Segoe UI/system sans. Main heading 2rem/1.2, section heading 1.25rem, body 1rem/1.5. Prose capped at 72ch. Use tabular numerals for timings.

## Layout and controls

Maximum workspace width 1280px. Narrow input/control column and wider results region. Standard Gradio tabs for single-image inference and model comparison. Image viewers have consistent heights and captions. Stack columns on smaller screens. Use standard buttons, radio selection, slider, file download and inline status text.

## States

Initial state explains what to upload and why a trained checkpoint may be unavailable. Running actions use Gradio progress feedback and queue limits. Errors are plain-language, specific and actionable. Results retain their model identity, threshold and timing. No fabricated example masks are included.
