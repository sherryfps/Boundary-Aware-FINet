# Final prose and attribution review

Reviewed November 17, 2026. The final paper remains seven IEEE conference pages with both application domains, ten tables, twenty numbered equation groups, four figures and eight references.

## Editorial scope

Applied Humanizer 4.0.0 to all seven main sections and fifteen subsections, the abstract, keywords, fourteen captions, table footnotes and bibliography. No section was omitted. Replaced staged objectives, repeated document narration and awkward implementation phrases with direct academic prose. Scientific limitations and meaningful technical contrasts remain.

Preservation checks passed for all numbered equations, inline and display mathematics, table bodies, numerical research tokens, author details, figure paths, existing reference entries and figure/evidence bytes. The only source-material addition is a FINet/F3Net loss-lineage credit and F3Net reference. The exact FINet scalar-BCE implementation and all measured results remain unchanged. No new inference or training was performed.

## Attribution and wording comparison

FINet's architecture, EfficientNet, DCT, datasets and Adam are credited. FINet's training source explicitly credits F3Net for the structure loss. The revised paper now includes that lineage while keeping FINet as the source of the exact implementation. F3Net's reference code uses pixelwise weighted BCE; the retained FINet code uses a scalar batch BCE. The existing mathematical explanation preserves this distinction.

The paper's prose was compared locally with nine public primary documents: FINet's pinned README, network, training and DCT files; EfficientNet, CAMO, COD10K and MVTec publication abstracts; and F3Net's author training code. The detailed report includes source URLs and hashes. Lowercase alphabetic tokens were compared in contiguous six- and eight-word windows after removing bibliography, equations, inline mathematics, table cells and LaTeX markup. No matching windows were found in that external corpus. Short matches with the project's own methodology documents describe the same implemented protocol and were reviewed separately.

## Limits

This is a limited wording and source-attribution review, not a comprehensive plagiarism search. It does not cover subscription databases, unpublished manuscripts or all relevant literature, and exact-phrase matching cannot identify paraphrased copying. The full FINet publisher PDF was not retrieved; its authors' README and implementation were checked. Formula definitions, standard terms, titles and correctly credited methods can also resemble other technical papers without implying plagiarism.

No AI detector or dedicated plagiarism service was run. No AI probability, similarity percentage, plagiarism clearance or detector-passing guarantee is claimed. Manuscript text was not sent to search engines or external checkers. The review improves clarity and attribution; it cannot certify authorship or comprehensive originality.

## Deliverable checks

The final LaTeX compiled successfully. All seven rendered pages were inspected for clipping, overlap, legibility and section flow. The source package includes all required figure PDFs and review notes. Current integrity and preservation checks are recorded in `verification.json`, `humanizer_review.json` and `attribution_review.json`.
