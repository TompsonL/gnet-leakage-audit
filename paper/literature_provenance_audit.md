# Literature provenance audit (working table, 2026-10-20)

Purpose: test whether the paper's concern is field-level or only a property of the
reproduced testbed. Scope: 14 recent / representative video foreground
segmentation and background-subtraction papers. Evidence is limited to what is
stated in the paper's full text or abstract; "not stated" means we could not
verify it from accessible materials, not that the authors did anything wrong.

Legend: BG = background. "Teacher baseline" = whether the F1 of the BGS/teacher
was found in the accessible text; "No" means not found, **not** proof that the
paper omits it (several publisher PDFs were inaccessible). The manuscript does
not make a field-level teacher claim from this column.

## Table

| # | Method (venue, year) | Network input | Background provenance | Causal / online? | Teacher baseline | Code / weights | Evidence |
|---|---|---|---|---|---|---|---|
| 1 | Kim & Ha (IEEE Access 2020) | BG model image + multiple current images | **SuBSENSE-generated** background model image | BG is BGS-updated; not stated whether future frames are used | No | Not stated | Abstract |
| 2 | Braham & Van Droogenbroeck (IWSSIP 2016) | current patch + one BG image | "background model reduced to a single background image"; generation not stated | Not stated | No | No | Abstract |
| 3 | Babaee et al. (Pattern Recognition 2018) | current frame + BG image (DeepBS) | BG image; generation not verified from accessible text | Not stated | No | Not stated | Secondary mentions |
| 4 | BSUV-Net (WACV 2020) | current frame + empty BG + recent BG + semantic FPM | **empty BG** = median of foreground-free frames in the first 100 frames (or hand-picked); **recent BG** = median of previous 100 frames | Recent BG is causal; empty BG is manually selected from the test video | No (reports DeepLabv3 FPM, not a BGS teacher) | Yes | Full text |
| 5 | BSUV-Net 2.0 (IEEE Access 2021) | same as BSUV-Net | **manually selected empty BG** + median of previous 100 frames; semantic FPM | Recent BG causal; empty BG manual | No | Yes | Full text |
| 6 | BgSubNet (IEEE Sensors J. 2024) | image + background (concatenation) | Background image; synthetic freeform pretraining, semi-supervised fine-tuning; real-scene source not stated in abstract | Not stated | No | Not verified | Abstract |
| 7 | Sauvalle & de La Fortelle (WACV 2023) | current frame + reconstructed BG | Autoencoder trained on the **complete video**; temporal median of reconstructed BGs | **Offline / future frames (oracle)** | No | Announced, not found | Full text |
| 8 | FgSegNet v2 (PAA 2020) | one frame, three scales | **No BG image** | N/A | No (scene-specific GT) | Yes | Full text |
| 9 | ZBS (CVPR 2023) | frame + instance-level BG model (internal) | Zero-shot object detector builds an instance-level BG model; no BG image input | Internal model | No | Yes | Full text |
| 10 | MUSTAN (arXiv 2024) | current frame + multi-scale temporal context | **No BG image** | Temporal context | No | Yes | Full text |
| 11 | ST-TRPCA (IEEE TNNLS 2025) | video tensor (BGS only) | Low-rank/sparse decomposition; no segmentation network | Offline (batch) | N/A | "coming soon" | Full text |
| 12 | Motion-as-Option (IEEE TCSVT 2025) | frame(s) + motion cue (VOS) | **No BG image** | N/A | No | Yes | Full text |
| 13 | RT-SBS (ICIP 2020) | frame + BGS mask + semantic mask | BGS mask (ViBe) refined by semantic segmentation and fed back | Causal BGS; semantic model | No | Patent / code not stated | Full text |
| 14 | GAN-BGS / UTGMP-BS (IEEE TCSVT 2025/26) | video frames (unsupervised BGS) | Pseudo-label generator + message passing; no BG image input | Unsupervised, per-video | No | Not verified | Abstract |

## Summary (as verified)

- **7/14** methods take a background image as a network input (#1–#7);
  the other 7 do not, so the audit should be scoped to *background-image guided*
  methods rather than "all video foreground segmentation".
- Among the 7 background-image methods:
  - **BGS-derived or offline/manual background**: #1 (SuBSENSE-generated),
    #4/#5 (manual/median empty BG + recent median), #7 (complete-video
    autoencoder + temporal median). These are exactly the provenance risks the
    paper formalizes (L1/L3).
  - **Synthetic/paired background**: #6.
  - **Not verifiable from accessible text**: #2, #3.
- **Teacher baselines were not systematically verifiable** from the accessible
  text. We therefore do **not** claim "0/14"; the paper's teacher-baseline
  recommendation rests on the controlled RQ2 experiment, not on this table.
- **Code/weights**: publicly available for BSUV-Net, BSUV-Net 2.0, the
  autoencoder method, FgSegNet v2, ZBS, MUSTAN, Motion-as-Option; not verified
  for several others.
- The most striking case is **BSUV-Net/2.0**: a "video-agnostic" method still
  depends on an **empty background selected from the test video** (median of
  foreground-free frames in the first 100 frames, or hand-picked), in addition to
  a causal recent median. This is a protocol caveat that fits the audit's
  "oracle/manual component" category, even though it is not BGS-guided leakage.

## Table used in the manuscript (5 rows, table*)

Corrections after re-checking (2026-10-20): BgSubNet's real-scene background
source is not stated in the abstract, so the paper now says "BG image (source
n.s.)" rather than "synthetic/paired"; the autoencoder paper says its code
"will be made available" and no repository was found, so Code is `n.s.`.

The paper now uses this compact version as Table 1 in Section 2.1 (no teacher
column, to avoid an unsupported field-wide claim):

| Method | Input | Background source | Online | Code |
|---|---|---|---|---|
| Kim & Ha 2020 | BG + frames | SuBSENSE-generated | BGS | n.s. |
| BSUV-Net 2020 | frame + 2 BG + FPM | selected empty median; recent 100-frame median | manual + causal | yes |
| BSUV-Net 2.0 | frame + 2 BG + FPM | manual empty + 100-frame median | manual + causal | yes |
| BgSubNet 2024 | image + BG | BG image (source n.s.) | n.s. | n.s. |
| Autoencoder 2023 | frame + reconstructed BG | complete-video AE + median | offline | n.s. |

Braham and Babaee are cited in the text as two further methods whose source is
not stated in the accessible materials. "n.s." = not stated.

The full 14-row working table above remains in this file (not in the manuscript)
for the released audit; it should be updated if the publisher PDFs are checked.

## Status and remaining checks

- Inserted into `main.tex`; compiled manuscript is 13 pages, body + appendix
  10 pages, references 3.5 pages, no overfull boxes.
- Do **not** claim "all background-guided methods leak"; claim "among the
  background-image methods we could verify, provenance varies from BGS-generated
  and manually selected backgrounds to offline whole-video reconstruction,
  synthetic/paired backgrounds, and unreported sources."
- Before submission, verify the Braham/Babaee/BgSubNet entries against the
  publisher PDFs if they are to be named in the text; otherwise keep "n.s.".
