# Decoding Music — Feature Extraction Pipeline

This is the feature extraction layer of a larger music intelligence project. It takes a song as an audio file, breaks it down into a structured set of numbers that describe how the song sounds, and outputs those numbers as a dataset that can be used for comparison and analysis.

The broader goal is to help artists — specifically emerging Afrobeats artists — understand how their music compares to songs already performing well in the space, and what specifically is working or not working about their sound. This pipeline is the first step in that process: turning audio into something a machine can reason about.

It does not make any judgements on its own. That comes later.

---

## What it actually does

Given an audio file, the pipeline extracts features across several dimensions of the song:

- **Energy and dynamics** — how loud the song is, how much it varies, where it peaks and drops
- **Tonal character** — how bright or dark the mix sounds, how that changes across the song
- **Timbre** — the overall texture and colour of the sound, captured through MFCCs
- **Rhythm** — onset strength, tempo, how consistently and intensely events occur
- **Harmony** — which pitch classes are active, how harmonically clear or complex the song is, what chord qualities dominate
- **Mix definition** — how well-separated the frequencies are across the bass, mids and highs
- **Section-level variation** — how the above properties change across four equal segments of the song

Each song ends up as a single row in a dataset, with all of these properties as columns. The feature set is split into two pools: one for comparison (used to measure similarity between songs) and one for interpretation (used to explain what the numbers mean in plain language).

---

## Project structure

```
src/
├── features/
│   ├── feature_extractor.py     # Main extraction logic — ties everything together
│   ├── feature_schema.py        # Defines which summary stats to compute per feature
│   ├── feature_registry.py      # Declares which features belong to comparison vs interpretation
│   ├── frame_features.py        # Low-level audio feature computation functions
│   ├── temporal_metric.py       # Summary stat functions (mean, std, slope, etc.)
│   └── spectral_representation.py
├── preprocessing/
│   └── audio_loader.py          # Loads audio, applies RMS normalisation, captures loudness
├── pipeline/
│   └── build_features.py        # Runs the pipeline across a folder of audio files
└── utils/
    └── logging_utils.py
```

---

## Setup

Requires Python 3.9 or above.

**Create and activate a virtual environment:**

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Mac/Linux
python -m venv .venv
source .venv/bin/activate
```

**Install dependencies:**

```bash
pip install librosa numpy scipy pandas pyarrow pyloudnorm
```

---

## How to run it

Point the pipeline at a folder containing audio files and it will process each one and write the results to a Parquet file.

```bash
python src/pipeline/build_features.py --input_dir /path/to/audio --output_path /path/to/output.parquet
```

Supported audio formats: `.mp3`, `.wav`, `.flac`, `.ogg`, `.m4a`

---

## Output

The output is a Parquet file — one row per song, one column per feature. Parquet is used over CSV because it's compressed, faster to read, and handles large datasets more cleanly.

The columns fall into two groups:

**Comparison features** — things like MFCCs, chroma, tempo and onset strength. These are used to measure how similar two songs are to each other. They are not directly surfaced to users.

**Interpretation features** — things like RMS energy, spectral centroid, spectral contrast per frequency band, chord ratios and loudness. These have plain English meaning and are what will eventually be communicated to an artist.

---

## Key design decisions worth knowing

**RMS normalisation** — audio is normalised to a consistent average energy level before feature extraction. This means loudness differences from mixing and mastering choices do not dominate the analysis. Absolute loudness is still captured separately (integrated LUFS and dynamic range) from the raw signal before normalisation happens.

**MFCCs start from coefficient 1** — coefficient 0 is dropped because it captures overall energy level, which is already covered by RMS. Keeping it would double-weight loudness in the comparison layer.

**Spectral contrast is computed per frequency band** — seven bands from sub-bass to brilliance, each treated as a separate feature. Collapsing them into one number loses most of the useful information about mix definition.

**Spectral flux is split into positive and negative** — positive flux captures onsets and attacks, negative flux captures decays and releases. Both matter for understanding the dynamic character of a production.

**Section features** — every song is divided into four equal segments and a focused set of features (energy, brightness, onset strength, MFCCs) is computed within each. This is a fixed-length approximation of structural analysis — it won't always align with actual musical sections but it captures how the song changes across its timeline without requiring variable-length output.

---

## What this does not do yet

- Compare songs against each other — that is the next layer, built on top of this output
- Detect actual musical sections (verse, chorus, bridge) — the current segment split is time-based, not structure-based
- Handle very short clips (under ~30 seconds) reliably — some features need enough frames to be meaningful
- Process stems or multi-track audio — this works on full mixed audio only

---

## Dependencies

| Library | What it's used for |
|---|---|
| `librosa` | Core audio loading and feature extraction |
| `numpy` | Array operations throughout |
| `scipy` | Statistical functions (skewness, kurtosis) |
| `pandas` | Dataset construction and output |
| `pyarrow` | Parquet file writing |
| `pyloudnorm` | Integrated LUFS loudness measurement |

---

*This is an early stage project. The extraction layer is complete and validated. The comparison and interpretation layers are in development.*
