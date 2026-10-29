# Data Card - Topic Model (LDA)

The project bundles one small, synthetic document collection. There is no external data and
nothing is downloaded.

## bundled collection (`src/topic_model_lda/data.py`)

- **Provenance:** generated for this project. Each document samples words from one of three
  themes (space, cooking, finance), so the latent structure is known.
- **Why this shape:** it makes topic recovery checkable - a good model should recover the three
  themes as distinct topics - and makes coherence meaningful.
- **Licensing:** authored by this repository; MIT.
- **Known gaps:** small, synthetic, three clean themes; real collections are noisier.

## Ethical considerations

The collection is synthetic with neutral general-interest vocabulary; no personal or sensitive
content.
