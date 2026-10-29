# End-to-End Case Study - Discovering Themes Without Labels

How a topic model finds the themes in a collection it was never told about.

## Scenario

A team has a pile of documents and wants to know what they are about, with no labels and no
predefined categories.

## Discovering topics

```bash
curl localhost:8000/topics
```

```json
{"topics": [
  {"topic": 0, "words": ["rocket", "orbit", "mission", "satellite", "planet", "launch"]},
  {"topic": 1, "words": ["bake", "flour", "oven", "sauce", "dish", "recipe"]},
  {"topic": 2, "words": ["portfolio", "price", "trade", "bank", "stock", "profit"]}]}
```

With no supervision, the model recovered three clear themes - space, cooking, and finance -
each as a distinct, coherent topic.

## Classifying a new document

```bash
curl -s -X POST localhost:8000/infer -d '{"text": "rocket orbit planet galaxy launch"}'
```

The response gives the document's topic distribution and its dominant topic (here, the space
topic), so new documents can be routed by theme.

## What an operator sees

- **Themes, not labels.** The topics emerge from co-occurrence; no taxonomy was defined.
- **A coherence score.** `/evaluate` reports how semantically tight the topics are.
- **Reproducibility.** A fixed seed regenerates the same topics and numbers.

## Takeaway

Latent Dirichlet allocation, implemented from the Gibbs step up, turns an unlabeled document
pile into interpretable themes, which is why it remains a staple of exploratory text analysis.
