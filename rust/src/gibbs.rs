//! Collapsed Gibbs sampling, step for step the same as the Python sampler.

use crate::mt::PyRandom;

/// Sampler state: per-token assignments and the three count tables.
pub struct Sampler {
    documents: Vec<Vec<u32>>,
    vocab: usize,
    k: usize,
    alpha: f64,
    beta: f64,
    compensated: bool,
    rng: PyRandom,
    pub assignments: Vec<Vec<u32>>,
    /// Row-major `k x vocab`.
    pub topic_word: Vec<u32>,
    pub topic_total: Vec<u32>,
    /// Row-major `documents x k`.
    pub doc_topic: Vec<u32>,
}

/// CPython's `sum()` over floats: plain left-to-right addition before 3.12, Neumaier
/// compensated summation from 3.12 on.
fn py_sum(values: &[f64], compensated: bool) -> f64 {
    if !compensated {
        return values.iter().fold(0.0, |acc, &v| acc + v);
    }
    let (mut f, mut c) = (0.0f64, 0.0f64);
    for &x in values {
        let t = f + x;
        if f.abs() >= x.abs() {
            c += (f - t) + x;
        } else {
            c += (x - t) + f;
        }
        f = t;
    }
    if c != 0.0 && c.is_finite() {
        f += c;
    }
    f
}

impl Sampler {
    /// Assign every token a random topic, consuming the generator exactly as
    /// `LDA.__init__` does.
    pub fn new(
        documents: Vec<Vec<u32>>,
        vocab: usize,
        k: usize,
        alpha: f64,
        beta: f64,
        seed: u64,
        compensated: bool,
    ) -> Result<Self, String> {
        if k < 2 {
            return Err("num_topics must be at least 2".into());
        }
        if documents.iter().flatten().any(|&w| w as usize >= vocab) {
            return Err("a word index is outside the vocabulary".into());
        }
        let mut rng = PyRandom::new(seed);
        let mut topic_word = vec![0u32; k * vocab];
        let mut topic_total = vec![0u32; k];
        let mut doc_topic = vec![0u32; documents.len() * k];
        let mut assignments = Vec::with_capacity(documents.len());
        for (d, doc) in documents.iter().enumerate() {
            let mut row = Vec::with_capacity(doc.len());
            for &w in doc {
                let t = rng.randrange(k as u32) as usize;
                row.push(t as u32);
                topic_word[t * vocab + w as usize] += 1;
                topic_total[t] += 1;
                doc_topic[d * k + t] += 1;
            }
            assignments.push(row);
        }
        Ok(Self {
            documents,
            vocab,
            k,
            alpha,
            beta,
            compensated,
            rng,
            assignments,
            topic_word,
            topic_total,
            doc_topic,
        })
    }

    /// Run `iterations` full sweeps over every token.
    pub fn sweep(&mut self, iterations: usize) {
        let (k, v) = (self.k, self.vocab);
        let v_beta = v as f64 * self.beta;
        let mut weights = vec![0.0f64; k];
        for _ in 0..iterations {
            for d in 0..self.documents.len() {
                for pos in 0..self.documents[d].len() {
                    let w = self.documents[d][pos] as usize;
                    let old = self.assignments[d][pos] as usize;
                    self.topic_word[old * v + w] -= 1;
                    self.topic_total[old] -= 1;
                    self.doc_topic[d * k + old] -= 1;
                    for (t, weight) in weights.iter_mut().enumerate() {
                        *weight = (self.doc_topic[d * k + t] as f64 + self.alpha)
                            * (self.topic_word[t * v + w] as f64 + self.beta)
                            / (self.topic_total[t] as f64 + v_beta);
                    }
                    let total = py_sum(&weights, self.compensated);
                    let threshold = self.rng.random() * total;
                    let mut cumulative = 0.0;
                    let mut new = k - 1;
                    for (t, &weight) in weights.iter().enumerate() {
                        cumulative += weight;
                        if cumulative >= threshold {
                            new = t;
                            break;
                        }
                    }
                    self.assignments[d][pos] = new as u32;
                    self.topic_word[new * v + w] += 1;
                    self.topic_total[new] += 1;
                    self.doc_topic[d * k + new] += 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_stay_consistent() {
        let docs = vec![vec![0, 1, 2, 0], vec![3, 4, 3], vec![], vec![1, 1, 1, 2]];
        let mut s = Sampler::new(docs.clone(), 5, 3, 0.1, 0.01, 7, true).unwrap();
        s.sweep(20);
        let tokens: usize = docs.iter().map(Vec::len).sum();
        assert_eq!(s.topic_total.iter().sum::<u32>() as usize, tokens);
        assert_eq!(s.topic_word.iter().sum::<u32>() as usize, tokens);
        assert_eq!(s.doc_topic.iter().sum::<u32>() as usize, tokens);
        for (d, doc) in docs.iter().enumerate() {
            assert_eq!(
                s.doc_topic[d * 3..d * 3 + 3].iter().sum::<u32>() as usize,
                doc.len()
            );
        }
    }

    #[test]
    fn rejects_bad_input() {
        assert!(Sampler::new(vec![vec![0]], 1, 1, 0.1, 0.01, 0, true).is_err());
        assert!(Sampler::new(vec![vec![5]], 3, 2, 0.1, 0.01, 0, true).is_err());
    }

    #[test]
    fn compensated_sum_matches_python() {
        assert_eq!(py_sum(&[0.1, 1e16, -1e16, 0.2], true), 0.30000000000000004);
        assert_eq!(py_sum(&[0.1, 1e16, -1e16, 0.2], false), 0.2);
    }
}
