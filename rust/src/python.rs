//! PyO3 bindings, built by maturin into the `topic_model_lda_rs` extension module.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::Sampler as Core;

#[pyclass(name = "Sampler")]
struct PySampler {
    inner: Core,
}

#[pymethods]
impl PySampler {
    #[new]
    #[pyo3(signature = (documents, vocab_size, num_topics, alpha, beta, seed, compensated))]
    fn new(
        documents: Vec<Vec<u32>>,
        vocab_size: usize,
        num_topics: usize,
        alpha: f64,
        beta: f64,
        seed: u64,
        compensated: bool,
    ) -> PyResult<Self> {
        Core::new(
            documents,
            vocab_size,
            num_topics,
            alpha,
            beta,
            seed,
            compensated,
        )
        .map(|inner| Self { inner })
        .map_err(PyValueError::new_err)
    }

    /// Run `iterations` sweeps with the GIL released.
    fn sweep(&mut self, py: Python<'_>, iterations: usize) {
        let inner = &mut self.inner;
        py.detach(|| inner.sweep(iterations));
    }

    fn assignments(&self) -> Vec<Vec<u32>> {
        self.inner.assignments.clone()
    }

    fn topic_word(&self) -> Vec<u32> {
        self.inner.topic_word.clone()
    }

    fn topic_total(&self) -> Vec<u32> {
        self.inner.topic_total.clone()
    }

    fn doc_topic(&self) -> Vec<u32> {
        self.inner.doc_topic.clone()
    }
}

#[pymodule]
fn topic_model_lda_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySampler>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
