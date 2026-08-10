//! Least-connections load balancing policy with tie-breaking jitter.
//!
//! Picks the worker with the lowest in-flight load. To avoid the herding
//! problem of a strict argmin at high QPS, any worker within
//! `TIE_THRESHOLD` of the minimum is considered a tie and one is chosen at
//! random. This keeps fast workers preferred when there's a real gap, while
//! still spreading load smoothly across near-equal workers.
//!
//! Compared with power-of-two, this can route a larger share of requests to
//! the fastest workers when the worker pool is heterogeneous (e.g. a few fast
//! GPUs alongside many slower ones).

use super::{get_healthy_worker_indices, LoadBalancingPolicy, RequestHeaders};
use crate::core::Worker;
use crate::metrics::RouterMetrics;
use rand::Rng;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Workers within this many connections of the global minimum are treated as a tie
/// and picked uniformly at random. 0 = strict argmin (more deterministic, more herding).
const TIE_THRESHOLD: isize = 2;

#[derive(Debug)]
pub struct LeastConnectionsPolicy {
    /// Cached load information from external monitoring (matches PowerOfTwoPolicy).
    cached_loads: RwLock<HashMap<String, isize>>,
}

impl LeastConnectionsPolicy {
    pub fn new() -> Self {
        Self {
            cached_loads: RwLock::new(HashMap::new()),
        }
    }

    fn get_worker_load(&self, worker: &dyn Worker) -> isize {
        if let Ok(loads) = self.cached_loads.read() {
            if let Some(&load) = loads.get(worker.url()) {
                return load;
            }
        }
        worker.load() as isize
    }
}

impl LoadBalancingPolicy for LeastConnectionsPolicy {
    fn select_worker_with_headers(
        &self,
        workers: &[Arc<dyn Worker>],
        _request_text: Option<&str>,
        _headers: Option<&RequestHeaders>,
    ) -> Option<usize> {
        let healthy_indices = get_healthy_worker_indices(workers);
        if healthy_indices.is_empty() {
            return None;
        }
        if healthy_indices.len() == 1 {
            return Some(healthy_indices[0]);
        }

        // Find minimum load across healthy workers.
        let mut min_load = isize::MAX;
        for &idx in &healthy_indices {
            let load = self.get_worker_load(workers[idx].as_ref());
            if load < min_load {
                min_load = load;
            }
        }

        // Collect all candidates within the tie threshold of min.
        let cutoff = min_load.saturating_add(TIE_THRESHOLD);
        let candidates: Vec<usize> = healthy_indices
            .iter()
            .copied()
            .filter(|&idx| self.get_worker_load(workers[idx].as_ref()) <= cutoff)
            .collect();

        let mut rng = rand::rng();
        let selected_idx = if candidates.is_empty() {
            healthy_indices[0]
        } else {
            candidates[rng.random_range(0..candidates.len())]
        };

        let selected = workers[selected_idx].as_ref();
        selected.increment_processed();
        RouterMetrics::record_policy_decision(self.name(), selected.url(), selected.instance());

        Some(selected_idx)
    }

    fn name(&self) -> &'static str {
        "least_connections"
    }

    fn update_loads(&self, loads: &HashMap<String, isize>) {
        if let Ok(mut cached) = self.cached_loads.write() {
            *cached = loads.clone();
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Default for LeastConnectionsPolicy {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorker, WorkerType};

    #[test]
    fn picks_min_load_worker_most_of_the_time() {
        let policy = LeastConnectionsPolicy::new();
        let w0 = BasicWorker::new("http://w0:8000".to_string(), WorkerType::Regular);
        let w1 = BasicWorker::new("http://w1:8000".to_string(), WorkerType::Regular);
        let w2 = BasicWorker::new("http://w2:8000".to_string(), WorkerType::Regular);
        for _ in 0..50 { w0.increment_load(); }
        for _ in 0..20 { w1.increment_load(); }
        // w2 stays at 0 — should be picked dominantly
        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(w0), Arc::new(w1), Arc::new(w2)];

        let mut counts = [0; 3];
        for _ in 0..200 {
            if let Some(idx) = policy.select_worker(&workers, None) {
                counts[idx] += 1;
            }
        }
        assert!(counts[2] > 190);   // overwhelmingly picks the min worker
        assert!(counts[0] <= 5);    // virtually never picks the most loaded
    }

    #[test]
    fn ties_within_threshold_are_split() {
        let policy = LeastConnectionsPolicy::new();
        let w0 = BasicWorker::new("http://w0:8000".to_string(), WorkerType::Regular);
        let w1 = BasicWorker::new("http://w1:8000".to_string(), WorkerType::Regular);
        // Same load → tie → roughly 50/50 over many samples
        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(w0), Arc::new(w1)];

        let mut counts = [0; 2];
        for _ in 0..1000 {
            if let Some(idx) = policy.select_worker(&workers, None) {
                counts[idx] += 1;
            }
        }
        // Allow generous slack but both must be picked substantially.
        assert!(counts[0] > 300 && counts[1] > 300);
    }
}
