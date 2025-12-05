//! Knapsack optimizer for block selection
//!
//! Implements dynamic programming and greedy algorithms to select
//! code blocks that maximize importance within token budget.

use crate::error::Result;
use std::collections::HashSet;

/// Block information for optimization
#[derive(Debug, Clone)]
pub struct Block {
    /// Block index
    pub index: usize,
    /// Text content
    pub text: String,
    /// Token count (weight)
    pub tokens: usize,
    /// Importance score (value)
    pub importance: f64,
}

/// Selection result with statistics
#[derive(Debug, Clone)]
pub struct SelectionResult {
    /// Indices of selected blocks
    pub selected_indices: HashSet<usize>,
    /// Total importance (value) of selection
    pub total_value: f64,
    /// Total tokens (weight) used
    pub total_weight: usize,
    /// Number of preserved blocks
    pub preserved_count: usize,
    /// Selection method used
    pub method: String,
    /// Efficiency (value per token)
    pub efficiency: f64,
}

/// Knapsack optimizer for block selection
pub struct KnapsackOptimizer {
    /// Maximum problem size for exact DP solution
    max_dp_items: usize,
    /// Maximum capacity for exact DP solution
    max_dp_capacity: usize,
}

impl KnapsackOptimizer {
    /// Create a new optimizer with default limits
    ///
    /// # Examples
    ///
    /// ```
    /// use longcodezip::optimizer::KnapsackOptimizer;
    ///
    /// let optimizer = KnapsackOptimizer::new();
    /// ```
    pub fn new() -> Self {
        Self {
            max_dp_items: 100,
            max_dp_capacity: 2000,
        }
    }

    /// Create optimizer with custom limits
    ///
    /// # Arguments
    ///
    /// * `max_dp_items` - Max items for DP algorithm
    /// * `max_dp_capacity` - Max capacity for DP algorithm
    pub fn with_limits(max_dp_items: usize, max_dp_capacity: usize) -> Self {
        Self {
            max_dp_items,
            max_dp_capacity,
        }
    }

    /// Select blocks using knapsack algorithm
    ///
    /// # Arguments
    ///
    /// * `blocks` - Available blocks
    /// * `target_tokens` - Maximum tokens allowed
    /// * `preserved_indices` - Blocks that must be included
    ///
    /// # Returns
    ///
    /// Selection result with chosen blocks and statistics
    pub fn select_blocks(
        &self,
        blocks: &[Block],
        target_tokens: usize,
        preserved_indices: &HashSet<usize>,
    ) -> Result<SelectionResult> {
        if blocks.is_empty() {
            return Ok(SelectionResult {
                selected_indices: HashSet::new(),
                total_value: 0.0,
                total_weight: 0,
                preserved_count: 0,
                method: "empty".to_string(),
                efficiency: 0.0,
            });
        }

        // Calculate preserved blocks' weight
        let preserved_weight: usize = preserved_indices
            .iter()
            .filter_map(|&i| blocks.get(i).map(|b| b.tokens))
            .sum();

        let remaining_budget = target_tokens.saturating_sub(preserved_weight);

        // If no budget left, return only preserved
        if remaining_budget == 0 {
            let preserved_value: f64 = preserved_indices
                .iter()
                .filter_map(|&i| blocks.get(i).map(|b| b.importance))
                .sum();

            return Ok(SelectionResult {
                selected_indices: preserved_indices.clone(),
                total_value: preserved_value,
                total_weight: preserved_weight,
                preserved_count: preserved_indices.len(),
                method: "preserved_only".to_string(),
                efficiency: if preserved_weight > 0 {
                    preserved_value / preserved_weight as f64
                } else {
                    0.0
                },
            });
        }

        // Prepare items for knapsack (excluding preserved)
        let mut items: Vec<(usize, usize, f64)> = Vec::new();
        
        for (i, block) in blocks.iter().enumerate() {
            if preserved_indices.contains(&i) {
                continue;
            }

            let weight = block.tokens;
            let mut value = block.importance;

            // Handle invalid values
            if value.is_nan() || value.is_infinite() {
                value = 0.0;
            }

            items.push((i, weight, value));
        }

        // Sort by value-to-weight ratio (for greedy fallback)
        items.sort_by(|a, b| {
            let ratio_a = a.2 / a.1.max(1) as f64;
            let ratio_b = b.2 / b.1.max(1) as f64;
            ratio_b.partial_cmp(&ratio_a).unwrap()
        });

        // Choose algorithm based on problem size
        let selected = if items.len() <= self.max_dp_items && remaining_budget <= self.max_dp_capacity {
            self.solve_dp(&items, remaining_budget)
        } else {
            self.solve_greedy(&items, remaining_budget)
        };

        // Combine with preserved blocks
        let mut final_selection = preserved_indices.clone();
        final_selection.extend(selected);

        // Calculate statistics
        let total_value: f64 = final_selection
            .iter()
            .filter_map(|&i| blocks.get(i).map(|b| b.importance))
            .sum();

        let total_weight: usize = final_selection
            .iter()
            .filter_map(|&i| blocks.get(i).map(|b| b.tokens))
            .sum();

        let method = if items.len() <= self.max_dp_items && remaining_budget <= self.max_dp_capacity {
            "dynamic_programming".to_string()
        } else {
            "greedy_approximation".to_string()
        };

        Ok(SelectionResult {
            selected_indices: final_selection,
            total_value,
            total_weight,
            preserved_count: preserved_indices.len(),
            method,
            efficiency: if total_weight > 0 {
                total_value / total_weight as f64
            } else {
                0.0
            },
        })
    }

    /// Solve knapsack using dynamic programming (exact solution)
    ///
    /// Time: O(n * capacity), Space: O(n * capacity)
    fn solve_dp(&self, items: &[(usize, usize, f64)], capacity: usize) -> HashSet<usize> {
        let n = items.len();
        if n == 0 || capacity == 0 {
            return HashSet::new();
        }

        // DP table: dp[i][w] = max value using first i items with capacity w
        let mut dp = vec![vec![0.0; capacity + 1]; n + 1];

        // Fill DP table
        for i in 1..=n {
            let (_, weight, value) = items[i - 1];
            
            for w in 0..=capacity {
                // Don't take item i
                dp[i][w] = dp[i - 1][w];

                // Take item i if it fits
                if weight <= w {
                    let take_value = dp[i - 1][w - weight] + value;
                    if take_value > dp[i][w] {
                        dp[i][w] = take_value;
                    }
                }
            }
        }

        // Backtrack to find selected items
        let mut selected = HashSet::new();
        let mut w = capacity;

        for i in (1..=n).rev() {
            if (dp[i][w] - dp[i - 1][w]).abs() > 1e-9 {
                // Item i was selected
                let (idx, weight, _) = items[i - 1];
                selected.insert(idx);
                w = w.saturating_sub(weight);
            }
        }

        selected
    }

    /// Solve knapsack using greedy approximation
    ///
    /// Items should be pre-sorted by value/weight ratio.
    /// Time: O(n), Space: O(1)
    fn solve_greedy(&self, items: &[(usize, usize, f64)], capacity: usize) -> HashSet<usize> {
        let mut selected = HashSet::new();
        let mut remaining_capacity = capacity;

        for &(idx, weight, _value) in items {
            if weight <= remaining_capacity {
                selected.insert(idx);
                remaining_capacity -= weight;
            }
        }

        selected
    }
}

impl Default for KnapsackOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_blocks() -> Vec<Block> {
        vec![
            Block {
                index: 0,
                text: "block0".to_string(),
                tokens: 10,
                importance: 5.0,
            },
            Block {
                index: 1,
                text: "block1".to_string(),
                tokens: 20,
                importance: 15.0,
            },
            Block {
                index: 2,
                text: "block2".to_string(),
                tokens: 15,
                importance: 10.0,
            },
            Block {
                index: 3,
                text: "block3".to_string(),
                tokens: 5,
                importance: 8.0,
            },
        ]
    }

    #[test]
    fn test_knapsack_basic() {
        let optimizer = KnapsackOptimizer::new();
        let blocks = create_test_blocks();
        let preserved = HashSet::new();

        let result = optimizer.select_blocks(&blocks, 30, &preserved).unwrap();

        assert!(result.total_weight <= 30);
        assert!(result.total_value > 0.0);
        assert!(!result.selected_indices.is_empty());
    }

    #[test]
    fn test_knapsack_with_preserved() {
        let optimizer = KnapsackOptimizer::new();
        let blocks = create_test_blocks();
        let mut preserved = HashSet::new();
        preserved.insert(0);

        let result = optimizer.select_blocks(&blocks, 30, &preserved).unwrap();

        // Block 0 must be selected
        assert!(result.selected_indices.contains(&0));
        assert_eq!(result.preserved_count, 1);
    }

    #[test]
    fn test_knapsack_exact_budget() {
        let optimizer = KnapsackOptimizer::new();
        let blocks = create_test_blocks();
        let preserved = HashSet::new();

        // Exact fit for block 1 (20 tokens)
        let result = optimizer.select_blocks(&blocks, 20, &preserved).unwrap();
        
        assert!(result.total_weight <= 20);
    }

    #[test]
    fn test_knapsack_empty_blocks() {
        let optimizer = KnapsackOptimizer::new();
        let blocks: Vec<Block> = vec![];
        let preserved = HashSet::new();

        let result = optimizer.select_blocks(&blocks, 100, &preserved).unwrap();

        assert_eq!(result.selected_indices.len(), 0);
        assert_eq!(result.total_value, 0.0);
    }

    #[test]
    fn test_greedy_vs_dp() {
        let optimizer_dp = KnapsackOptimizer::with_limits(100, 2000);
        let optimizer_greedy = KnapsackOptimizer::with_limits(0, 0); // Force greedy
        
        let blocks = create_test_blocks();
        let preserved = HashSet::new();

        let result_dp = optimizer_dp.select_blocks(&blocks, 30, &preserved).unwrap();
        let result_greedy = optimizer_greedy.select_blocks(&blocks, 30, &preserved).unwrap();

        // Both should find valid solutions
        assert!(result_dp.total_weight <= 30);
        assert!(result_greedy.total_weight <= 30);
        
        // DP should be at least as good as greedy
        assert!(result_dp.total_value >= result_greedy.total_value - 0.01);
    }

    #[test]
    fn test_efficiency_calculation() {
        let optimizer = KnapsackOptimizer::new();
        let blocks = create_test_blocks();
        let preserved = HashSet::new();

        let result = optimizer.select_blocks(&blocks, 50, &preserved).unwrap();

        // Efficiency should be value/weight
        let expected_efficiency = result.total_value / result.total_weight as f64;
        assert!((result.efficiency - expected_efficiency).abs() < 0.01);
    }
}
