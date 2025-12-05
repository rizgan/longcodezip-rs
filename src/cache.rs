//! LLM response caching system

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

/// Cache entry with timestamp
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    /// Cached relevance score
    pub score: f64,
    /// Unix timestamp when cached
    pub timestamp: u64,
}

/// LLM response cache
pub struct ResponseCache {
    /// In-memory cache
    cache: HashMap<String, CacheEntry>,
    /// Cache directory path
    cache_dir: PathBuf,
    /// Time-to-live in seconds (default: 7 days)
    ttl: u64,
    /// Whether caching is enabled
    enabled: bool,
}

impl ResponseCache {
    /// Create a new cache with default settings
    pub fn new() -> Result<Self> {
        Self::with_ttl(7 * 24 * 60 * 60) // 7 days
    }
    
    /// Create a new cache with custom TTL
    pub fn with_ttl(ttl: u64) -> Result<Self> {
        let cache_dir = Self::get_cache_dir()?;
        
        let mut cache = Self {
            cache: HashMap::new(),
            cache_dir,
            ttl,
            enabled: true,
        };
        
        // Load existing cache from disk
        cache.load_from_disk()?;
        
        Ok(cache)
    }
    
    /// Create a disabled cache (no-op)
    pub fn disabled() -> Self {
        Self {
            cache: HashMap::new(),
            cache_dir: PathBuf::new(),
            ttl: 0,
            enabled: false,
        }
    }
    
    /// Get cache directory path (~/.longcodezip/cache)
    fn get_cache_dir() -> Result<PathBuf> {
        let home = dirs::home_dir()
            .ok_or_else(|| Error::ConfigError("Could not determine home directory".to_string()))?;
        
        let cache_dir = home.join(".longcodezip").join("cache");
        
        if !cache_dir.exists() {
            fs::create_dir_all(&cache_dir)?;
        }
        
        Ok(cache_dir)
    }
    
    /// Generate cache key from context and query
    pub fn generate_key(context: &str, query: &str, model: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        context.hash(&mut hasher);
        query.hash(&mut hasher);
        model.hash(&mut hasher);
        
        format!("{:x}", hasher.finish())
    }
    
    /// Get cached score if available and not expired
    pub fn get(&self, key: &str) -> Option<f64> {
        if !self.enabled {
            return None;
        }
        
        if let Some(entry) = self.cache.get(key) {
            // Check if expired
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();
            
            if now - entry.timestamp < self.ttl {
                return Some(entry.score);
            }
        }
        
        None
    }
    
    /// Store score in cache
    pub fn set(&mut self, key: String, score: f64) {
        if !self.enabled {
            return;
        }
        
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        self.cache.insert(key, CacheEntry { score, timestamp });
    }
    
    /// Load cache from disk
    fn load_from_disk(&mut self) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        
        let cache_file = self.cache_dir.join("relevance_cache.json");
        
        if !cache_file.exists() {
            return Ok(());
        }
        
        let data = fs::read_to_string(&cache_file)?;
        let loaded: HashMap<String, CacheEntry> = serde_json::from_str(&data)
            .unwrap_or_else(|_| HashMap::new());
        
        // Filter out expired entries
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        for (key, entry) in loaded {
            if now - entry.timestamp < self.ttl {
                self.cache.insert(key, entry);
            }
        }
        
        Ok(())
    }
    
    /// Save cache to disk
    pub fn save_to_disk(&self) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        
        let cache_file = self.cache_dir.join("relevance_cache.json");
        let data = serde_json::to_string_pretty(&self.cache)?;
        fs::write(&cache_file, data)?;
        
        Ok(())
    }
    
    /// Clear all cached entries
    pub fn clear(&mut self) {
        self.cache.clear();
    }
    
    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let valid_entries = self.cache.iter()
            .filter(|(_, entry)| now - entry.timestamp < self.ttl)
            .count();
        
        CacheStats {
            total_entries: self.cache.len(),
            valid_entries,
            expired_entries: self.cache.len() - valid_entries,
        }
    }
}

impl Drop for ResponseCache {
    fn drop(&mut self) {
        // Save cache when dropped
        let _ = self.save_to_disk();
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub total_entries: usize,
    pub valid_entries: usize,
    pub expired_entries: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_generate_key() {
        let key1 = ResponseCache::generate_key("context", "query", "model");
        let key2 = ResponseCache::generate_key("context", "query", "model");
        let key3 = ResponseCache::generate_key("different", "query", "model");
        
        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }
    
    #[test]
    fn test_disabled_cache() {
        let mut cache = ResponseCache::disabled();
        
        cache.set("key".to_string(), 0.5);
        assert_eq!(cache.get("key"), None);
    }
    
    #[test]
    fn test_cache_operations() {
        // Create a simple in-memory cache for testing
        let mut cache = ResponseCache {
            cache: HashMap::new(),
            cache_dir: PathBuf::new(),
            ttl: 3600,
            enabled: true,
        };
        
        let key = "test_key".to_string();
        
        // Initially empty
        assert_eq!(cache.get(&key), None);
        
        // Set and get
        cache.set(key.clone(), 0.75);
        assert_eq!(cache.get(&key), Some(0.75));
        
        // Clear
        cache.clear();
        assert_eq!(cache.get(&key), None);
    }
}
