use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};

/// Thread-safe cache with LRU eviction policy
pub struct LRUCache<K, V> {
    capacity: usize,
    cache: Arc<RwLock<HashMap<K, CacheEntry<V>>>>,
    access_order: Arc<Mutex<Vec<K>>>,
}

struct CacheEntry<V> {
    value: V,
    last_access: Instant,
    access_count: u64,
}

impl<K: Clone + Eq + std::hash::Hash, V: Clone> LRUCache<K, V> {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            cache: Arc::new(RwLock::new(HashMap::new())),
            access_order: Arc::new(Mutex::new(Vec::new())),
        }
    }
    
    pub fn get(&self, key: &K) -> Option<V> {
        let mut cache = self.cache.write().unwrap();
        
        if let Some(entry) = cache.get_mut(key) {
            entry.last_access = Instant::now();
            entry.access_count += 1;
            
            // Update access order
            let mut order = self.access_order.lock().unwrap();
            order.retain(|k| k != key);
            order.push(key.clone());
            
            Some(entry.value.clone())
        } else {
            None
        }
    }
    
    pub fn insert(&self, key: K, value: V) {
        let mut cache = self.cache.write().unwrap();
        let mut order = self.access_order.lock().unwrap();
        
        // Check if we need to evict
        if cache.len() >= self.capacity && !cache.contains_key(&key) {
            if let Some(lru_key) = order.first().cloned() {
                cache.remove(&lru_key);
                order.remove(0);
            }
        }
        
        let entry = CacheEntry {
            value,
            last_access: Instant::now(),
            access_count: 0,
        };
        
        cache.insert(key.clone(), entry);
        order.retain(|k| k != &key);
        order.push(key);
    }
    
    pub fn len(&self) -> usize {
        self.cache.read().unwrap().len()
    }
    
    pub fn clear(&self) {
        self.cache.write().unwrap().clear();
        self.access_order.lock().unwrap().clear();
    }
}

/// Simple HTTP server with request handling
pub struct HttpServer {
    host: String,
    port: u16,
    routes: Arc<RwLock<HashMap<String, Box<dyn Fn() -> String + Send + Sync>>>>,
}

impl HttpServer {
    pub fn new(host: &str, port: u16) -> Self {
        Self {
            host: host.to_string(),
            port,
            routes: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub fn route<F>(&mut self, path: &str, handler: F)
    where
        F: Fn() -> String + Send + Sync + 'static,
    {
        let mut routes = self.routes.write().unwrap();
        routes.insert(path.to_string(), Box::new(handler));
    }
    
    pub fn handle_request(&self, path: &str) -> Option<String> {
        let routes = self.routes.read().unwrap();
        routes.get(path).map(|handler| handler())
    }
}

/// Connection pool for database connections
pub struct ConnectionPool<T> {
    connections: Arc<Mutex<Vec<T>>>,
    max_size: usize,
    creator: Arc<dyn Fn() -> T + Send + Sync>,
}

impl<T: Send> ConnectionPool<T> {
    pub fn new<F>(max_size: usize, creator: F) -> Self
    where
        F: Fn() -> T + Send + Sync + 'static,
    {
        Self {
            connections: Arc::new(Mutex::new(Vec::new())),
            max_size,
            creator: Arc::new(creator),
        }
    }
    
    pub fn get_connection(&self) -> Option<T> {
        let mut connections = self.connections.lock().unwrap();
        
        if let Some(conn) = connections.pop() {
            Some(conn)
        } else if connections.len() < self.max_size {
            Some((self.creator)())
        } else {
            None
        }
    }
    
    pub fn return_connection(&self, conn: T) {
        let mut connections = self.connections.lock().unwrap();
        if connections.len() < self.max_size {
            connections.push(conn);
        }
    }
}

/// Worker thread pool for async tasks
pub struct ThreadPool {
    workers: Vec<Worker>,
    sender: std::sync::mpsc::Sender<Job>,
}

type Job = Box<dyn FnOnce() + Send + 'static>;

struct Worker {
    id: usize,
    thread: Option<thread::JoinHandle<()>>,
}

impl ThreadPool {
    pub fn new(size: usize) -> Self {
        assert!(size > 0);
        
        let (sender, receiver) = std::sync::mpsc::channel();
        let receiver = Arc::new(Mutex::new(receiver));
        
        let mut workers = Vec::with_capacity(size);
        
        for id in 0..size {
            workers.push(Worker::new(id, Arc::clone(&receiver)));
        }
        
        ThreadPool { workers, sender }
    }
    
    pub fn execute<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        let job = Box::new(f);
        self.sender.send(job).unwrap();
    }
}

impl Worker {
    fn new(id: usize, receiver: Arc<Mutex<std::sync::mpsc::Receiver<Job>>>) -> Worker {
        let thread = thread::spawn(move || loop {
            let job = receiver.lock().unwrap().recv();
            
            match job {
                Ok(job) => {
                    println!("Worker {} executing job", id);
                    job();
                }
                Err(_) => {
                    println!("Worker {} shutting down", id);
                    break;
                }
            }
        });
        
        Worker {
            id,
            thread: Some(thread),
        }
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        for worker in &mut self.workers {
            if let Some(thread) = worker.thread.take() {
                thread.join().unwrap();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_lru_cache() {
        let cache = LRUCache::new(2);
        
        cache.insert("key1", "value1");
        cache.insert("key2", "value2");
        
        assert_eq!(cache.get(&"key1"), Some("value1"));
        
        cache.insert("key3", "value3");
        assert_eq!(cache.get(&"key2"), None); // Evicted
        assert_eq!(cache.get(&"key1"), Some("value1"));
        assert_eq!(cache.get(&"key3"), Some("value3"));
    }
    
    #[test]
    fn test_http_server() {
        let mut server = HttpServer::new("localhost", 8080);
        
        server.route("/", || "Hello World".to_string());
        server.route("/api", || r#"{"status": "ok"}"#.to_string());
        
        assert_eq!(server.handle_request("/"), Some("Hello World".to_string()));
        assert_eq!(
            server.handle_request("/api"),
            Some(r#"{"status": "ok"}"#.to_string())
        );
        assert_eq!(server.handle_request("/unknown"), None);
    }
}
