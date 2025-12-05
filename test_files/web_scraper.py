"""
Web scraper for collecting product data from e-commerce sites.
Supports pagination, rate limiting, and error recovery.
"""

import requests
from bs4 import BeautifulSoup
import time
import json
from typing import List, Dict, Optional
import logging
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass
import sqlite3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Product:
    """Product data model."""
    name: str
    price: float
    url: str
    description: Optional[str] = None
    rating: Optional[float] = None
    reviews_count: Optional[int] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'price': self.price,
            'url': self.url,
            'description': self.description,
            'rating': self.rating,
            'reviews_count': self.reviews_count
        }

class RateLimiter:
    """Simple rate limiter to avoid overwhelming servers."""
    
    def __init__(self, requests_per_second: float = 1.0):
        self.requests_per_second = requests_per_second
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0
    
    def wait_if_needed(self):
        """Wait if necessary to maintain rate limit."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_interval:
            sleep_time = self.min_interval - time_since_last_request
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()

class WebScraper:
    """Main web scraper class."""
    
    def __init__(self, base_url: str, rate_limit: float = 1.0):
        self.base_url = base_url
        self.rate_limiter = RateLimiter(rate_limit)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def fetch_page(self, url: str, max_retries: int = 3) -> Optional[BeautifulSoup]:
        """Fetch and parse a web page with retry logic."""
        self.rate_limiter.wait_if_needed()
        
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                return BeautifulSoup(response.content, 'html.parser')
            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Failed to fetch {url} after {max_retries} attempts")
                    return None
    
    def extract_products(self, soup: BeautifulSoup) -> List[Product]:
        """Extract product information from parsed HTML."""
        products = []
        
        # This is a generic selector - needs customization per site
        product_elements = soup.select('.product-item')
        
        for element in product_elements:
            try:
                name = element.select_one('.product-name').text.strip()
                price_text = element.select_one('.product-price').text.strip()
                price = float(price_text.replace('$', '').replace(',', ''))
                url = urljoin(self.base_url, element.select_one('a')['href'])
                
                # Optional fields
                description_elem = element.select_one('.product-description')
                description = description_elem.text.strip() if description_elem else None
                
                rating_elem = element.select_one('.product-rating')
                rating = float(rating_elem['data-rating']) if rating_elem else None
                
                reviews_elem = element.select_one('.reviews-count')
                reviews_count = int(reviews_elem.text.strip()) if reviews_elem else None
                
                product = Product(
                    name=name,
                    price=price,
                    url=url,
                    description=description,
                    rating=rating,
                    reviews_count=reviews_count
                )
                products.append(product)
                
            except (AttributeError, ValueError, KeyError) as e:
                logger.warning(f"Failed to parse product: {e}")
                continue
        
        return products
    
    def scrape_category(self, category_url: str, max_pages: int = 5) -> List[Product]:
        """Scrape all products from a category with pagination."""
        all_products = []
        
        for page_num in range(1, max_pages + 1):
            logger.info(f"Scraping page {page_num} of {category_url}")
            
            # Construct paginated URL
            page_url = f"{category_url}?page={page_num}"
            soup = self.fetch_page(page_url)
            
            if not soup:
                break
            
            products = self.extract_products(soup)
            
            if not products:
                logger.info(f"No products found on page {page_num}, stopping")
                break
            
            all_products.extend(products)
            logger.info(f"Found {len(products)} products on page {page_num}")
        
        return all_products

class DataStorage:
    """Store scraped data in SQLite database."""
    
    def __init__(self, db_path: str = 'products.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.create_tables()
    
    def create_tables(self):
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                price REAL NOT NULL,
                url TEXT UNIQUE NOT NULL,
                description TEXT,
                rating REAL,
                reviews_count INTEGER,
                scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.commit()
    
    def save_products(self, products: List[Product]):
        """Save products to database."""
        cursor = self.conn.cursor()
        
        for product in products:
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO products 
                    (name, price, url, description, rating, reviews_count)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    product.name,
                    product.price,
                    product.url,
                    product.description,
                    product.rating,
                    product.reviews_count
                ))
            except sqlite3.Error as e:
                logger.error(f"Database error: {e}")
        
        self.conn.commit()
        logger.info(f"Saved {len(products)} products to database")
    
    def close(self):
        """Close database connection."""
        self.conn.close()

def main():
    """Main execution function."""
    base_url = 'https://example-shop.com'
    category_url = f'{base_url}/category/electronics'
    
    # Initialize scraper
    scraper = WebScraper(base_url, rate_limit=2.0)
    
    # Scrape products
    products = scraper.scrape_category(category_url, max_pages=10)
    
    logger.info(f"Total products scraped: {len(products)}")
    
    # Save to database
    storage = DataStorage()
    storage.save_products(products)
    storage.close()
    
    # Export to JSON
    with open('products.json', 'w', encoding='utf-8') as f:
        json.dump([p.to_dict() for p in products], f, indent=2, ensure_ascii=False)
    
    logger.info("Scraping completed successfully")

if __name__ == '__main__':
    main()
