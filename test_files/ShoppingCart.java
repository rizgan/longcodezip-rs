import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;

/**
 * E-commerce shopping cart system with inventory management.
 */
public class ShoppingCart {
    
    private final Map<String, CartItem> items;
    private final InventoryService inventory;
    private final PricingService pricing;
    private String customerId;
    private CouponCode appliedCoupon;
    
    public ShoppingCart(String customerId, InventoryService inventory, PricingService pricing) {
        this.customerId = customerId;
        this.items = new ConcurrentHashMap<>();
        this.inventory = inventory;
        this.pricing = pricing;
    }
    
    /**
     * Add item to cart with quantity validation.
     */
    public synchronized boolean addItem(String productId, int quantity) {
        if (quantity <= 0) {
            throw new IllegalArgumentException("Quantity must be positive");
        }
        
        // Check inventory availability
        if (!inventory.checkAvailability(productId, quantity)) {
            return false;
        }
        
        CartItem existing = items.get(productId);
        if (existing != null) {
            int newQuantity = existing.getQuantity() + quantity;
            if (!inventory.checkAvailability(productId, newQuantity)) {
                return false;
            }
            existing.setQuantity(newQuantity);
        } else {
            Product product = inventory.getProduct(productId);
            if (product == null) {
                throw new IllegalArgumentException("Product not found: " + productId);
            }
            items.put(productId, new CartItem(product, quantity));
        }
        
        return true;
    }
    
    /**
     * Remove item from cart.
     */
    public synchronized void removeItem(String productId) {
        items.remove(productId);
    }
    
    /**
     * Update item quantity.
     */
    public synchronized boolean updateQuantity(String productId, int newQuantity) {
        if (newQuantity <= 0) {
            removeItem(productId);
            return true;
        }
        
        if (!inventory.checkAvailability(productId, newQuantity)) {
            return false;
        }
        
        CartItem item = items.get(productId);
        if (item != null) {
            item.setQuantity(newQuantity);
            return true;
        }
        
        return false;
    }
    
    /**
     * Apply coupon code to cart.
     */
    public boolean applyCoupon(String code) {
        CouponCode coupon = pricing.validateCoupon(code);
        if (coupon == null || !coupon.isValid()) {
            return false;
        }
        
        if (getSubtotal() < coupon.getMinimumPurchase()) {
            return false;
        }
        
        this.appliedCoupon = coupon;
        return true;
    }
    
    /**
     * Calculate subtotal before discounts.
     */
    public double getSubtotal() {
        return items.values().stream()
                .mapToDouble(item -> item.getProduct().getPrice() * item.getQuantity())
                .sum();
    }
    
    /**
     * Calculate discount amount.
     */
    public double getDiscount() {
        if (appliedCoupon == null) {
            return 0.0;
        }
        
        double subtotal = getSubtotal();
        
        if (appliedCoupon.isPercentage()) {
            double discount = subtotal * (appliedCoupon.getValue() / 100.0);
            return Math.min(discount, appliedCoupon.getMaxDiscount());
        } else {
            return Math.min(appliedCoupon.getValue(), subtotal);
        }
    }
    
    /**
     * Calculate tax based on location.
     */
    public double getTax(String zipCode) {
        double taxRate = pricing.getTaxRate(zipCode);
        double taxableAmount = getSubtotal() - getDiscount();
        return taxableAmount * taxRate;
    }
    
    /**
     * Calculate shipping cost.
     */
    public double getShipping(String zipCode) {
        double weight = items.values().stream()
                .mapToDouble(item -> item.getProduct().getWeight() * item.getQuantity())
                .sum();
        
        return pricing.calculateShipping(weight, zipCode);
    }
    
    /**
     * Calculate total including tax and shipping.
     */
    public double getTotal(String zipCode) {
        return getSubtotal() - getDiscount() + getTax(zipCode) + getShipping(zipCode);
    }
    
    /**
     * Checkout and create order.
     */
    public Order checkout(PaymentMethod payment, String shippingAddress, String zipCode) {
        // Validate cart not empty
        if (items.isEmpty()) {
            throw new IllegalStateException("Cart is empty");
        }
        
        // Reserve inventory
        List<String> reservationIds = new ArrayList<>();
        try {
            for (CartItem item : items.values()) {
                String reservationId = inventory.reserveStock(
                    item.getProduct().getId(), 
                    item.getQuantity()
                );
                reservationIds.add(reservationId);
            }
            
            // Process payment
            double total = getTotal(zipCode);
            String transactionId = payment.charge(total);
            
            // Create order
            Order order = new Order(
                generateOrderId(),
                customerId,
                new ArrayList<>(items.values()),
                getSubtotal(),
                getDiscount(),
                getTax(zipCode),
                getShipping(zipCode),
                total,
                shippingAddress,
                transactionId
            );
            
            // Clear cart
            items.clear();
            appliedCoupon = null;
            
            return order;
            
        } catch (Exception e) {
            // Release reservations on failure
            for (String reservationId : reservationIds) {
                inventory.releaseReservation(reservationId);
            }
            throw new RuntimeException("Checkout failed: " + e.getMessage(), e);
        }
    }
    
    /**
     * Get cart summary.
     */
    public CartSummary getSummary(String zipCode) {
        return new CartSummary(
            items.size(),
            items.values().stream().mapToInt(CartItem::getQuantity).sum(),
            getSubtotal(),
            getDiscount(),
            getTax(zipCode),
            getShipping(zipCode),
            getTotal(zipCode),
            appliedCoupon != null ? appliedCoupon.getCode() : null
        );
    }
    
    private String generateOrderId() {
        return "ORD-" + System.currentTimeMillis() + "-" + customerId;
    }
    
    // Inner classes
    
    public static class CartItem {
        private final Product product;
        private int quantity;
        
        public CartItem(Product product, int quantity) {
            this.product = product;
            this.quantity = quantity;
        }
        
        public Product getProduct() { return product; }
        public int getQuantity() { return quantity; }
        public void setQuantity(int quantity) { this.quantity = quantity; }
    }
    
    public static class CartSummary {
        private final int itemCount;
        private final int totalQuantity;
        private final double subtotal;
        private final double discount;
        private final double tax;
        private final double shipping;
        private final double total;
        private final String couponCode;
        
        public CartSummary(int itemCount, int totalQuantity, double subtotal,
                          double discount, double tax, double shipping, 
                          double total, String couponCode) {
            this.itemCount = itemCount;
            this.totalQuantity = totalQuantity;
            this.subtotal = subtotal;
            this.discount = discount;
            this.tax = tax;
            this.shipping = shipping;
            this.total = total;
            this.couponCode = couponCode;
        }
        
        // Getters omitted for brevity
    }
}
