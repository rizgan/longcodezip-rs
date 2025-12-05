SELECT 
    u.id AS user_id,
    u.username,
    u.email,
    u.created_at,
    COUNT(DISTINCT o.id) AS total_orders,
    SUM(o.total_amount) AS lifetime_value,
    AVG(o.total_amount) AS avg_order_value,
    MAX(o.created_at) AS last_order_date,
    DATEDIFF(CURRENT_DATE, MAX(o.created_at)) AS days_since_last_order,
    CASE 
        WHEN DATEDIFF(CURRENT_DATE, MAX(o.created_at)) <= 30 THEN 'Active'
        WHEN DATEDIFF(CURRENT_DATE, MAX(o.created_at)) <= 90 THEN 'At Risk'
        ELSE 'Churned'
    END AS customer_status
FROM 
    users u
LEFT JOIN 
    orders o ON u.id = o.user_id
LEFT JOIN 
    order_items oi ON o.id = oi.order_id
LEFT JOIN 
    products p ON oi.product_id = p.id
WHERE 
    u.created_at >= DATE_SUB(CURRENT_DATE, INTERVAL 1 YEAR)
    AND u.is_active = TRUE
GROUP BY 
    u.id, u.username, u.email, u.created_at
HAVING 
    COUNT(DISTINCT o.id) > 0
ORDER BY 
    lifetime_value DESC, 
    last_order_date DESC
LIMIT 1000;

-- Create materialized view for performance
CREATE MATERIALIZED VIEW customer_analytics AS
SELECT 
    DATE_TRUNC('month', o.created_at) AS month,
    COUNT(DISTINCT o.user_id) AS active_customers,
    COUNT(DISTINCT o.id) AS total_orders,
    SUM(o.total_amount) AS revenue,
    AVG(o.total_amount) AS avg_order_value,
    SUM(oi.quantity) AS items_sold
FROM 
    orders o
JOIN 
    order_items oi ON o.id = oi.order_id
WHERE 
    o.status = 'completed'
GROUP BY 
    DATE_TRUNC('month', o.created_at)
ORDER BY 
    month DESC;

-- Index for optimization
CREATE INDEX idx_orders_user_date ON orders(user_id, created_at);
CREATE INDEX idx_order_items_product ON order_items(product_id, order_id);
