-- Category translation (small lookup table, load first)
CREATE TABLE category_translation (
    product_category_name         VARCHAR(100) PRIMARY KEY,
    product_category_name_english VARCHAR(100)
);

-- Customers
CREATE TABLE customers (
    customer_id               VARCHAR(32) PRIMARY KEY,
    customer_unique_id        VARCHAR(32) NOT NULL,
    customer_zip_code_prefix  INTEGER,
    customer_city             VARCHAR(100),
    customer_state            VARCHAR(2)
);

-- Sellers
CREATE TABLE sellers (
    seller_id               VARCHAR(32) PRIMARY KEY,
    seller_zip_code_prefix  INTEGER,
    seller_city             VARCHAR(100),
    seller_state            VARCHAR(2)
);

-- Products
CREATE TABLE products (
    product_id             VARCHAR(32) PRIMARY KEY,
    product_category_name  VARCHAR(100) REFERENCES category_translation(product_category_name),
    product_weight_g       NUMERIC,
    product_length_cm      NUMERIC,
    product_height_cm      NUMERIC,
    product_width_cm       NUMERIC
);

-- Orders
CREATE TABLE orders (
    order_id                        VARCHAR(32) PRIMARY KEY,
    customer_id                     VARCHAR(32) REFERENCES customers(customer_id),
    order_status                    VARCHAR(20),
    order_purchase_timestamp        TIMESTAMP,
    order_approved_at               TIMESTAMP,
    order_delivered_carrier_date    TIMESTAMP,
    order_delivered_customer_date   TIMESTAMP,
    order_estimated_delivery_date   TIMESTAMP
);

-- Order items (composite PK, order can have multiple line items)
CREATE TABLE order_items (
    order_id             VARCHAR(32) REFERENCES orders(order_id),
    order_item_id        INTEGER,
    product_id           VARCHAR(32) REFERENCES products(product_id),
    seller_id            VARCHAR(32) REFERENCES sellers(seller_id),
    shipping_limit_date  TIMESTAMP,
    price                NUMERIC(10, 2),
    freight_value        NUMERIC(10, 2),
    PRIMARY KEY (order_id, order_item_id)
);

-- Order payments (composite PK, order can have multiple payment rows)
CREATE TABLE order_payments (
    order_id               VARCHAR(32) REFERENCES orders(order_id),
    payment_sequential     INTEGER,
    payment_type           VARCHAR(20),
    payment_installments   INTEGER,
    payment_value          NUMERIC(10, 2),
    PRIMARY KEY (order_id, payment_sequential)
);