import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime

# Initialize Faker for realistic product names
fake = Faker()

# Parameters
num_products = 200  # Number of unique products
years = [2019, 2020, 2021, 2022]
months = list(range(1, 13))
num_entries = 200000

# Generate product pool with realistic names and costs
products = []
for _ in range(num_products):
    product_type = random.choice(['Electronics', 'Clothing', 'Grocery', 'Home', 'Beauty'])
    
    if product_type == 'Electronics':
        name = f"{fake.word().capitalize()} {random.choice(['Smartphone', 'Laptop', 'Headphones', 'Tablet', 'Smartwatch'])}"
        cost = round(random.uniform(100, 2000), 2)
    elif product_type == 'Clothing':
        name = f"{fake.color_name()} {random.choice(['T-Shirt', 'Jeans', 'Dress', 'Jacket', 'Sneakers'])}"
        cost = round(random.uniform(15, 200), 2)
    elif product_type == 'Grocery':
        name = f"{fake.word().capitalize()} {random.choice(['Pasta', 'Cereal', 'Snacks', 'Beverage', 'Coffee'])}"
        cost = round(random.uniform(1, 15), 2)
    elif product_type == 'Home':
        name = f"{fake.word().capitalize()} {random.choice(['Lamp', 'Chair', 'Table', 'Decor', 'Cookware'])}"
        cost = round(random.uniform(20, 500), 2)
    else:  # Beauty
        name = f"{fake.word().capitalize()} {random.choice(['Shampoo', 'Perfume', 'Cream', 'Makeup', 'Serum'])}"
        cost = round(random.uniform(5, 150), 2)
    
    products.append({
        'product_name': name,
        'product_cost': cost
    })

# Generate sales data
data = []
for _ in range(num_entries):
    product = random.choice(products)
    year = random.choice(years)
    month = random.choice(months)
    
    # Base sales with some seasonality and random fluctuations
    if month in [11, 12]:  # Holiday season boost
        base_sales = random.randint(50, 300)
    elif month in [6, 7]:  # Summer slump
        base_sales = random.randint(10, 100)
    else:
        base_sales = random.randint(20, 150)
    
    # Add some yearly growth (2020 dip due to pandemic simulation)
    if year == 2019:
        sales = base_sales * random.uniform(0.9, 1.1)
    elif year == 2020:
        sales = base_sales * random.uniform(0.7, 1.0)  # Reduced sales
    elif year == 2021:
        sales = base_sales * random.uniform(1.0, 1.3)  # Recovery
    else:  # 2022
        sales = base_sales * random.uniform(1.1, 1.5)  # Growth
    
    # Add product-specific multiplier (some products sell better than others)
    sales = int(sales * (product['product_cost']/100 + 0.5))
    
    data.append({
        'Product Name': product['product_name'],
        'Product Cost': product['product_cost'],
        'Year': year,
        'Month': month,
        'Monthly Sales': max(1, int(sales))  # Ensure at least 1 sale
    })

# Create DataFrame
df = pd.DataFrame(data)

# Add date column for easier time series analysis
df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str))

# Save to CSV
df.to_csv('retail_sales_dataset.csv', index=False)
print("Dataset generated with shape:", df.shape)