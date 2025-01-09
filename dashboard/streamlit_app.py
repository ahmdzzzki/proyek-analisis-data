import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import numpy as np

# Set page config
st.set_page_config(
    page_title="E-commerce Data Analysis",
    page_icon="🛍️",
    layout="wide"
)

# Title and introduction
st.title("🛍️ E-commerce Data Analysis Dashboard")
st.markdown("""
This dashboard analyzes e-commerce data to provide insights about product performance 
and delivery logistics across different regions.
""")

# Function to load data
@st.cache_data
def load_data():
    customers_df = pd.read_csv('customers_dataset.csv')
    orders_df = pd.read_csv('orders_dataset.csv')
    order_reviews_df = pd.read_csv('order_reviews_dataset.csv')
    sellers_df = pd.read_csv('sellers_dataset.csv')
    order_items_df = pd.read_csv('order_items_dataset.csv')
    products_df = pd.read_csv('products_dataset.csv')
    
    # Convert date columns
    date_columns = [
        'order_purchase_timestamp',
        'order_approved_at',
        'order_delivered_carrier_date',
        'order_delivered_customer_date',
        'order_estimated_delivery_date'
    ]
    
    for col in date_columns:
        if col in orders_df.columns:
            orders_df[col] = pd.to_datetime(orders_df[col])
    
    return customers_df, orders_df, order_reviews_df, sellers_df, order_items_df, products_df

# Load data
try:
    with st.spinner('Loading data...'):
        customers_df, orders_df, order_reviews_df, sellers_df, order_items_df, products_df = load_data()
    st.success('Data loaded successfully!')
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Sidebar
st.sidebar.header("Navigation")
analysis_type = st.sidebar.radio(
    "Choose Analysis",
    ["Product Category Performance", "Delivery Logistics", "Customer RFM Analysis"]
)

# Product Category Performance Analysis
if analysis_type == "Product Category Performance":
    st.header("Product Category Performance Analysis")
    
    # Merge relevant dataframes
    merged_df = order_items_df.merge(orders_df, on='order_id')
    merged_df = merged_df.merge(order_reviews_df[['order_id', 'review_score']], on='order_id', how='left')
    merged_df = merged_df.merge(products_df[['product_id', 'product_category_name']], on='product_id', how='left')
    
    # Calculate category performance
    category_performance = merged_df.groupby('product_category_name').agg(
        total_sales=('price', 'sum'),
        average_rating=('review_score', 'mean'),
        order_count=('order_id', 'count')
    ).reset_index()
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Categories", len(category_performance))
    with col2:
        st.metric("Average Rating", f"{category_performance['average_rating'].mean():.2f}")
    with col3:
        st.metric("Total Sales", f"${category_performance['total_sales'].sum():,.2f}")
    
    # Visualizations
    tab1, tab2 = st.tabs(["Sales Analysis", "Rating Analysis"])
    
    with tab1:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(
            data=category_performance.sort_values(by='total_sales', ascending=False).head(10),
            x='total_sales',
            y='product_category_name',
            palette='viridis'
        )
        plt.title('Top 10 Categories by Total Sales')
        plt.xlabel('Total Sales ($)')
        plt.ylabel('Product Category')
        st.pyplot(fig)
        
    with tab2:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(
            data=category_performance.sort_values(by='average_rating', ascending=False).head(10),
            x='average_rating',
            y='product_category_name',
            palette='coolwarm'
        )
        plt.title('Top 10 Categories by Average Rating')
        plt.xlabel('Average Rating')
        plt.ylabel('Product Category')
        st.pyplot(fig)

# Delivery Logistics Analysis
elif analysis_type == "Delivery Logistics":
    st.header("Delivery Logistics Analysis")
    
    # Prepare logistics data
    logistics_df = order_items_df.merge(
        orders_df[['order_id', 'order_delivered_customer_date', 'order_purchase_timestamp']], 
        on='order_id'
    )
    logistics_df = logistics_df.merge(sellers_df[['seller_id', 'seller_state']], on='seller_id', how='left')
    
    logistics_df['delivery_time_days'] = (
        logistics_df['order_delivered_customer_date'] - 
        logistics_df['order_purchase_timestamp']
    ).dt.days
    
    state_delivery_performance = logistics_df.groupby('seller_state').agg(
        average_delivery_time=('delivery_time_days', 'mean'),
        total_orders=('order_id', 'count')
    ).reset_index()
    
    # Display metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Average Delivery Time", 
                 f"{state_delivery_performance['average_delivery_time'].mean():.1f} days")
    with col2:
        st.metric("Total States", len(state_delivery_performance))
    
    # Visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=state_delivery_performance.sort_values(by='average_delivery_time', ascending=False),
        x='seller_state',
        y='average_delivery_time',
        palette='mako'
    )
    plt.title('Average Delivery Time by Seller State')
    plt.xlabel('Seller State')
    plt.ylabel('Average Delivery Time (Days)')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Display detailed statistics
    st.subheader("Delivery Performance by State")
    st.dataframe(
        state_delivery_performance.sort_values(by='average_delivery_time', ascending=False)
        .style.format({
            'average_delivery_time': '{:.1f}',
            'total_orders': '{:,.0f}'
        })
    )

# Customer RFM Analysis
else:
    st.header("Customer RFM Analysis")
    
    # Prepare RFM data
    rfm_df = orders_df.merge(order_items_df, on='order_id')
    rfm_df = rfm_df.merge(customers_df, on='customer_id')
    
    reference_date = rfm_df['order_purchase_timestamp'].max() + timedelta(days=1)
    rfm_table = rfm_df.groupby('customer_unique_id').agg(
        Recency=('order_purchase_timestamp', lambda x: (reference_date - x.max()).days),
        Frequency=('order_id', 'nunique'),
        Monetary=('price', 'sum')
    ).reset_index()
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Recency", f"{rfm_table['Recency'].mean():.1f} days")
    with col2:
        st.metric("Average Frequency", f"{rfm_table['Frequency'].mean():.2f} orders")
    with col3:
        st.metric("Average Monetary", f"${rfm_table['Monetary'].mean():.2f}")
    
    # Visualizations
    tab1, tab2, tab3 = st.tabs(["Recency Distribution", "Frequency Distribution", "Monetary Distribution"])
    
    with tab1:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=rfm_table, x='Recency', bins=50)
        plt.title('Distribution of Customer Recency')
        plt.xlabel('Recency (days)')
        plt.ylabel('Count')
        st.pyplot(fig)
        
    with tab2:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=rfm_table, x='Frequency', bins=30)
        plt.title('Distribution of Customer Frequency')
        plt.xlabel('Frequency (orders)')
        plt.ylabel('Count')
        st.pyplot(fig)
        
    with tab3:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=rfm_table[rfm_table['Monetary'] <= rfm_table['Monetary'].quantile(0.95)], 
                    x='Monetary', bins=50)
        plt.title('Distribution of Customer Monetary Value (95th percentile)')
        plt.xlabel('Monetary Value ($)')
        plt.ylabel('Count')
        st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("Created by Ahmad Zaki | Data Analysis Project")