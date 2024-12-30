import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="E-commerce Analysis Dashboard",
    page_icon="🛍️",
    layout="wide"
)

# Function to load data
@st.cache_data
def load_data():
    products_df = pd.read_csv('/workspaces/proyek-analisis-data/data/products_dataset.csv')
    orders_df = pd.read_csv('/workspaces/proyek-analisis-data/data/orders_dataset.csv')
    order_reviews_df = pd.read_csv('/workspaces/proyek-analisis-data/data/order_reviews_dataset.csv')
    sellers_df = pd.read_csv('/workspaces/proyek-analisis-data/data/sellers_dataset.csv')
    order_items_df = pd.read_csv('/workspaces/proyek-analisis-data/data/order_items_dataset.csv')
    
    # Clean and prepare data
    products_df = products_df.dropna(subset=['product_category_name'])
    products_df['product_weight_g'] = products_df['product_weight_g'].fillna(products_df['product_weight_g'].mean())
    products_df['product_length_cm'] = products_df['product_length_cm'].fillna(products_df['product_length_cm'].mean())
    products_df['product_height_cm'] = products_df['product_height_cm'].fillna(products_df['product_height_cm'].mean())
    products_df['product_width_cm'] = products_df['product_width_cm'].fillna(products_df['product_width_cm'].mean())
    
    orders_df['order_purchase_timestamp'] = pd.to_datetime(orders_df['order_purchase_timestamp'])
    orders_df['order_delivered_customer_date'] = pd.to_datetime(orders_df['order_delivered_customer_date'])
    
    return products_df, orders_df, order_reviews_df, sellers_df, order_items_df

# Load data
products_df, orders_df, order_reviews_df, sellers_df, order_items_df = load_data()

# Title
st.title("🛍️ E-commerce Analysis Dashboard")
st.markdown("### By Ahmad Zaki")

# Create tabs
tab1, tab2 = st.tabs(["Product Performance", "Delivery Analysis"])

with tab1:
    st.header("Product Category Performance Analysis")
    
    # Prepare data for product analysis
    @st.cache_data
    def get_category_performance():
        order_details = orders_df.merge(order_items_df, on='order_id')
        order_details_with_products = order_details.merge(products_df, on='product_id')
        
        order_reviews_with_product = order_reviews_df.merge(order_items_df, on="order_id")
        order_reviews_with_product = order_reviews_with_product.merge(products_df, on="product_id")
        
        category_sales = order_details_with_products.groupby('product_category_name')['order_id'].count().reset_index()
        category_sales = category_sales.rename(columns={"order_id": "order_count"})
        
        category_reviews = order_reviews_with_product.groupby('product_category_name')['review_score'].mean().reset_index()
        
        return pd.merge(category_sales, category_reviews, on="product_category_name")
    
    category_performance = get_category_performance()
    
    # Add filters
    st.sidebar.header("Filters")
    min_orders = st.sidebar.slider("Minimum Orders", 
                                 min_value=int(category_performance['order_count'].min()),
                                 max_value=int(category_performance['order_count'].max()),
                                 value=int(category_performance['order_count'].min()))
    
    filtered_categories = category_performance[category_performance['order_count'] >= min_orders]
    
    # Create two columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Orders by Category")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.barplot(data=filtered_categories.head(10), x="order_count", y="product_category_name")
        plt.title("Top 10 Categories by Number of Orders")
        plt.xlabel("Number of Orders")
        plt.ylabel("Category")
        st.pyplot(fig1)
    
    with col2:
        st.subheader("Average Rating by Category")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.barplot(data=filtered_categories.head(10), x="review_score", y="product_category_name")
        plt.title("Top 10 Categories by Average Rating")
        plt.xlabel("Average Rating")
        plt.ylabel("Category")
        st.pyplot(fig2)

with tab2:
    st.header("Delivery Time Analysis by Seller Location")
    
    # Prepare data for delivery analysis
    @st.cache_data
    def get_delivery_analysis():
        orders_df['delivery_time'] = (orders_df['order_delivered_customer_date'] - 
                                    orders_df['order_purchase_timestamp']).dt.days
        
        order_details = orders_df.merge(order_items_df, on='order_id')
        order_seller = order_details.merge(sellers_df, on='seller_id')
        
        return order_seller.groupby('seller_state')['delivery_time'].agg(['mean', 'count']).reset_index()
    
    delivery_analysis = get_delivery_analysis()
    
    # Add filters
    min_orders_delivery = st.sidebar.slider("Minimum Orders per State",
                                          min_value=int(delivery_analysis['count'].min()),
                                          max_value=int(delivery_analysis['count'].max()),
                                          value=int(delivery_analysis['count'].min()))
    
    filtered_delivery = delivery_analysis[delivery_analysis['count'] >= min_orders_delivery]
    
    # Create delivery time visualization
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    sns.barplot(data=filtered_delivery, x="seller_state", y="mean")
    plt.title("Average Delivery Time by Seller State")
    plt.xlabel("Seller State")
    plt.ylabel("Average Delivery Time (Days)")
    plt.xticks(rotation=45)
    st.pyplot(fig3)
    
    # Add metrics
    col3, col4, col5 = st.columns(3)
    with col3:
        st.metric("Average Delivery Time", f"{delivery_analysis['mean'].mean():.1f} days")
    with col4:
        st.metric("Fastest State", 
                 f"{delivery_analysis.loc[delivery_analysis['mean'].idxmin(), 'seller_state']}")
    with col5:
        st.metric("Slowest State", 
                 f"{delivery_analysis.loc[delivery_analysis['mean'].idxmax(), 'seller_state']}")

# Footer
st.markdown("---")
st.markdown("Dashboard created as part of Dicoding Data Analysis Project")