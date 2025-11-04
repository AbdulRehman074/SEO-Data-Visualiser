# seo_data_visualiser_app.py
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import io

# =========================
# App Title & Description
# =========================
st.set_page_config(page_title="SEO Data Visualiser", layout="wide")
st.title("📊 SEO Data Visualiser & Report Generator")
st.write("Upload your SEO Excel file to analyse keyword performance and download a professional report.")

# =========================
# File Upload
# =========================
uploaded_file = st.file_uploader("📤 Upload Excel File", type=["xlsx"])

if uploaded_file is not None:
    # Load Data
    df = pd.read_excel(uploaded_file)
    st.success("✅ File uploaded successfully!")
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    # =========================
    # Basic Cleaning
    # =========================
    df.dropna(subset=['Keyword', 'Clicks', 'Impressions'], inplace=True)
    df['CTR (%)'] = df['Clicks'] / df['Impressions'] * 100
    df['Opportunity Score'] = df['Impressions'] * (1 - df['CTR (%)'] / 100)

    # =========================
    # Keyword Clustering
    # =========================
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['Keyword'])
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X)

    st.subheader("🧩 Keyword Clusters")
    for cluster_num in range(4):
        keywords = ", ".join(df[df['Cluster'] == cluster_num]['Keyword'].tolist())
        st.markdown(f"**Cluster {cluster_num+1}:** {keywords}")

    # =========================
    # Visualisations
    # =========================
    st.subheader("📈 Top Keywords by Clicks")
    top_keywords = df.sort_values('Clicks', ascending=False).head(10)
    st.bar_chart(data=top_keywords, x='Keyword', y='Clicks')

    st.subheader("🎯 CTR vs Impressions")
    st.scatter_chart(data=df, x='Impressions', y='CTR (%)')

    # =========================
    # Generate PDF Report
    # =========================
    st.subheader("📄 Generate & Download Report")
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        # 1️⃣ Title Page
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.5, 0.6, "SEO Data Visualiser Report", ha='center', fontsize=20)
        fig.text(0.5, 0.5, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                 ha='center', fontsize=10)
        pdf.savefig(fig)
        plt.close(fig)

        # 2️⃣ Top Keywords by Clicks
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(top_keywords['Keyword'], top_keywords['Clicks'], color='skyblue')
        ax.set_title('Top Keywords by Clicks')
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # 3️⃣ CTR vs Impressions
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(df['Impressions'], df['CTR (%)'], c='green', alpha=0.6)
        ax.set_title('CTR vs Impressions')
        ax.set_xlabel('Impressions')
        ax.set_ylabel('CTR (%)')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    st.success("✅ PDF report generated successfully!")
    st.download_button(
        label="📥 Download PDF Report",
        data=buffer.getvalue(),
        file_name="seo_data_visualiser_report.pdf",
        mime="application/pdf"
    )

else:
    st.info("👆 Please upload an Excel file to start your SEO analysis.")
