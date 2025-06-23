import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# App configuration
st.set_page_config(page_title="Student Clustering App", layout="wide")
st.title("🎓 Student Academic Clustering App")

# Sidebar always visible settings
st.sidebar.header("⚙️ Clustering Settings")
n_clusters = st.sidebar.slider("Select number of clusters", min_value=1, max_value=5, value=3, step=1)

# File upload
uploaded_file = st.file_uploader("📁 Upload your student CSV file", type=["csv"])

if uploaded_file:
    # Read and preview data
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    # Auto-select numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if not numeric_cols:
        st.warning("⚠️ No numeric columns found in the uploaded file.")
    else:
        # Feature selection
        selected_features = st.sidebar.multiselect("Select features for clustering", numeric_cols, default=numeric_cols)

        # Preprocess
        data_selected = df[selected_features].fillna(0)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_selected)

        # Apply KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['Cluster'] = kmeans.fit_predict(scaled_data)

        # Show clustered data
        st.subheader("📊 Clustered Data Preview")
        st.dataframe(df.head())

        # Download clustered data
        st.download_button("⬇️ Download Clustered Data", df.to_csv(index=False).encode('utf-8'), "clustered_student_data.csv", "text/csv")

        # 📈 Visualizations
        st.subheader("📈 Visualizations")

        # Bar chart - cluster sizes
        st.markdown("**Cluster Size Bar Chart**")
        cluster_counts = df['Cluster'].value_counts().sort_index()
        fig1, ax1 = plt.subplots()
        sns.barplot(x=cluster_counts.index, y=cluster_counts.values, ax=ax1)
        ax1.set_xlabel("Cluster")
        ax1.set_ylabel("Number of Students")
        st.pyplot(fig1)

        # Scatter Plot - if 2 or more features selected
        if len(selected_features) >= 2:
            st.markdown("**Scatter Plot of Two Features**")
            fig2, ax2 = plt.subplots()
            sns.scatterplot(x=df[selected_features[0]], y=df[selected_features[1]], hue=df['Cluster'], palette="viridis", ax=ax2)
            ax2.set_xlabel(selected_features[0])
            ax2.set_ylabel(selected_features[1])
            st.pyplot(fig2)

        # Pie Chart - select feature
        selected_pie_feature = st.selectbox("📊 Select feature for Pie Chart (sum by cluster)", selected_features)
        pie_data = df.groupby('Cluster')[selected_pie_feature].sum()
        fig3, ax3 = plt.subplots()
        ax3.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90)
        ax3.set_title(f"Distribution of {selected_pie_feature} by Cluster")
        st.pyplot(fig3)

        # Line Chart - compare two features
        if len(selected_features) >= 2:
            st.markdown("📈 Line Chart Comparing Two Metrics by Cluster")
            line_feature_x = st.selectbox("Select X-axis Feature", selected_features, index=0, key="line_x")
            line_feature_y = st.selectbox("Select Y-axis Feature", selected_features, index=1, key="line_y")
            line_data = df.groupby('Cluster')[[line_feature_x, line_feature_y]].mean().reset_index()
            fig4, ax4 = plt.subplots()
            for i in line_data['Cluster']:
                ax4.plot([line_feature_x, line_feature_y],
                         [line_data[line_data['Cluster'] == i][line_feature_x].values[0],
                          line_data[line_data['Cluster'] == i][line_feature_y].values[0]],
                         marker='o', label=f'Cluster {i}')
            ax4.set_ylabel("Average Value")
            ax4.set_title("Line Chart of Feature Averages")
            ax4.legend()
            st.pyplot(fig4)
else:
    st.info("👈 Please upload a student dataset to begin.")
