import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# App configuration
st.set_page_config(page_title="Student Clustering App", layout="wide")
st.title("🎓 Student Academic Clustering App")

# File upload
uploaded_file = st.file_uploader("📁 Upload your student CSV file", type=["csv"])

if uploaded_file:
    # Read data
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    # Auto-select numeric columns only
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    st.sidebar.header("⚙️ Clustering Settings")

    if not numeric_cols:
        st.warning("No numeric columns found in the uploaded file.")
    else:
        # Choose number of clusters with range 1 to 5
        n_clusters = st.sidebar.slider("Select number of clusters", 1, 5, 3)

        # Feature selection (all numeric by default)
        selected_features = st.sidebar.multiselect("Select features for clustering", numeric_cols, default=numeric_cols)

        # Preprocessing
        data_selected = df[selected_features].fillna(0)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_selected)

        # KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['Cluster'] = kmeans.fit_predict(scaled_data)

        st.subheader("📊 Clustered Data Preview")
        st.dataframe(df.head())

        # Download button
        st.download_button("⬇️ Download Clustered Data", df.to_csv(index=False).encode('utf-8'), "clustered_student_data.csv", "text/csv")

        # 📈 Visualizations
        st.subheader("📈 Visualizations")

        # Bar Chart - Cluster sizes
        st.markdown("**Cluster Size Bar Chart**")
        cluster_counts = df['Cluster'].value_counts().sort_index()
        fig1, ax1 = plt.subplots()
        sns.barplot(x=cluster_counts.index, y=cluster_counts.values, ax=ax1)
        ax1.set_xlabel("Cluster")
        ax1.set_ylabel("Number of Students")
        st.pyplot(fig1)

        # Scatter Plot - First 2 selected features
        if len(selected_features) >= 2:
            st.markdown("**Scatter Plot of Two Features**")
            fig2, ax2 = plt.subplots()
            sns.scatterplot(x=df[selected_features[0]], y=df[selected_features[1]], hue=df['Cluster'], palette="viridis", ax=ax2)
            st.pyplot(fig2)

        # Pie Chart - Choose feature to visualize
        selected_pie_feature = st.selectbox("Select feature for Pie Chart (sum by cluster)", selected_features)
        pie_data = df.groupby('Cluster')[selected_pie_feature].sum()
        fig3, ax3 = plt.subplots()
        ax3.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90)
        ax3.set_title(f"Distribution of {selected_pie_feature} by Cluster")
        st.pyplot(fig3)

        # Line Chart - Compare two features
        if len(selected_features) >= 2:
            st.markdown("**Line Chart Comparing Two Metrics by Cluster**")
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
