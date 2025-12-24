import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


@st.cache_data
def load_data() -> pd.DataFrame:
    """
    Load the 'tips' dataset (restaurant bills) hosted on GitHub and
    construct additional columns for analysis.
    """
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
    df = pd.read_csv(url)

    # Quantity  -> number of people at table
    # UnitPrice -> average spend per person (approx.)
    df["Quantity"] = df["size"]
    df["UnitPrice"] = df["total_bill"] / df["size"]

    # Create a pseudo-date column for a simple transactional time series
    n = len(df)
    df = df.sort_index().reset_index(drop=True)
    df["InvoiceDate"] = pd.date_range(start="2021-01-01", periods=n, freq="H")

    # Use 'day' column as a country-like categorical variable
    df["Country"] = df["day"]

    # Create Revenue
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    # Add a simple transaction ID
    df["InvoiceNo"] = np.arange(1, n + 1)

    return df


def show_basic_table(df: pd.DataFrame) -> None:
    st.subheader("1. First 10 rows (interactive table)")
    st.dataframe(df.head(10))


def show_structure_info(df: pd.DataFrame) -> None:
    st.subheader("2. Structural information")

    n_obs, n_vars = df.shape
    st.write(f"**Number of observations:** {n_obs}")
    st.write(f"**Number of variables:** {n_vars}")

    st.write("**Data types:**")
    st.write(pd.DataFrame(df.dtypes, columns=["dtype"]))


def show_categorical_pie(df: pd.DataFrame) -> None:
    st.subheader("3. Categorical distribution (pie chart)")

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    selected_cat = st.radio(
        "Select a categorical variable", categorical_cols, key="cat_pie", horizontal=True
    )

    counts = df[selected_cat].value_counts().reset_index()
    counts.columns = [selected_cat, "count"]

    st.write(f"Category distribution for **{selected_cat}**")
    st.plotly_chart(
        _pie_chart(counts, names_col=selected_cat, values_col="count"), use_container_width=True
    )


def _pie_chart(df: pd.DataFrame, names_col: str, values_col: str):
    import plotly.express as px

    fig = px.pie(df, names=names_col, values=values_col, hole=0.3)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return fig


def show_top_countries_bar(df: pd.DataFrame) -> None:
    st.subheader("4. Top 10 'countries' by number of transactions (bar chart)")

    country_counts = df["Country"].value_counts().head(10).reset_index()
    country_counts.columns = ["Country", "Transactions"]

    st.bar_chart(country_counts.set_index("Country"))


def show_quantity_price_scatter(df: pd.DataFrame) -> None:
    st.subheader("5. Quantity vs UnitPrice (scatter plot)")
    st.write("Relationship between number of people at a table and average spend per person.")
    st.scatter_chart(df[["Quantity", "UnitPrice"]], x="Quantity", y="UnitPrice")


def show_revenue_hist(df: pd.DataFrame) -> None:
    st.subheader("6. Revenue histogram")
    st.write("Revenue is defined as Quantity × UnitPrice.")

    bins = st.slider("Number of bins", 10, 100, 30, step=1)

    hist_values, bin_edges = np.histogram(df["Revenue"], bins=bins)
    # Format bin edges to 2 decimal places for better readability
    hist_df = pd.DataFrame({
        "Revenue Range": [f"{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}" for i in range(len(bin_edges)-1)],
        "Count": hist_values
    })
    st.bar_chart(hist_df.set_index("Revenue Range"))


def show_transactions_over_time(df: pd.DataFrame) -> None:
    st.subheader("7. Transactions over time (line chart)")

    tx_per_day = (
        df.set_index("InvoiceDate")
        .resample("D")
        .size()
        .reset_index(name="transactions")
    )
    st.line_chart(tx_per_day.set_index("InvoiceDate"))


def compute_pca(df: pd.DataFrame):
    numeric_vars = ["Quantity", "UnitPrice", "Revenue"]
    X = df[numeric_vars].dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(
        components, columns=["PC1", "PC2"], index=X.index
    )

    return pca_df, pca, X.index


def show_pca_scatter(df: pd.DataFrame) -> None:
    st.subheader("8. PCA (2 components) on numeric variables")
    
    st.markdown("""
    **PCA Variables Used:**
    - **Quantity**: Number of people at the table
    - **UnitPrice**: Average spend per person (total_bill / size)
    - **Revenue**: Quantity × UnitPrice
    
    These three numeric variables are standardized (mean=0, std=1) and then transformed 
    into 2 principal components that capture the most variance in the data.
    """)
    
    pca_df, pca, idx = compute_pca(df)
    
    var_explained = pca.explained_variance_ratio_
    st.write(f"**Explained variance ratio:** PC1: {var_explained[0]:.3f} ({var_explained[0]*100:.1f}%), "
             f"PC2: {var_explained[1]:.3f} ({var_explained[1]*100:.1f}%)")
    st.write(f"**Total variance explained:** {var_explained.sum():.3f} ({var_explained.sum()*100:.1f}%)")
    
    # Use Plotly for better visualization (avoids integer-like appearance issue)
    import plotly.express as px
    fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        title="PCA 2D Scatter Plot (Standardized Features)",
        labels={"PC1": f"PC1 ({var_explained[0]*100:.1f}% variance)",
                "PC2": f"PC2 ({var_explained[1]*100:.1f}% variance)"},
    )
    fig.update_traces(marker=dict(size=5, opacity=0.6))
    st.plotly_chart(fig, use_container_width=True)


def show_pca_colored_by_country(df: pd.DataFrame) -> None:
    st.subheader("9. PCA colored by 'Country' (top 5 only)")

    pca_df, pca, idx = compute_pca(df)
    pca_df = pca_df.copy()
    pca_df["Country"] = df.loc[idx, "Country"]

    top5 = pca_df["Country"].value_counts().head(5).index
    pca_top5 = pca_df[pca_df["Country"].isin(top5)]

    import plotly.express as px

    fig = px.scatter(
        pca_top5,
        x="PC1",
        y="PC2",
        color="Country",
        title="PCA (2D) colored by top 5 'countries'",
    )
    st.plotly_chart(fig, use_container_width=True)


def show_feature_selection(df: pd.DataFrame) -> None:
    st.subheader("10. Feature selection for Revenue")
    
    st.markdown("""
    **Method:** Random Forest Feature Importance  
    A Random Forest regression model is trained to predict Revenue using all numeric features 
    (excluding Revenue itself). The importance scores indicate how much each feature contributes 
    to predicting Revenue.
    """)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "Revenue" not in numeric_cols:
        st.error("Revenue column not found.")
        return

    feature_cols = [c for c in numeric_cols if c != "Revenue"]
    
    st.write(f"**Features analyzed:** {', '.join(feature_cols)}")

    X = df[feature_cols].fillna(0)
    y = df["Revenue"].fillna(0)

    rf = RandomForestRegressor(
        n_estimators=200, random_state=42, n_jobs=-1, oob_score=False
    )
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(
        ascending=False
    )
    
    # Create a formatted DataFrame for better display
    importance_df = pd.DataFrame({
        "Feature": importances.index,
        "Importance": importances.values
    })
    importance_df["Importance"] = importance_df["Importance"].round(4)

    st.write("**Feature importances (higher means more important for predicting Revenue):**")
    
    # Display as a table first for exact values
    st.dataframe(importance_df.set_index("Feature"), use_container_width=True)
    
    # Then show as bar chart
    st.bar_chart(importances)


def show_random_forest_model(df: pd.DataFrame) -> None:
    st.subheader("11. Random Forest model to predict Revenue")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "Revenue" not in numeric_cols:
        st.error("Revenue column not found.")
        return

    feature_cols = [c for c in numeric_cols if c != "Revenue"]

    X = df[feature_cols].fillna(0)
    y = df["Revenue"].fillna(0)

    test_size = st.slider("Test size (fraction)", 0.1, 0.5, 0.2)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    n_estimators = st.slider("Number of trees", 50, 500, 200, step=50)
    max_depth = st.slider("Max depth (0 = None)", 0, 20, 0)
    depth_param = None if max_depth == 0 else max_depth

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=depth_param,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    st.write(f"**RMSE:** {rmse:.4f}")
    st.write(f"**R²:** {r2:.4f}")


def show_dashboard(df: pd.DataFrame) -> None:
    st.subheader("12. Integrated dashboard")

    tab1, tab2, tab3 = st.tabs(
        ["Country-based chart", "Revenue histogram", "PCA scatter plot"]
    )

    with tab1:
        st.write("Top 10 'countries' by number of transactions")
        country_counts = df["Country"].value_counts().head(10).reset_index()
        country_counts.columns = ["Country", "Transactions"]
        st.bar_chart(country_counts.set_index("Country"))

    with tab2:
        st.write("Revenue distribution")
        bins = 40
        hist_values, bin_edges = np.histogram(df["Revenue"], bins=bins)
        # Format bin edges for better readability
        hist_df = pd.DataFrame({
            "Revenue Range": [f"{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}" for i in range(len(bin_edges)-1)],
            "Count": hist_values
        })
        st.bar_chart(hist_df.set_index("Revenue Range"))

    with tab3:
        st.write("PCA scatter plot (colored by top 5 'countries')")
        st.caption("PCA performed on Quantity, UnitPrice, and Revenue (standardized).")
        pca_df, pca, idx = compute_pca(df)
        pca_df = pca_df.copy()
        pca_df["Country"] = df.loc[idx, "Country"]
        top5 = pca_df["Country"].value_counts().head(5).index
        pca_top5 = pca_df[pca_df["Country"].isin(top5)]

        import plotly.express as px

        var_explained = pca.explained_variance_ratio_
        fig = px.scatter(
            pca_top5,
            x="PC1",
            y="PC2",
            color="Country",
            title=f"PCA (2D) colored by top 5 'countries' (Explained variance: {var_explained.sum()*100:.1f}%)",
            labels={"PC1": f"PC1 ({var_explained[0]*100:.1f}% variance)",
                    "PC2": f"PC2 ({var_explained[1]*100:.1f}% variance)"}
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Interpretation questions")

    st.markdown(
        """
**Which 'countries' generate the most revenue?**  
Use the country-based chart along with a group-by summary such as:

```python
revenue_by_country = df.groupby("Country")["Revenue"].sum().sort_values(ascending=False)
```

In this adapted dataset, "countries" correspond to days of the week (Thu, Fri, Sat, Sun).

**What does PCA reveal about transaction patterns?**  
PCA shows whether numeric behavior (Quantity, UnitPrice, Revenue) differs systematically
between these groups. Clusters or separation in the PCA scatter plot suggest distinct
spending patterns across days.

**How can this dashboard support managerial decision-making?**  
Managers can see which groups (days) drive the most revenue, understand spending intensity
via Revenue distribution, and detect segments with similar transaction patterns from PCA.
This helps in staffing, promotions, and pricing decisions.
"""
    )


def main() -> None:
    st.set_page_config(page_title="Tips Dataset Analytics", layout="wide")
    st.title("🍽️ Tips Dataset Analytics Dashboard")
    
    # Author information
    with st.sidebar:
        st.markdown("---")
        st.markdown("**Created by:** Abdullah Gökalp ÇIRAY")
        st.markdown("**Course:** YBSB3003 – Programming for Data Science")
        st.markdown("---")

    st.markdown(
        """
This app explores the classic **tips** dataset (restaurant bills) with a set of views:
tables, structure information, categorical charts, Revenue analysis, PCA, feature selection,
Random Forest modelling, and an integrated dashboard.
"""
    )

    df = load_data()

    pages = {
        "1 – First 10 rows": show_basic_table,
        "2 – Structure info": show_structure_info,
        "3 – Categorical pie chart": show_categorical_pie,
        "4 – Top 10 countries (bar)": show_top_countries_bar,
        "5 – Quantity vs UnitPrice": show_quantity_price_scatter,
        "6 – Revenue histogram": show_revenue_hist,
        "7 – Transactions over time": show_transactions_over_time,
        "8 – PCA (2D)": show_pca_scatter,
        "9 – PCA colored by Country": show_pca_colored_by_country,
        "10 – Feature selection for Revenue": show_feature_selection,
        "11 – Random Forest model": show_random_forest_model,
        "12 – Integrated dashboard": show_dashboard,
    }

    st.sidebar.markdown("## 📊 Navigation")
    choice = st.sidebar.radio("Select analysis page", list(pages.keys()), label_visibility="collapsed")
    pages[choice](df)


if __name__ == "__main__":
    main()


