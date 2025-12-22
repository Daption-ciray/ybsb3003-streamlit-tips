## Programming for Data Science – Tips Dataset Dashboard

This project builds an interactive Streamlit dashboard on top of the classic **tips** dataset
(restaurant bills). It demonstrates common data science steps such as exploratory analysis,
visualization, PCA, feature selection and Random Forest modelling within a single web app.

### Dataset

- **Source**: `https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv`
- **Original columns (examples)**: `total_bill`, `tip`, `sex`, `smoker`, `day`, `time`, `size`

The app additionally creates a few helper columns for analysis:

- `Quantity`  → `size` (number of people at the table)  
- `UnitPrice` → `total_bill / size` (approximate spend per person)  
- `InvoiceDate` → synthetic hourly timestamps  
- `Country` → `day` (different days treated as different groups/segments)  
- `Revenue` → `Quantity * UnitPrice`  
- `InvoiceNo` → simple running transaction id  

### Implemented Tasks (1–12)

All tasks are implemented in a **single Streamlit app** (`app.py`) with a sidebar menu:

1. First 10 rows – interactive table  
2. Structural information (number of observations/variables, data types)  
3. Categorical variable pie chart (sidebar dropdown)  
4. Top 10 "countries" (here: days) by number of transactions – bar chart  
5. Quantity vs UnitPrice – scatter plot  
6. Revenue histogram (`Revenue = Quantity × UnitPrice`)  
7. Transactions over time (using synthetic `InvoiceDate`) – line chart  
8. Standardization + PCA (2 components) – 2D scatter  
9. PCA colored by "Country" (top 5 groups only)  
10. Feature selection for Revenue (Random Forest feature importances on numeric vars)  
11. Random Forest regression to predict Revenue (RMSE and R² reported)  
12. Integrated dashboard:  
   - Country-based bar chart  
   - Revenue histogram  
   - PCA scatter plot (colored by "Country")  
   - Text discussion for:
     - Which groups generate the most revenue  
     - What PCA reveals about transaction patterns  
     - How the dashboard supports managerial decision-making  

### How to Run (with `.venv311`)

Assuming you want to use the existing `.venv311` virtual environment:

```bash
cd "/Users/daption-ciray/Desktop/MIS dersler/Elif Kartal Data Science/streamlit"

# (Only once) create and activate the venv if needed
python3.11 -m venv .venv311
source .venv311/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

### Git & GitHub – Tek Repo ile Kullanım

Bu projeyi tek bir GitHub reposuna göndermek için komut satırından şu adımları izleyebilirsin:

1. **Git’i başlat ve ilk commit’i yap**
   ```bash
   cd "/Users/daption-ciray/Desktop/MIS dersler/Elif Kartal Data Science/streamlit"
   git init
   git add app.py requirements.txt README.md
   git commit -m "Initial commit: Streamlit tips exercise"
   ```

2. **GitHub’da yeni repo oluştur**
   - GitHub → *New repository*  
   - Örneğin isim: **ybsb3003-streamlit-tips**  
   - README ekleme (bu projede zaten README var)

3. **Remote ekle ve push et**
   ```bash
   git remote add origin https://github.com/<kullanici-adin>/ybsb3003-streamlit-tips.git
   git branch -M main
   git push -u origin main
   ```

Bu adımlardan sonra bütün proje tek bir GitHub reposu içinde yer almış olur.


