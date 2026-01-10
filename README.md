# Crime Hotspot Prediction in Ottawa (2016–2022)

## Project Overview
This project analyzes and predicts crime hotspots in Ottawa using machine learning techniques.  
The dataset contains all founded Criminal Code of Canada offences reported to the Ottawa Police Service from **2016 to 2022**.  
For privacy protection, all crime locations are geomasked to the nearest intersection.

This is an **academic group project** completed as part of an Machine learning course (Level 1).


## My Contribution
- Exploratory Data Analysis (EDA) and data visualization  
- Feature engineering (date/time extraction, encoding, scaling)  
- Model building, tuning, and evaluation  
- Performance comparison across multiple train-test splits  
- Result interpretation and conclusion writing  

## Data Files
- `Criminal_Offences.csv`: Original dataset from Ottawa Police Service
- `Criminal_Offences_final_cleaned.csv`: Preprocessed dataset used for modeling


## 📊 Dataset Information
**Source:** Ottawa Police Service  
**Years Covered:** 2016–2022  
**Date Created:** June 1, 2023  

### Dataset Description
The dataset includes all founded criminal offences categorized using the **Uniform Crime Reporting (UCR) Survey**.

**Important Notes:**
- Data may change due to ongoing investigations and quality checks

## 🧾 Dataset Attributes
1. ID  
2. Year  
3. Reported Date & Time  
4. Occurrence Date & Time  
5. Weekday  
6. Criminal Offence Category (Target Variable)  
7. Primary Violation  
8. Neighbourhood  
9. Police Sector  
10. Police Division  
11. Census Tract  

## Exploratory Data Analysis
EDA was performed to understand:
- Crime distribution across divisions and neighborhoods  
- Crime frequency by weekday and hour  
- Emerging crime trends over time  
- Geographic consistency within the Ottawa region  

Multiple visualizations (bar charts, stacked plots, violin plots) were used to identify trends and hotspots.


## Feature Engineering
Key preprocessing steps included:
- Splitting date and time columns into meaningful components  
- Encoding categorical variables using Label Encoding  
- Removing irrelevant or noisy features  
- Normalizing numerical variables using Min-Max Scaling  
- Handling outliers and invalid values  

The cleaned dataset was saved as: Criminal_Offences_final_cleaned.csv


---

## Machine Learning Models
The following models were implemented and evaluated:

- **Decision Tree Classifier**
- **Random Forest Classifier**
- **Neural Network (MLP Classifier)**

### Evaluation Strategy
- Train-test splits: **80:20, 70:30, 85:15**
- Stratified sampling to handle class imbalance
- Hyperparameter tuning
- K-Fold Cross-Validation

---

## Results Summary
### Best Model
- **Decision Tree** achieved the highest accuracy (above **99.9%**) across all splits
- Random Forest performed consistently well but slightly lower than Decision Trees
- Neural Networks showed good performance but were more sensitive to tuning

### Most Important Features
- Primary Violation  
- Longitude (X) and Latitude (Y)  
- Police Sector  
- Police Division  
- Time of Crime Occurrence  

## 📈 Key Takeaways
- Tree-based models were most effective for crime prediction
- Temporal and geographic features played a major role
- Feature engineering significantly improved model performance

---

##  How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/crime-hotspot-prediction-ottawa.git

2. Install dependencies:

   pip install numpy pandas matplotlib seaborn scikit-learn

3. Open the Jupyter Notebook:

   jupyter notebook

This project is for educational purposes only.
The predictions should not be used for real-world law enforcement or policy decisions.

