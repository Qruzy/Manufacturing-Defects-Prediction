import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from submodule.load_data import load_data
from submodule.train_model import train_models  # Modified to match updated train_models
from submodule.plot_graph import plot_graph

def main():
    st.title("Manufacturing Defects Prediction Dashboard")
    
    # Load Data
    data = load_data()    
    page = st.sidebar.selectbox("Select a Page:", ["Homepage", "Exploration", "Modeling"])

    # Homepage Section
    if page == "Homepage":
        st.header("Homepage")
        st.write("### Manufacturing Defects Prediction Dataset Overview")
        st.dataframe(data)
    
    # Exploration Section
    elif page == "Exploration":
        st.header("Exploratory Data Analysis")
        fig1, fig2, fig3 = plot_graph(data)
        
        st.write("### Dataset Summary")
        st.pyplot(fig1)
        
        st.write("### Correlation Matrix")
        st.pyplot(fig2)
        
        st.write("### Feature Importances (Random Forest)")
        st.pyplot(fig3)

    # Modeling Section
    else:
        st.header("Modeling")
        
        # Declare models
        models = [
            RandomForestClassifier(), ExtraTreesClassifier(), AdaBoostClassifier(),
            GradientBoostingClassifier(), LogisticRegression(), SVC(),
            XGBClassifier(), LGBMClassifier()
        ]
        
        model_names = [
            "Random Forest", "Extra Trees", "Ada Boost", "Gradient Boosting",
            "Logistic Regression", "Support Vector Machine", "XGBoost", "LightGBM"
        ]

        # Train and evaluate models
        scores, reports, cms = train_models(models, model_names, data)
        
        # Display model accuracy scores
        scores_df = pd.DataFrame({"Model": model_names, "Accuracy (%)": scores}).sort_values(by="Accuracy (%)", ascending=False)
        scores_df["Accuracy (%)"] = scores_df["Accuracy (%)"].round(2)
        
        st.write("### Model Accuracy Comparison")
        st.table(scores_df)
        
        # Plot model accuracies
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=scores_df, x="Model", y="Accuracy (%)", ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        for container in ax.containers:
            ax.bar_label(container)
        st.pyplot(fig)

if __name__ == '__main__':
    main()
