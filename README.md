# Loan-Approval-Prediction
#  Loan Approval Prediction with Machine Learning

This project uses supervised machine learning to predict whether a loan application will be approved based on applicant details. It includes data preprocessing, model evaluation, hyperparameter tuning, and performance analysis.

##  Project Structure

- `Loan_approval_model.ipynb`: Full Jupyter Notebook with code, comments, visualizations, and model evaluation.
- `README.md`: Project overview and usage guide.

##  Dataset

The dataset contains information about loan applicants, including:
- Loan	A unique id 
-	Gender	Gender of the applicant Male/female
-	Married	Marital Status of the applicant, values will be Yes/ No
-	Dependents	It tells whether the applicant has any dependents or not.
-	Education	It will tell us whether the applicant is Graduated or not.
-	Self_Employed	This defines that the applicant is self-employed i.e. Yes/ No
-	ApplicantIncome	Applicant income
-	CoapplicantIncome	Co-applicant income
-	LoanAmount	Loan amount (in thousands)
-	Loan_Amount_Term	Terms of loan (in months)
-	Credit_History	Credit history of individual's repayment of their debts
-	Property_Area	Area of property i.e. Rural/Urban/Semi-urban 
-	Loan_Status	Status of Loan Approved or not i.e. Y- Yes, N-No 

##  Preprocessing Steps

- Handled missing values using SimpleImpuyer.
- Converted categorical features using Label Encoding.
- Applied **SMOTE** to address class imbalance before training models.

##  Models Evaluated

Baseline classifiers were trained and compared:
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

###  Best Model: Random Forest

The Random Forest model achieved the best performance on the test set:
- **Accuracy:** 74.00%
- **Precision:** 77.78%
- **Recall:** 87.50%
- **F1 Score:** 82.35%
- **ROC AUC:** 75.04%

###  Tuned Model Performance

After tuning , the model demonstrated improved classification performance—especially for approved applications:

- **Accuracy:** 77%
- **Precision (Approved class):** 79%
- **Recall (Approved class):** 92%
- **F1 Score (Approved class):** 85%
- **ROC AUC:** 74.44%

Despite class imbalance being addressed with SMOTE, recall for the denied class remains lower (43%), indicating potential overlap in feature space or insufficient signal for that class.

This model prioritizes minimizing false negatives in loan approvals, aligning with risk-averse financial decision-making strategies.

##  Next Steps

- Threshold tuning to balance false positives/negatives
- Feature engineering and domain-specific ratios
- Try advanced models like XGBoost, LightGBM

##  Requirements

```bash
python>=3.7
pandas
numpy
matplotlib
seaborn
scikit-learn
imblearn
```

##  How to Use

1. Clone the repository
2. Install dependencies using `pip install -r requirements.txt`
3. Run the notebook: `loan_approval_model.ipynb`

## ✍ Author
Nana Kwame Ofori-Boakye
