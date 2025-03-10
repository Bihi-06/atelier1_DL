# atelier1_DL
# PyTorch Classification and Regression Lab

## **Objective**
This lab provides hands-on experience with the PyTorch library for tackling **Regression** and **Multi-Class Classification** tasks using **Deep Neural Networks (DNNs)** and **Multi-Layer Perceptrons (MLPs)**. The primary goal is to explore dataset characteristics, develop models, tune hyperparameters, and apply regularization techniques to improve performance.

---

## **Work to Do**

### **Part 1: Regression**

**Dataset:** NYSE Stock Market Data

#### **1. Exploratory Data Analysis (EDA)**
- Perform thorough data exploration techniques to understand and visualize the dataset.
- Identify missing values, outliers, and analyze feature correlations.

#### **2. Model Development**
- Design a **Deep Neural Network (DNN)** using PyTorch to perform **regression**.

#### **3. Hyperparameter Tuning**
- Utilize **GridSearchCV** from scikit-learn to optimize hyperparameters such as:
  - Learning rate
  - Optimizers
  - Number of epochs
  - Model architecture

#### **4. Model Performance Visualization**
- Plot **Loss vs. Epochs** and **Accuracy vs. Epochs** graphs for both training and test datasets.
- Interpret and discuss observed trends in the model's learning process.

#### **5. Regularization Techniques**
- Implement **Dropout, Weight Decay, and Batch Normalization** to prevent overfitting.
- Compare the performance of the **regularized model** with the initial model.

---

### **Part 2: Multi-Class Classification**

**Dataset:** Predictive Maintenance Classification Dataset

#### **1. Preprocessing**
- Apply preprocessing techniques to clean, transform, and standardize/normalize the dataset.

#### **2. Exploratory Data Analysis (EDA)**
- Visualize and analyze dataset distributions, relationships, and patterns.

#### **3. Data Augmentation**
- Implement **data augmentation** methods to balance the dataset and enhance model generalization.

#### **4. Model Development**
- Design a **Deep Neural Network (DNN)** using PyTorch for **multi-class classification**.

#### **5. Hyperparameter Tuning**
- Use **GridSearchCV** from scikit-learn to determine the best hyperparameters for model training.

#### **6. Model Performance Visualization**
- Plot **Loss vs. Epochs** and **Accuracy vs. Epochs** graphs for both training and test datasets.
- Provide interpretations of the learning process.

#### **7. Evaluation Metrics**
- Compute and analyze the following metrics:
  - Accuracy
  - Sensitivity (Recall)
  - F1-score
  - Other relevant classification metrics

#### **8. Regularization Techniques**
- Implement **Dropout, L2 Regularization, and Batch Normalization**.
- Compare results with the baseline model to measure improvement.

---

## **Tools & Libraries**
- **Python**
- **PyTorch**
- **scikit-learn**
- **pandas**
- **numpy**
- **matplotlib / seaborn** (for visualization)
- **GridSearchCV** (for hyperparameter tuning)

---

## **How to Run**

1. Install the required dependencies:
   ```bash
   pip install torch torchvision pandas numpy matplotlib seaborn scikit-learn
   ```
2. Download the datasets from Kaggle.
3. Follow the structured steps for both **regression** and **classification** tasks.
4. Optimize hyperparameters and apply regularization techniques.
5. Evaluate and visualize the model performance.

---

## **Expected Outcomes**
- A well-trained **regression model** with optimized hyperparameters.
- A **multi-class classification model** with balanced dataset handling and strong evaluation metrics.
- A deeper understanding of **regularization techniques** and their impact on model performance.
- Effective visualizations of model training and learning trends.

By completing this lab, you will gain practical experience in **building, tuning, and improving deep learning models with PyTorch**. 🚀

