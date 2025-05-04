# **SARS-CoV-2 Main Protease Inhibitor Discovery Framework**

## **1. Project Overview**

Since the emergence of the COVID-19 pandemic, SARS-CoV-2 variants with increased transmissibility and immune evasion have driven successive waves of infection. As of 2025, Omicron subvariants dominate globally, with region-specific prevalence patterns. At the molecular level, viral replication depends on the main protease (**M<sub>pro</sub>**), a homodimeric cysteine protease comprising three distinct domains. The catalytic triad — **His41, Cys145, and Asp187** — plays a central role in the proteolytic process.

Although over **55,000 compounds** have been tested against M<sub>pro</sub>, clinical translation remains limited due to pharmacokinetic (**PK**) challenges. Agents like **Boceprevir** showed promising binding affinity but poor PK profiles, necessitating combination therapies such as **Nirmatrelvir** and **Ritonavir** (*Paxlovid*) to enhance bioavailability.

This project investigates the interplay between **pharmacodynamic (PD)** and **PK** properties of M<sub>pro</sub> inhibitors through an **integrated computational pipeline**. We employed supervised **machine learning (ML)** algorithms and **molecular dynamics (MD)** simulations to develop predictive models and uncover key molecular features driving efficacy and drug-like behavior.

Classification models — including **Gaussian Naïve Bayes**, **Support Vector Machines**, **Decision Trees**, **Logistic regression** and **Multilayer Perceptrons** — were trained on IC<sub>50</sub> data from diverse biochemical assays. These models achieved good accuracies on chemically diverse datasets. High-affinity inhibitors tended to exhibit **hydrophilic features** favoring M<sub>pro</sub> binding but with suboptimal PK profiles. Conversely, **hydrophobic moieties** played a critical role in stabilizing interactions within the **S2 subsite**, as confirmed by MD simulations.

Next, we evaluated all models by ROC (Receiver Operating Characteristic) curve in order to verify performance of these models. Only Logistic regression and Support Vector Machine presented satisfactory results.

Our results emphasize the importance of **balancing PD and PK properties** during lead optimization and offer a computational framework for **rational design of SARS-CoV-2 M<sub>pro</sub> inhibitors**.

---

## **2. Objectives**

### **2.1 General Objective**

To develop a **scalable computational framework** that integrates **ligand-based** and **structure-based drug design** approaches for the discovery and optimization of potent SARS-CoV-2 M<sub>pro</sub> inhibitors. This includes:

* Data preprocessing
* Machine learning
* Molecular docking
* Molecular dynamics simulations

This pipeline supports **high-throughput virtual screening** of large chemical libraries (e.g., **ZINC database**).

---

### **2.2 Specific Objectives**

#### **1. Dataset Construction and Preprocessing**

* Curate and clean a comprehensive dataset of M<sub>pro</sub> inhibitors
* Convert IC<sub>50</sub> values to pIC<sub>50</sub> and refine binding constants (K<sub>i</sub>, K<sub>d</sub>, K<sub>m</sub>)
* Develop methods to manage missing, censored, or ambiguous inhibition data

#### **2. Computational Feature Engineering**

* Calculate 2D and 3D molecular descriptors using parallelized algorithms
* Apply dimensionality reduction, correlation filtering, and imputation to enhance model robustness

#### **3. Machine Learning Model Development**

* Train multiple ML models using k-fold cross-validation and hyperparameter tuning
* Retain high-performance models based on accuracy for downstream integration

#### **4. Data Visualization and Dimensionality Reduction**

* Use Principal Component Analysis (PCA) to reduce descriptor dimensionality
* Explore chemical-biological relationships using histograms, boxplots, and descriptive statistics
* Apply statistical inference to build confidence intervals

#### **5. Molecular Docking and Dynamics Simulations**

* Predict binding poses and affinity of inhibitors via molecular docking
* Perform **150-ns MD simulations** to assess binding free energy and protein–ligand stability

#### **6. Statistical and Chemoinformatics Analysis**

* Interpret decision rules from ML models to identify key molecular features
* Compute confidence intervals for statistically relevant inhibitor subgroups

#### **7. Integrated Framework Development**

* Combine **ligand-based (ML)** and **structure-based (docking/MD)** approaches
* Validate predictions and guide antiviral compound optimization

