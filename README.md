\documentclass[12pt]{article}
\usepackage[a4paper, margin=1in]{geometry}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{enumitem}

\title{SARS-CoV-2 Main Protease Inhibitor Discovery Framework}
\author{}
\date{}

\begin{document}

\maketitle

\section*{1. Project Overview}

Since the emergence of the COVID-19 pandemic, SARS-CoV-2 variants with increased transmissibility and immune evasion have driven successive waves of infection. As of 2025, Omicron subvariants dominate globally, with region-specific prevalence patterns. At the molecular level, viral replication depends on the main protease (M\textsubscript{pro}), a homodimeric cysteine protease comprising three distinct domains. The catalytic triad---His41, Cys145, and Asp187---plays a central role in the proteolytic process.

Although over 55{,}000 compounds have been tested against M\textsubscript{pro}, clinical translation remains limited due to pharmacokinetic (PK) challenges. Agents like Boceprevir showed promising binding affinity but poor PK profiles, necessitating combination therapies such as Nirmatrelvir and Ritonavir (Paxlovid) to enhance bioavailability.

This project investigates the interplay between pharmacodynamic (PD) and PK properties of M\textsubscript{pro} inhibitors through an integrated computational pipeline. We employed supervised machine learning (ML) algorithms and molecular dynamics (MD) simulations to develop predictive models and uncover key molecular features driving efficacy and drug-like behavior.

Classification models---including Gaussian Naïve Bayes, Support Vector Machines, Decision Trees, and Multilayer Perceptrons---were trained on IC\textsubscript{50} data from diverse biochemical assays. These models achieved accuracies ranging from 0.72 to 0.96, with external validation scores between 0.75 and 0.83 on chemically diverse datasets. High-affinity inhibitors tended to exhibit hydrophilic features favoring M\textsubscript{pro} binding but with suboptimal PK profiles. Conversely, hydrophobic moieties played a critical role in stabilizing interactions within the S2 subsite, as confirmed by MD simulations.

Our results emphasize the importance of balancing PD and PK properties during lead optimization and offer a computational framework for rational design of SARS-CoV-2 M\textsubscript{pro} inhibitors.

\section*{2. Objectives}

\subsection*{2.1 General Objective}

To develop a scalable computational framework that integrates ligand-based and structure-based drug design approaches for the discovery and optimization of potent SARS-CoV-2 M\textsubscript{pro} inhibitors. This includes data preprocessing, machine learning, molecular docking, and molecular dynamics simulations, supporting high-throughput virtual screening of large chemical libraries (e.g., ZINC database).

\subsection*{2.2 Specific Objectives}

\begin{enumerate}[label=\textbf{\arabic*.}]
    \item \textbf{Dataset Construction and Preprocessing}
    \begin{itemize}
        \item Curate and clean a comprehensive dataset of M\textsubscript{pro} inhibitors.
        \item Convert IC\textsubscript{50} values to pIC\textsubscript{50} and refine binding constants (K\textsubscript{i}, K\textsubscript{d}, K\textsubscript{m}) based on assay conditions.
        \item Develop methods to manage missing, censored, or ambiguous inhibition data.
    \end{itemize}

    \item \textbf{Computational Feature Engineering}
    \begin{itemize}
        \item Calculate 2D and 3D molecular descriptors using parallelized algorithms.
        \item Apply dimensionality reduction, correlation filtering, and imputation to enhance model robustness.
    \end{itemize}

    \item \textbf{Machine Learning Model Development}
    \begin{itemize}
        \item Train multiple ML models using k-fold cross-validation and hyperparameter tuning.
        \item Retain high-performance models based on accuracy for downstream integration.
    \end{itemize}

    \item \textbf{Data Visualization and Dimensionality Reduction}
    \begin{itemize}
        \item Use Principal Component Analysis (PCA) to reduce descriptor dimensionality and explore chemical-biological relationships.
        \item Employ statistical inference and graphical tools (e.g., histograms, boxplots) for exploratory data analysis.
    \end{itemize}

    \item \textbf{Molecular Docking and Dynamics Simulations}
    \begin{itemize}
        \item Predict binding poses and affinity of inhibitors via docking.
        \item Perform 150-ns MD simulations to assess binding free energy and protein–ligand stability.
    \end{itemize}

    \item \textbf{Statistical and Chemoinformatics Analysis}
    \begin{itemize}
        \item Interpret decision rules from ML models to identify key molecular features.
        \item Compute confidence intervals for relevant inhibitor subgroups.
    \end{itemize}

    \item \textbf{Integrated Framework Development}
    \begin{itemize}
        \item Combine ligand-based (ML) and structure-based (docking/MD) approaches to validate predictions and guide antiviral compound optimization.
    \end{itemize}
\end{enumerate}

\end{document}
