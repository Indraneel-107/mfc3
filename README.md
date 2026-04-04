<p align="center">
  <img src="amrita-red-logo.svg" width="150"/>
</p>

# MFC3: Enhancing Summarization Efficiency  
## Compressive Text Summarization using Sparse Optimization + T5

---

### 👥 Group D1 — Computationally Efficient Text Summarization Tool

> *School of Engineering, Amrita Vishwa Vidyapeetham*

---

## 👥 Team Members

| Name | Roll Number |
|:---:|:---:|
| Indraneel R | CB.SC.U4AIE24323 |
| Rishi Dasari | CB.SC.U4AIE24365 |
| Abhishek Reddy | CB.SC.U4AIE24325 |
| M Aryan | CB.SC.U4AIE24341 |

---

# 🎯 Primary Objective

The objective of this project is to design a **computationally efficient, mathematically grounded text summarization system** that:

- Minimizes redundancy  
- Preserves important semantic content  
- Produces fluent summaries  

This is achieved through a **hybrid framework** combining:

1. **Sparse Optimization (Convex formulation)**
2. **ADMM-based efficient solver**
3. **Transformer-based abstractive refinement (T5)**

---

# 🖥️ Platform

| Component | Details |
|----------|--------|
| Language | Python |
| Framework | PyTorch + HuggingFace |
| Model | T5-base |
| Dataset | BBC News Summary |
| Optimization | ADMM |
| Environment | Colab / Local GPU |

---

# ⚙️ SYSTEM PIPELINE

### Stage 1 — Document Representation

Given a document with \( n \) sentences and vocabulary size \( d \):

\[
D \in \mathbb{R}^{d \times n}
\]

Each column \( D_j \) represents sentence \( j \) using TF-IDF encoding.

---

# 🧠 MATHEMATICAL FORMULATION (CORE)

## 1. Data Reconstruction Principle

A good summary should reconstruct the document:

\[
D \approx D A
\]

where:

- \( A \in \mathbb{R}^{n \times n} \) → sentence selection matrix  

---

## 2. Reconstruction Loss

\[
\min_{A} \; \| D - DA \|_F^2
\]

---

## 3. Row Sparsity Constraint

To select only important sentences:

\[
\| A \|_{2,1} = \sum_{i=1}^{n} \| A_i \|_2
\]

This enforces **row sparsity** → selects key sentences.

---

## 4. Final Optimization Problem

\[
\min_{A} \; \| D - DA \|_F^2 + \lambda \| A \|_{2,1}
\]

Subject to:

\[
A_{ij} \geq 0 \quad \forall i,j
\]

\[
\text{diag}(A) = 0
\]

---

# ⚡ ADMM SOLVER (STEP-BY-STEP DERIVATION)

## Step 1: Variable Splitting

Introduce auxiliary variable \( Z \):

\[
X = Z
\]

Reformulated problem:

\[
\min_{X,Z} \; \| D - DX \|_F^2 + \lambda \| Z \|_{2,1}
\]

---

## Step 2: Augmented Lagrangian

\[
\mathcal{L}(X,Z,U) = \| D - DX \|_F^2 + \lambda \| Z \|_{2,1} + \frac{\rho}{2} \| X - Z + U \|_F^2
\]

---

## Step 3: X-Update (Derivation)

\[
X_j^{k+1} = \arg\min \; \| D_j - D X_j \|_2^2 + \frac{\rho}{2} \| X_j - Z_j + U_j \|_2^2
\]

Taking derivative:

\[
-2D^T(D_j - DX_j) + \rho(X_j - Z_j + U_j) = 0
\]

\[
( D^T D + \rho I ) X_j = D^T D_j + \rho (Z_j - U_j)
\]

Final update:

\[
X_j^{k+1} = ( D^T D + \rho I )^{-1} ( D^T D_j + \rho (Z_j - U_j) )
\]

---

## Step 4: Z-Update (Group Shrinkage)

\[
Z_i^{k+1} = \arg\min \; \lambda \| Z_i \|_2 + \frac{\rho}{2} \| Z_i - (X_i + U_i) \|_2^2
\]

Solution:

\[
Z_i^{k+1} = \max\left(1 - \frac{\lambda}{\rho \|x\|_2}, 0 \right) x
\]

where:

\[
x = X_i + U_i
\]

---

## Step 5: U-Update

\[
U^{k+1} = U^k + \rho (X^{k+1} - Z^{k+1})
\]

---

# 🔄 DIVERSITY ENHANCEMENT

## Dissimilarity Matrix

\[
\Delta = [\delta_{ij}]
\]

---

## Modified Objective

\[
\min_{X,Z} \; \| D - DX \|_F^2 + \mu \, \text{tr}(\Delta^T X) + \lambda \| Z \|_{2,1}
\]

---

## Updated X-Step

\[
X_j^{k+1} = (D^T D + \rho I)^{-1} ( D^T D_j + \rho(Z_j - U_j) - \mu (\Delta^T)_j )
\]

---

# 🔬 COMPRESSIVE SUMMARIZATION

## Joint Optimization

\[
\min_{R,A} \; \| D - RA \|_F^2 + \lambda_1 \| A \|_{2,1} + \lambda_2 \sum_{i=1}^{n} \| R_i \|_1
\]

---

## Solution Strategy

- Fix \( R \) → solve for \( A \) (ADMM)  
- Fix \( A \) → solve for \( R \) (LASSO)  

---

# 🤖 TRANSFORMER INTEGRATION (T5)

Final stage uses:

- Model: `t5-base`
- Input:  Text or Paragraph
- Output: Fluent abstractive summary

---

## Generation Parameters

- Max Length: 150  
- Min Length: 40  
- Beam Size: 4  
- Length Penalty: 2.0  

---

# 📊 DATASET

- BBC News Summary Dataset  
- 2225 articles  
- Multi-domain  

---

# 📈 RESULTS

- Reduced redundancy  
- Improved semantic coverage  
- Fluent summaries via T5  
- Efficient convergence using ADMM  

---

# 🚀 HOW TO RUN

```bash
pip install transformers datasets torch
