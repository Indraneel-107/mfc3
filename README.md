<p align="center">
  <img src="amrita-red-logo.svg" width="150"/>
</p>

# MFC3: Enhancing Summarization Efficiency  
## Text Summarization using T5

---

### 👥 Group D1 — Computationally Efficient Text Summarization Tool

> *School of Engineering, Amrita Vishwa Vidyapeetham*

---

## Team Members

| Name | Roll Number |
|:---:|:---:|
| Indraneel R | CB.SC.U4AIE24323 |
| Rishi Dasari | CB.SC.U4AIE24365 |
| Abhishek Reddy | CB.SC.U4AIE24325 |
| M Aryan | CB.SC.U4AIE24341 |

---

## 🎯 Primary Objective

The primary objective of this project is to build a **computationally efficient text summarization system** that produces concise, non-redundant, and readable summaries.

The project combines:
- **Sparse Optimization (theoretical framework)**
- **Transformer-based summarization (T5 model)**

This hybrid approach improves both:
- Efficiency (optimization-based selection)
- Fluency (neural abstractive summarization)

---

## 🖥️ Platform

| Component | Details |
|-----------|---------|
| **Language** | Python 3.9+ |
| **Framework** | PyTorch / HuggingFace Transformers |
| **Model** | T5 (Text-to-Text Transfer Transformer) — `t5-base` |
| **Dataset** | BBC News Summary Dataset |
| **Optimization (Theory)** | ADMM, Sparse Optimization |
| **Evaluation** | ROUGE Score |
| **Environment** | Google Colab / Local |

---

## ⚙️ Implementation

The system is implemented as a **hybrid pipeline**:

### Stage 1 — Data Processing
- Dataset: BBC News Summary Dataset
- Text cleaning and preprocessing
- Tokenization using T5 tokenizer

### Stage 2 — Sparse Optimization (Conceptual Framework)
- Document represented as matrix:

\[
D \in \mathbb{R}^{d \times n}
\]

- Sentence selection formulated as sparse reconstruction problem

### Stage 3 — Transformer-Based Summarization
- Input format:
- Model: `t5-base`
- Uses encoder-decoder architecture

### Stage 4 — Output Generation
- Beam search decoding
- Final abstractive summary generation

---

## 🤖 Model Details

- Model: **T5-base**
- Library: HuggingFace Transformers

### Generation Parameters:
- Max Length: 150  
- Min Length: 40  
- Beam Size: 4  
- Length Penalty: 2.0  
- Early Stopping: True  

---

## 🧠 Mathematical Formulation

### Data Reconstruction Objective

\[
\| D - DA \|_F^2
\]

---

### Row-Sparsity Regularization

\[
\| A \|_{2,1} = \sum_{i=1}^{n} \| A_i \|_2
\]

---

### Full Optimization Problem

\[
\min_{A} \; \| D - DA \|_F^2 + \lambda \| A \|_{2,1}
\]

Subject to:

\[
A_{ij} \geq 0, \quad \forall i,j
\]

\[
\text{diag}(A) = 0
\]

---

## ⚡ ADMM Optimization

### X-Update

\[
X_j^{k+1} = (D^T D + \rho I)^{-1} \left( D^T D_j + \rho (Z_j^k - U_j^k) \right)
\]

---

### Z-Update (Group Shrinkage)

\[
Z_i^{k+1} = \max\left(1 - \frac{\lambda}{\rho \|x\|_2}, 0 \right) x
\]

where:

\[
x = X_i^{k+1} + U_i^k
\]

---

### U-Update

\[
U^{k+1} = U^k + \rho (X^{k+1} - Z^{k+1})
\]

---

## 🔄 Diversity Enhancement

### Modified Optimization with Dissimilarity

\[
\min_{X,Z} \; \| D - DX \|_F^2 + \mu \, \text{tr}(\Delta^T X) + \lambda \| Z \|_{2,1}
\]

---

## 🔬 Compressive Summarization

### Joint Optimization Problem

\[
\min_{R,A} \; \| D - RA \|_F^2 + \lambda_1 \| A \|_{2,1} + \lambda_2 \sum_{i=1}^{n} \| R_i \|_1
\]

Subject to:

\[
R_{ij}, A_{ij} \geq 0
\]

---

## 📊 Dataset

- **BBC News Summary Dataset**
- ~2,225 articles
- Categories:
  - Business
  - Entertainment
  - Politics
  - Sport
  - Tech

---

## 📈 Results

- Improved summary quality using T5
- Reduced redundancy via sparse modeling
- Better readability compared to extractive methods
- Efficient summarization pipeline

---

## 🚀 How to Run

1. Install dependencies:
2. Run notebook:
   
---

## 📊 Evaluation Metrics

- ROUGE-1  
- ROUGE-2  
- ROUGE-L  
- Precision  
- Recall  
- F1 Score  

---

## 📌 Conclusion

- Sparse optimization improves sentence selection  
- T5 enhances fluency and coherence  
- Hybrid approach provides efficient and high-quality summaries  

---

## 🔮 Future Work

- Fine-tuning T5 on custom datasets  
- Real-time summarization  
- Improved compression techniques  
- Integration with web applications  

---

## 📚 References

- Yao et al., IJCAI 2015  
- HuggingFace Transformers Documentation  
- BBC News Dataset  

---

## 👨‍💻 Author

Indraneel R
