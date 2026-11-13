# Artificial Intelligence (SOC4040): code for the methodological guide

Code accompanying the methodological guide *Artificial Intelligence: Methodological Guide for the Course SOC4040*
by Dr. Oybek Eraliyev, School of Computer and Information Engineering, Inha University in Tashkent (2026).
The repository contains **code only**; the guide itself (text, figures, exercises and answers) is distributed separately.

Every chapter of the guide has one folder `chapters/<chapter>/code/` with the scripts and notebooks that produce
the numbers and figures of that chapter. Seeds are fixed, so the printed results can be reproduced.
Notebooks named `*_original.ipynb` are the lecture notebooks as first written; the notebooks without that suffix
are the corrected versions described in the corresponding chapter.

| Ch. | Folder | Title | Week |
|---:|---|---|---:|
| 1 | [`ch01_intro_to_ai`](chapters/ch01_intro_to_ai/code) | Introduction to Artificial Intelligence | 1 |
| 2 | [`ch02_intro_to_ml`](chapters/ch02_intro_to_ml/code) | Introduction to Machine Learning | 1 |
| 3 | [`ch03_linear_polynomial_regression`](chapters/ch03_linear_polynomial_regression/code) | Linear and Polynomial Regression | 2 |
| 4 | [`ch04_logistic_regression`](chapters/ch04_logistic_regression/code) | Logistic Regression | 2 |
| 5 | [`ch05_evaluating_algorithms`](chapters/ch05_evaluating_algorithms/code) | Evaluating Learning Algorithms | 3 |
| 6 | [`ch06_naive_bayes_knn`](chapters/ch06_naive_bayes_knn/code) | Naive Bayes and k-Nearest Neighbours | 3 |
| 7 | [`ch07_decision_trees_random_forests`](chapters/ch07_decision_trees_random_forests/code) | Decision Trees and Random Forests | 4 |
| 8 | [`ch08_svm`](chapters/ch08_svm/code) | Support Vector Machines | 4 |
| 9 | [`ch09_kernel_methods`](chapters/ch09_kernel_methods/code) | Kernel Methods in SVM | 5 |
| 10 | [`ch10_data_preparation`](chapters/ch10_data_preparation/code) | Data Preparation for Machine Learning | 5 |
| 11 | [`ch11_kmeans`](chapters/ch11_kmeans/code) | Clustering with K-Means | 6 |
| 12 | [`ch12_pca`](chapters/ch12_pca/code) | Principal Component Analysis | 6 |
| 13 | [`ch13_deep_neural_networks`](chapters/ch13_deep_neural_networks/code) | Deep Neural Networks | 7 |
| 14 | [`ch14_cnn`](chapters/ch14_cnn/code) | Convolutional Neural Networks | 9 |
| 15 | [`ch15_lightweight_cnn`](chapters/ch15_lightweight_cnn/code) | Lightweight CNN Architectures | 9 |
| 16 | [`ch16_object_detection`](chapters/ch16_object_detection/code) | Object Detection | 10 |
| 17 | [`ch17_rnn_lstm_gru`](chapters/ch17_rnn_lstm_gru/code) | Recurrent Neural Networks: RNN, LSTM and GRU | 11 |
| 18 | [`ch18_nlp_intro`](chapters/ch18_nlp_intro/code) | Introduction to Natural Language Processing | 12 |
| 19 | [`ch19_transformers`](chapters/ch19_transformers/code) | Transformers and Large Language Models | 13 |
| 20 | [`ch20_ai_agents`](chapters/ch20_ai_agents/code) | AI Agents and Multi-Agent Systems | 14 |

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # chapters 13-20 also need torch (see the comments in the file)
cd chapters/ch12_pca/code
python3 pca_scratch.py              # run a script from its own folder
```
`make_figures.py` in a chapter folder regenerates that chapter's figures into `../figures/`.
`python3 scripts/check_code.py` runs every script marked `# ci: run` and syntax-checks the others.

## Notes
* Chapters 18-20 use optional libraries (NLTK, gensim, CrewAI); scripts that need them say so at the top
  and fall back to offline versions where possible.
* Third-party code is not redistributed here. The code is released under the MIT licence (see `LICENSE`); the text of the guide is licensed separately.
