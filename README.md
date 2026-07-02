
 📈 Algorithmic Trading using Long Short-Term Memory (LSTM)

## Overview

This project develops a supervised deep learning model that predicts whether the next day's closing price of Tesla (NASDAQ: PYPL) will move **UP** or **DOWN** based on the previous **60 trading days** of historical prices.

The project demonstrates the complete machine learning workflow, from downloading financial data to preprocessing, training an LSTM neural network, and evaluating predictive performance.

---

## Project Objectives

- Download historical stock data from Yahoo Finance
- Preprocess financial time-series data
- Create 60-day rolling input sequences
- Train an LSTM neural network
- Predict next-day price direction
- Evaluate model performance using classification metrics

---

## Dataset

- **Source:** Yahoo Finance
- **Stock:** Tesla Inc. (NASDAQ: PYPL)
- **Features:** Daily Closing Price
- **Period:** 2014–2024

---

## Technologies Used

- Python
- TensorFlow / Keras
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- yfinance

---

## Machine Learning Workflow

```text
Yahoo Finance
      │
      ▼
Download Data
      │
      ▼
Data Cleaning
      │
      ▼
Normalization
      │
      ▼
Create 60-Day Sequences
      │
      ▼
Train LSTM
      │
      ▼
Predict UP / DOWN
      │
      ▼
Evaluate Model
```

---

## Repository Structure

```text
algorithmic-trading-lstm/
│
├── notebooks/
├── src/
├── data/
├── images/
├── report/
├── results/
├── README.md
├── requirements.txt
└── LICENSE
```

---

## Future Improvements

- Include technical indicators (RSI, MACD)
- Compare with GRU and Transformer models
- Train using multiple stocks
- Deploy as a web application using Streamlit

---

## Author

**Olajide Ogunbayo.**

Applied Artificial Intelligence Student

---

## Disclaimer

This repository was developed as part of an Applied Artificial Intelligence coursework project and has been further refined as a portfolio project.
