# AI-based Sentiment Analysis for Hotel Customer Reviews

An end-to-end solution for analyzing hotel customer reviews using artificial neural networks (ANN) and Ordinary Least Squares (OLS) regression. This platform provides a comprehensive ecosystem that enables researchers to advance sentiment analysis capabilities and allows developers to build and deploy sentiment analysis applications.

## 🎯 Project Overview

The platform focuses on efficient sentiment analysis through:
- Situational `model application`
- `Model training and validation`
- `Model refinement`

Implementation is handled via Over-The-Air deployment of:
1. AI/OLS models (`knowledge base`)
2. Model activations (`activation base`)
3. Training material (`learning base`)
4. Analysis routines (`codeBase`)

## 📊 Data Source

Our analysis is based on Booking.com reviews for hotels in Spain, comprising approximately 950 reviews. The collected features include:
- Customer review scores (1-10 scale)
- Review text classifications
- Free cancellation information
- Review counts and price metrics

## 🚀 Model Performance

### Neural Network (TensorFlow)
- R² score: 0.875 (87.5% variance explained)
- RMSE: 0.212 (average deviation of 0.21 points on 10-point scale)
- MAE: 0.161 (predictions off by 0.16 points on average)

### OLS Model
- R² score: 0.576 (57.6% variance explained)
- RMSE: 0.391
- MAE: 0.170

### Comparative Analysis
The Neural Network outperforms the OLS model due to its ability to capture non-linear relationships. While both models maintain consistent prediction accuracy (similar MAE), the Neural Network's lower RMSE indicates fewer extreme prediction errors.

## 🛠️ Installation

### Prerequisites
```bash
pip install tensorflow==2.15.0
pip install pandas==2.1.4
pip install numpy==1.26.3
pip install matplotlib==3.8.2
pip install seaborn==0.13.1
pip install scikit-learn==1.3.2
pip install statsmodels==0.14.1
```

### Docker Setup

1. Install Docker Desktop from the official website
2. Create required volume:
```bash
docker volume create ai_system
```

3. Pull Docker Images:
```bash
docker pull wasif89/knowledgebase_sentiment_analysis:latest
docker pull wasif89/codespace_sentiment_analysis:latest
```

## 📁 Project Structure

### Code Organization
```
code/
├── scrape_reviews.py    # Data collection script
├── data_clean.ipynb    # Preprocessing notebook
├── train_model.py      # Neural network implementation
├── train_ols.py        # OLS model implementation
```

### Data Organization
```
data/
├── hotels_list.csv            # Raw data
├── joint_data_collection.csv  # Processed data
```

### Docker Images
```
images/
├── knowledgeBase_sentiment_analysis/  # Model container
├── codeBase_sentiment_analysis/       # Data container
```

## 🔍 Model Details

### Features Used
- review_text: Text sentiment (0-4)
- free_cancellation: Binary flag
- reviews_count_scaled: Normalized count
- price_scaled: Normalized price

### Running Analysis

For AI analysis:
```bash
cd scenarios/apply_ai_sentiment_analysis
docker-compose up
```

For OLS analysis:
```bash
cd ../apply_ols_sentiment_analysis
docker-compose up
```

## 🐳 Docker Implementation

### Docker Compose Configuration
```yaml
version: '3'
services:
  knowledgebase:
    image: wasif89/knowledgebase_sentiment_analysis:latest
    volumes:
      - ai_system:/tmp

  codespace:
    image: wasif89/codespace_sentiment_analysis:latest
    volumes:
      - ai_system:/tmp

volumes:
  ai_system:
    external: true
```

## 📝 Course Information

This repository is created and maintained by Mustafa Wasif as part of the course 'M. Grum: Advanced AI-based Application Systems' by the Junior Chair for Business Information Science, esp. AI-based Application Systems at University of Potsdam.

## 📄 License

This project is licensed under the AGPL-3.0 license.
