# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 16:22:23 2024

@author: Andres
"""

import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np

# Download necessary NLTK data
nltk.download('vader_lexicon')

def scrape_financial_news(company, days=30):
    """Scrape financial news for a given company over the past 30 days."""
    base_url = "https://news.google.com/rss/search"
    query = f"{company} financial"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    params = {
        "q": query,
        "hl": "en-US",
        "gl": "US",
        "ceid": "US:en",
        "when:range": f"{start_date.strftime('%Y-%m-%d')},{end_date.strftime('%Y-%m-%d')}"
    }
    
    response = requests.get(base_url, params=params)
    soup = BeautifulSoup(response.content, 'xml')
    
    articles = []
    for item in soup.find_all('item'):
        title = item.title.text
        date = datetime.strptime(item.pubDate.text, "%a, %d %b %Y %H:%M:%S %Z")
        articles.append({"title": title, "date": date})
    
    return articles

def analyze_sentiment(text):
    """Analyze the sentiment of a given text."""
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)['compound']

def detect_anomalies(df, threshold=1.5):
    """Detect anomalies in sentiment scores."""
    mean = df['sentiment'].mean()
    std = df['sentiment'].std()
    df['anomaly'] = df['sentiment'].apply(lambda x: abs(x - mean) > threshold * std)
    return df

def plot_sentiment_analysis(df, company):
    """Create a color-coded dot plot of sentiment analysis results."""
    plt.figure(figsize=(12, 6))
    
    # Create a color map
    cmap = plt.cm.get_cmap('RdYlGn')  # Red-Yellow-Green colormap
    normalize = plt.Normalize(vmin=-1, vmax=1)
    
    # Plot all points
    scatter = plt.scatter(df['date'], df['sentiment'], 
                          c=df['sentiment'], cmap=cmap, norm=normalize,
                          s=50, alpha=0.7)
    
    # Highlight anomalies
    plt.scatter(df[df['anomaly']]['date'], df[df['anomaly']]['sentiment'], 
                color='none', edgecolor='red', s=100, linewidth=2, label='Anomaly')
    
    plt.title(f"Sentiment Analysis for {company}")
    plt.xlabel("Date")
    plt.ylabel("Sentiment Score")
    plt.legend(loc='upper left')
    plt.colorbar(scatter, label='Sentiment Score')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{company}_sentiment_analysis.png")
    print(f"Sentiment analysis plot saved as '{company}_sentiment_analysis.png'")

def main(company):
    # Scrape news
    articles = scrape_financial_news(company)
    
    # Analyze sentiment
    df = pd.DataFrame(articles)
    df['sentiment'] = df['title'].apply(analyze_sentiment)
    
    # Detect anomalies
    df = detect_anomalies(df)
    
    # Plot results
    plot_sentiment_analysis(df, company)
    
    # Print anomalies
    anomalies = df[df['anomaly']]
    if not anomalies.empty:
        print(f"Potential anomalies detected for {company}:")
        for _, row in anomalies.iterrows():
            print(f"Date: {row['date']}, Sentiment: {row['sentiment']:.2f}, Title: {row['title']}")
    else:
        print(f"No anomalies detected for {company}")

if __name__ == "__main__":
    company = "Tesla"  # You can change this to any company you're interested in
    main(company)