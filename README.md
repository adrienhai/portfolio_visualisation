# portfolio_visualisation
Simple plots for portfolio viz

# User guide
Install dependencies
```
pip install -r requirements.txt
```

Fill the file portfolio.yaml following the pattern:
```
{YAHOO FINANCE TICKER}:
  {TRANSACTION_DATE}:
    - QTE: {QUANTITY}
    - PRICE: {BUY/SELL PRICE}
```
where {QUANTITY} can be either positive or negative (buy or sell). If you do not remember {BUY/SELL PRICE}, then it will be infered with the closing price for the given ticker at the transaction date.

Run the app:
```
streamlit run app/main.py
```

