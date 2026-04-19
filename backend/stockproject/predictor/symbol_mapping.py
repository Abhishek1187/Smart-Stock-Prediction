# Mapping from stock symbols to company names for news queries
symbol_to_company_name = {
    "RELIANCE.NS": "Reliance Industries",
    "AXISBANK.NS": "Axis Bank",
    "HDFCBANK.NS": "HDFC Bank",
    "ONGC.NS": "ONGC",
    "SBIN.NS": "State Bank of India",
    "INFY.NS": "Infosys",
    "TCS.NS": "Tata Consultancy Services",
    "ICICIBANK.NS": "ICICI Bank",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "ADANIPORTS.NS": "Adani Ports",
    "ADANIENT.NS": "Adani Enterprises",
    "BAJFINANCE.NS": "Bajaj Finance",
    "BHARTIARTL.NS": "Bharti Airtel",
    "^NSEI": "NIFTY 50",
    "^NSEBANK": "NIFTY BANK",
    "^NSEMDCP50": "NIFTY MIDCAP 50",
    "^CNXAUTO": "NIFTY AUTO",
    # Add more mappings as needed
}

def get_company_name_from_symbol(symbol: str) -> str:
    """
    Convert stock symbol to company name for news query.
    If symbol not found, return symbol itself.
    """
    return symbol_to_company_name.get(symbol.upper(), symbol.upper())
