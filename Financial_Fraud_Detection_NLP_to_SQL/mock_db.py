# In-memory SQLite DB that mimics a PostgreSQL schema, for testing.

import pandas as pd
from sqlalchemy import create_engine

def get_mock_db():
    engine = create_engine('sqlite:///:memory:')

    # Mock transaction data.
    data = {
        'tx_id': [101, 102, 103, 104],
        'amount': [500.0, 7500.0, 120.0, 15000.0],
        'currency': ['USD', 'USD', 'EUR', 'USD'],
        'country': ['US', 'US', 'DE', 'KY'],
        'timestamp': pd.to_datetime(['2024-01-01 10:00', '2024-01-01 10:05', '2024-01-01 10:10', '2024-01-01 10:15']),
        'is_flagged': [0, 0, 0, 0]
    }

    df = pd.DataFrame(data)
    df.to_sql('transactions', engine, index=False)
    return engine