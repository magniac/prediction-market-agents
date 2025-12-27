import os
import argparse
import requests
import pandas as pd
import sys
import time
def get_price_history(token_id, startTs):
    """
    Fetch the price history for a given token ID starting at startTs.
    Returns a DataFrame with columns 'timestamp' and 'p' (price).
    """
    url = f"https://clob.polymarket.com/prices-history?market={token_id}&startTs={startTs}"
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()
    # Expect data like: {"history": [{"t": timestamp, "p": price}, ...]}
    history = data.get("history", [])
    df = pd.DataFrame(history)
    if not df.empty:
        # Convert UNIX timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['t'], unit='s')
        # Keep only the timestamp and price columns
        df = df[['timestamp', 'p']]
    return df

def get_data(name, category, token_prob_yes, token_prob_no, startTs=1):
    """
    Fetch historical price data for the two token IDs, merge on timestamp,
    and save the result as a JSON file in the folder corresponding to the category.
    
    Parameters:
      name           - The base name for the output file (without extension)
      category       - One of: politics, economics, sports, or culture
      token_prob_yes - Token ID for "prob_yes"
      token_prob_no  - Token ID for "prob_no"
      startTs        - The start timestamp for data retrieval (default is 1)
    """
    # Create a mapping for renaming columns in the DataFrames
    token_mapping = {
        token_prob_yes: "prob_yes",
        token_prob_no: "prob_no",
    }
    
    dfs = []
    for token_id, col_name in token_mapping.items():
        try:
            df = get_price_history(token_id, startTs)
            if not df.empty:
                # Rename the price column to our desired column name
                df = df.rename(columns={'p': col_name})
                dfs.append(df)
            else:
                print(f"No data for token {token_id}")
        except Exception as e:
            print(f"Error fetching data for token {token_id}: {e}")
    
    if dfs:
        # Merge data on 'timestamp' with an inner join to only keep rows with both tokens
        merged_df = dfs[0]
        for df in dfs[1:]:
            merged_df = pd.merge(merged_df, df, on='timestamp', how='inner')
        merged_df = merged_df.sort_values(by='timestamp')
        
        # Create the directory for the category if it doesn't exist
        os.makedirs(category, exist_ok=True)
        filename = os.path.join(category, f"{name}.json")
        
        # Save the merged DataFrame to a JSON file (records format with ISO date format)
        merged_df.to_json(filename, orient="records", date_format="iso", index=False)
        print(f"Saved merged data to {filename}")
    else:
        print("No data fetched for any token.")

if __name__ == '__main__':
    # If no command-line arguments are provided, read from 'scrape.txt'
    if len(sys.argv) == 1:
        if os.path.exists("scrape.txt"):
            with open("scrape.txt", "r") as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue  # Skip empty lines
                # Each line should be in the format: name, category, prob_yes, prob_no
                parts = [part.strip() for part in line.split(',')]
                if len(parts) != 4:
                    print(f"Skipping line (unexpected format): {line}")
                    continue
                name, category, prob_yes, prob_no = parts
                print(f"Processing: name={name}, category={category}, prob_yes={prob_yes}, prob_no={prob_no}")
                get_data(name, category, prob_yes, prob_no)
                time.sleep(5)
        else:
            print("No arguments provided and 'scrape.txt' file not found.")
    else:
        # Otherwise, parse the command-line arguments
        parser = argparse.ArgumentParser(
            description="Fetch historical token price data and save as JSON."
        )
        parser.add_argument("--name", required=True, help="Name for the event (e.g., will_biden_win)")
        parser.add_argument("--category", required=True, choices=["politics", "economics", "sports", "culture"],
                            help="Event category (politics, economics, sports, or culture)")
        parser.add_argument("--prob_yes", required=True, help="Token ID for probability yes")
        parser.add_argument("--prob_no", required=True, help="Token ID for probability no")
        parser.add_argument("--startTs", type=int, default=1, help="Start timestamp for fetching data (default: 1)")
        args = parser.parse_args()
        
        get_data(args.name, args.category, args.prob_yes, args.prob_no, args.startTs)
