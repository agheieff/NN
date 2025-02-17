import os
import requests
import zipfile
import io
from datetime import datetime
from dateutil.relativedelta import relativedelta

# PARAMETERS (change as needed)
BASE_URL = "https://data.binance.vision/data/futures/um/monthly/klines"
SYMBOL = "BTCUSDT"
TIMEFRAME = "1m"
START_DATE = "2020-01"  # YYYY-MM
END_DATE = "2024-12"    # YYYY-MM
OUTPUT_FILE = "BTCUSDT_1m.csv"

def daterange(start, end):
    current = start
    while current <= end:
        yield current
        current += relativedelta(months=1)

def download_and_extract_csv(url):
    r = requests.get(url)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        # Assumes one CSV file per zip
        csv_filename = z.namelist()[0]
        with z.open(csv_filename) as f:
            return f.read().decode('utf-8')

def main():
    start_dt = datetime.strptime(START_DATE, "%Y-%m")
    end_dt = datetime.strptime(END_DATE, "%Y-%m")

    first_file = True
    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        for dt in daterange(start_dt, end_dt):
            month_str = dt.strftime("%Y-%m")
            file_name = f"{SYMBOL}-{TIMEFRAME}-{month_str}.zip"
            url = f"{BASE_URL}/{SYMBOL}/{TIMEFRAME}/{file_name}"
            print(f"Processing: {url}")
            try:
                csv_content = download_and_extract_csv(url)
            except Exception as e:
                print(f"Error processing {url}: {e}")
                continue

            lines = csv_content.splitlines()
            if not lines:
                continue

            if first_file:
                # Write the entire first file including headers
                outfile.write(csv_content)
                first_file = False
            else:
                # For subsequent files, skip the header line
                outfile.write("\n".join(lines[1:]) + "\n")

if __name__ == "__main__":
    main()
