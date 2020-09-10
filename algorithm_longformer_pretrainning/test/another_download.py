from datetime import datetime
from algorithm_longformer_pretrainning.test.stat_year_report import download_url

if __name__ == "__main__":
    start = datetime.now()
    download_url(False)
    end = datetime.now()
    print(f"download 1000 pdf cost: {end-start}")
