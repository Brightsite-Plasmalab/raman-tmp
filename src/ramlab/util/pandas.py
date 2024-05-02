import pandas as pd
from tqdm import tqdm


def load_csv_with_progress(file_path, chunksize=1000, **kwargs):
    df = pd.concat(
        [
            chunk
            for chunk in tqdm(
                pd.read_csv(file_path, chunksize=chunksize, **kwargs),
                unit="rows",
                unit_scale=chunksize,
                desc="Loading csv file",
            )
        ]
    )

    return df
