"""Load and validate the original UCI measurements."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ['AT', 'V', 'AP', 'RH']
LABELS = {'AT': 'Temperature (°C)', 'V': 'Exhaust vacuum (cm Hg)',
          'AP': 'Ambient pressure (mbar)', 'RH': 'Relative humidity (%)',
          'PE': 'Electrical output (MW)'}


def load_data():
    path = ROOT / 'data/ccpp.csv'
    if path.exists():
        df = pd.read_csv(path)
    else:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=294)
        df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
        df = df[FEATURES + ['PE']]
    df = df[FEATURES + ['PE']].apply(pd.to_numeric, errors='raise')
    if len(df) != 9568 or df.isna().any().any():
        raise ValueError('Expected 9,568 complete UCI observations.')
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        df.to_csv(path, index=False)
    return df


if __name__ == '__main__':
    print(load_data().shape)
