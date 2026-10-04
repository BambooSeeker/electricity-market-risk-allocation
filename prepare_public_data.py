"""Rebuild NYISO prices and forecasts from their official dated ZIP archives."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile
import pandas as pd

ZONES = {'CAPITL': 'Capitl', 'CENTRL': 'Centrl', 'DUNWOD': 'Dunwod',
         'GENESE': 'Genese', 'HUD VL': 'Hud Vl', 'LONGIL': 'Longil',
         'MHK VL': 'Mhk Vl', 'MILLWD': 'Millwd', 'N.Y.C.': 'N.Y.C.',
         'NORTH': 'North', 'WEST': 'West'}


def archives(raw, family):
    paths = sorted(raw.glob(f'nyiso_{family}_2025*.zip'))
    if len(paths) != 12:
        raise ValueError(f'Expected 12 monthly {family} archives, got {len(paths)}')
    for path in paths:
        with zipfile.ZipFile(path) as archive:
            for name in sorted(archive.namelist()):
                if name.endswith('.csv'):
                    with archive.open(name) as stream:
                        yield name, pd.read_csv(stream)


def prices(raw, family):
    frames = []
    for _, frame in archives(raw, family):
        frame = frame[frame.Name.isin(ZONES)].copy()
        frame['timestamp'] = pd.to_datetime(frame['Time Stamp'])
        frame['occurrence'] = frame.groupby(['Name', 'timestamp']).cumcount()
        frame = frame.rename(columns={'Name': 'zone', 'LBMP ($/MWHr)': family + '_price',
                                      'Marginal Cost Congestion ($/MWHr)': family + '_congestion'})
        frames.append(frame[['timestamp', 'zone', 'occurrence', family + '_price', family + '_congestion']])
    return pd.concat(frames, ignore_index=True)


def loads(raw):
    frames = []
    for name, frame in archives(raw, 'isolf'):
        frame['timestamp'] = pd.to_datetime(frame['Time Stamp'])
        frame['issue_date'] = pd.Timestamp(name[:8])
        frame['occurrence'] = frame.groupby('timestamp').cumcount()
        frames.append(frame[frame.issue_date < frame.timestamp.dt.normalize()])
    wide = pd.concat(frames, ignore_index=True).sort_values(['timestamp', 'occurrence', 'issue_date'])
    wide = wide.drop_duplicates(['timestamp', 'occurrence'], keep='last')
    return pd.concat([wide[['timestamp', 'occurrence', 'issue_date', column]]
                      .rename(columns={column: 'load_forecast'}).assign(zone=zone)
                      for zone, column in ZONES.items()], ignore_index=True)


def main(raw, output, source):
    output.mkdir(parents=True, exist_ok=True)
    keys = ['timestamp', 'zone', 'occurrence']
    frame = prices(raw, 'damlbmp').merge(prices(raw, 'rtlbmp'), on=keys, validate='one_to_one')
    frame = frame.merge(loads(raw), on=keys, validate='one_to_one')
    frame = frame.sort_values(['zone', 'timestamp', 'occurrence'], kind='stable').reset_index(drop=True)
    frame['da_rt_spread'] = frame.rtlbmp_price - frame.damlbmp_price
    thresholds = frame[frame.timestamp < '2025-04-01'].groupby('zone').da_rt_spread.quantile(.1)
    frame['tail_threshold'] = frame.zone.map(thresholds)
    frame['lower_tail_event'] = (frame.da_rt_spread <= frame.tail_threshold).astype(int)
    frame['negative_excess'] = (frame.tail_threshold - frame.da_rt_spread).clip(lower=0)
    frame['month'] = frame.timestamp.dt.to_period('M').astype(str)
    frame['zone_month'] = frame.zone + '_' + frame.month
    frame = frame.rename(columns={'damlbmp_price': 'day_ahead_price',
                                  'rtlbmp_price': 'real_time_price',
                                  'damlbmp_congestion': 'day_ahead_congestion'})
    if source and source.exists():
        scores = pd.read_csv(source, parse_dates=['timestamp'])
        frame = frame.merge(scores, on=keys, how='left', validate='one_to_one')
    else:
        frame['zj_source_score'] = float('nan')
    if frame.duplicated(keys).any():
        raise ValueError('Duplicate hourly settlement identifier')
    if not (frame.issue_date < frame.timestamp.dt.normalize()).all():
        raise ValueError('Invalid forecast vintage')
    frame.to_csv(output / 'nyiso_clean_dataset.csv', index=False)
    metadata = {'annual_rows': len(frame), 'confirmation_rows': int((frame.timestamp >= '2025-07-01').sum()),
                'zones': frame.zone.nunique(), 'price_units': 'USD/MWh', 'interval_hours': 1,
                'forecast_rule': 'Latest dated issue strictly before the delivery date',
                'dst_rule': 'Repeated local hour retained by occurrence identifier',
                'source_score': 'Archived export; optional for historical reconstruction'}
    (output / 'rebuild_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(metadata)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, default=Path('data/nyiso/raw'))
    parser.add_argument('--out', type=Path, default=Path('data/nyiso/processed'))
    parser.add_argument('--source', type=Path, default=Path('data/nyiso/source_score_export.csv'))
    args = parser.parse_args()
    main(args.raw, args.out, args.source)
