"""
Debug script to see what's happening in the ICT system
"""

import pandas as pd
import sys
sys.path.append('/home/nyale/Desktop/ML TESTING/ICT')

from ict_trading_system import ICTTradingSystem, ICTConfig, ICTBiasDetector

# Load data
print("Loading data...")
df = pd.read_csv(
    '../xauusd_2023_5m.csv',
    header=None,
    names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']
)
df['DateTime'] = pd.to_datetime(
    df['Date'].astype(str) + ' ' + df['Time'].astype(str),
    format='%Y.%m.%d %H:%M'
)
df.set_index('DateTime', inplace=True)
print(f"Loaded {len(df)} candles")

# Initialize bias detector
bias_detector = ICTBiasDetector(df)
bias_detector.resample_timeframes()

# Test bias detection at different points
print("\n" + "="*70)
print("TESTING BIAS DETECTION")
print("="*70)

test_times = df.resample('1D').first().index[:10]  # First 10 days

for timestamp in test_times:
    bias_info = bias_detector.get_bias(timestamp, ICTConfig())
    print(f"\n{timestamp}")
    print(f"  Weekly Bias: {bias_info['weekly_bias']} (strength: {bias_info.get('weekly_strength', 0):.2f})")
    print(f"  4H Bias: {bias_info['h4_bias']} (strength: {bias_info.get('h4_strength', 0):.2f})")
    print(f"  Combined: {bias_info['bias']} (strength: {bias_info['strength']:.2f})")
    print(f"  Aligned: {bias_info['aligned']}")

# Test Order Block detection
print("\n" + "="*70)
print("TESTING ORDER BLOCK DETECTION")
print("="*70)

from ict_trading_system import ICTOrderBlockDetector

# Test on a sample of data
sample_df = df.tail(200)
obs = ICTOrderBlockDetector.find_order_blocks(sample_df, lookback=50, min_size_pct=0.1)
print(f"\nFound {len(obs)} Order Blocks in last 200 candles")
for ob in obs[:5]:  # Show first 5
    print(f"  {ob['type']} at {ob['time']}: Entry={ob['entry']:.2f}, Target={ob['target']:.2f}")

# Test FVG detection
print("\n" + "="*70)
print("TESTING FVG DETECTION")
print("="*70)

from ict_trading_system import ICTFVDetector

fvgs = ICTFVDetector.find_fvgs(sample_df, lookback=20, min_gap_pct=0.05)
print(f"\nFound {len(fvgs)} Fair Value Gaps in last 200 candles")
for fvg in fvgs[:5]:  # Show first 5
    print(f"  {fvg['type']} at {fvg['time']}: Mid={fvg['mid']:.2f}, Top={fvg['top']:.2f}, Bottom={fvg['bottom']:.2f}")

# Test full system with relaxed config
print("\n" + "="*70)
print("TESTING FULL SYSTEM WITH RELAXED CONFIG")
print("="*70)

relaxed_config = ICTConfig(
    use_order_blocks=True,
    use_fvg=True,
    min_bias_strength=0.3,  # Lower threshold
    require_alignment=False,  # Don't require alignment
    ob_min_candle_size=0.05,  # Smaller OB size
    fvg_min_gap_size=0.02  # Smaller FVG gap
)

system = ICTTradingSystem('../xauusd_2023_5m.csv', relaxed_config)
system.load_data()

# Test finding entries at a few timestamps
test_times = df.resample('6H').first().index[:20]  # Every 6 hours for first 20

entries_found = 0
for timestamp in test_times:
    entries = system.find_ict_entries(timestamp)
    if len(entries) > 0:
        entries_found += len(entries)
        print(f"\n{timestamp}: Found {len(entries)} entries")
        for entry in entries:
            print(f"  {entry['type']} {entry['direction']} @ {entry['entry']:.2f}")

print(f"\nTotal entries found: {entries_found}")

