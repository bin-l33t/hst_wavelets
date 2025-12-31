"""
Analyze ROM forecast results from JSON file.
"""

import json
import numpy as np
import math
import sys

def analyze_results(filename):
    print(f"Loading {filename}...")
    with open(filename) as f:
        results = json.load(f)
    
    print(f"Loaded {len(results)} results")
    
    TC = 2 / math.log(1 + math.sqrt(2))
    T_VALUES = sorted(set(r["T"] for r in results))
    
    # Figure out horizon key format (int or str)
    sample = results[0]["horizons"]
    horizons = list(sample.keys())
    print(f"Horizon keys: {horizons}")
    
    # Convert to ints for sorting
    HORIZONS = sorted([int(h) for h in horizons])
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    print(f"\n{'T':<7} {'h':<4} | {'Raw':<8} {'FFT':<8} {'Mallat':<8} | {'HST k=8':<9} {'Mlt k=8':<9} {'Best':<10}")
    print("-"*80)
    
    for T in T_VALUES:
        T_results = [r for r in results if abs(r["T"] - T) < 0.01]
        
        for h in HORIZONS:
            h_key = str(h) if str(h) in results[0]["horizons"] else h
            
            try:
                raw = np.mean([r["horizons"][h_key]["baselines"]["raw_ridge"] for r in T_results])
                fft = np.mean([r["horizons"][h_key]["baselines"]["fft_ridge"] for r in T_results])
                mallat = np.mean([r["horizons"][h_key]["baselines"]["mallat_ridge"] for r in T_results])
                
                # PCA keys might be int or str
                hst_vals = []
                mlt_vals = []
                for r in T_results:
                    hst_dict = r["horizons"][h_key].get("hst_pca", {})
                    mlt_dict = r["horizons"][h_key].get("mallat_pca", {})
                    
                    # Try both "8" and 8
                    hst_v = hst_dict.get("8", hst_dict.get(8, np.nan))
                    mlt_v = mlt_dict.get("8", mlt_dict.get(8, np.nan))
                    hst_vals.append(hst_v)
                    mlt_vals.append(mlt_v)
                
                hst_8 = np.nanmean(hst_vals)
                mlt_8 = np.nanmean(mlt_vals)
                
                scores = {"raw": raw, "fft": fft, "mallat": mallat, "hst_k8": hst_8, "mlt_k8": mlt_8}
                best = min(scores, key=lambda k: scores[k] if not np.isnan(scores[k]) else float('inf'))
                
                marker = "*" if abs(T - TC) < 0.01 else " "
                print(f"{T:<7.3f}{marker}{h:<4} | {raw:<8.4f} {fft:<8.4f} {mallat:<8.4f} | {hst_8:<9.4f} {mlt_8:<9.4f} {best:<10}")
            except Exception as e:
                print(f"{T:<7.3f} {h:<4} | ERROR: {e}")
    
    # Winner counts
    print("\n" + "="*80)
    print("WINNER ANALYSIS")
    print("="*80)
    
    winners = {}
    for T in T_VALUES:
        T_results = [r for r in results if abs(r["T"] - T) < 0.01]
        for h in HORIZONS:
            h_key = str(h) if str(h) in results[0]["horizons"] else h
            try:
                raw = np.mean([r["horizons"][h_key]["baselines"]["raw_ridge"] for r in T_results])
                fft = np.mean([r["horizons"][h_key]["baselines"]["fft_ridge"] for r in T_results])
                mallat = np.mean([r["horizons"][h_key]["baselines"]["mallat_ridge"] for r in T_results])
                
                hst_vals = []
                mlt_vals = []
                raw_pca_vals = []
                for r in T_results:
                    hst_dict = r["horizons"][h_key].get("hst_pca", {})
                    mlt_dict = r["horizons"][h_key].get("mallat_pca", {})
                    raw_dict = r["horizons"][h_key].get("raw_pca", {})
                    hst_vals.append(hst_dict.get("8", hst_dict.get(8, np.nan)))
                    mlt_vals.append(mlt_dict.get("8", mlt_dict.get(8, np.nan)))
                    raw_pca_vals.append(raw_dict.get("8", raw_dict.get(8, np.nan)))
                
                scores = {
                    "raw_ridge": raw,
                    "fft_ridge": fft, 
                    "mallat_ridge": mallat,
                    "hst_pca8": np.nanmean(hst_vals),
                    "mallat_pca8": np.nanmean(mlt_vals),
                    "raw_pca8": np.nanmean(raw_pca_vals),
                }
                best = min(scores, key=lambda k: scores[k] if not np.isnan(scores[k]) else float('inf'))
                winners[best] = winners.get(best, 0) + 1
            except:
                pass
    
    print("\nWins by method:")
    for method, wins in sorted(winners.items(), key=lambda x: -x[1]):
        print(f"  {method}: {wins}")
    
    print("\n" + "="*80)
    print("DONE")
    print("="*80)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_results(sys.argv[1])
    else:
        # Try to find most recent file
        import glob
        files = glob.glob("rom_forecast_*.json")
        if files:
            analyze_results(sorted(files)[-1])
        else:
            print("Usage: python analyze_rom.py <json_file>")
