import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
from datetime import datetime, timezone, timedelta
from pathlib import Path
import logging
import os
import matplotlib.pyplot as plt

## Risk Free Rate is Hard coded to r=0.045 in back-scholes

# --- 0. SETUP & LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- 1. UTILITIES ---
def get_data_path(filename="gex_history.csv"):
    if os.path.exists('./'): #('./content/drive/MyDrive'):
        base_path = Path ('./')#('/content/drive/MyDrive/QuantTrading/Data')
    else:
        base_path = Path('./')
    base_path.mkdir(parents=True, exist_ok=True)
    return base_path / filename

def get_market_seconds_remaining(expiry_date_str):
    now = datetime.now(timezone.utc)
    expiry_dt = datetime.strptime(expiry_date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    market_close = expiry_dt.replace(hour=21, minute=0, second=0)
    seconds_remaining = (market_close - now).total_seconds()
    return max(seconds_remaining, 60) / (365 * 24 * 3600)

def find_flip_level_old(prices, gex_array):
    """ Still used as a fallback if the new stable flip finder finds nothing. """
    sign_changes = np.where(np.diff(np.sign(gex_array)))[0]
    if len(sign_changes) == 0: return None
    i = np.max(sign_changes)
    x0, x1 = prices[i], prices[i + 1]
    y0, y1 = gex_array[i], gex_array[i + 1]
    return x0 - y0 * (x1 - x0) / (y1 - y0)

import numpy as np

def find_flip_level(prices, gex_array, w=5):
    """Finds the interpolated price level where GEX undergoes the most recent stable sign flip, ignoring short-lived sign noise."""
    s = np.sign(gex_array).astype(int)
    # --- handle zeros (inherit nearest sign) ---
    if np.any(s == 0):
        for i in range(1, len(s)):
            if s[i] == 0:
                s[i] = s[i-1]
        for i in range(len(s)-2, -1, -1):
            if s[i] == 0:
                s[i] = s[i+1]
    # --- run-length encoding ---
    run_starts = np.r_[0, np.flatnonzero(s[1:] != s[:-1]) + 1]
    run_ends   = np.r_[run_starts[1:], len(s)]
    run_lens   = run_ends - run_starts
    run_signs  = s[run_starts]
    mask = (
        (run_signs[1:] * run_signs[:-1] == -1) &
        (run_lens[:-1] >= w) &
        (run_lens[1:]  >= w)
    )

    if not np.any(mask):
        return find_flip_level_old(prices, gex_array) #Revert to original method if no stable flips found. Happens if GEX there a few micro flips just before the big flip, 
                                                     #Ex if big flip is at loc 45 and there was a micro flip on location 42, then this method will fail to find a valid flip if window is specified as 5.

    # pick the LAST stable flip (matches your original behavior)
    run_idx = np.where(mask)[0][-1] + 1
    i = run_starts[run_idx] - 1  # boundary index for interpolation

    # --- your original interpolation ---
    x0, x1 = prices[i], prices[i + 1]
    y0, y1 = gex_array[i], gex_array[i + 1]

    return x0 - y0 * (x1 - x0) / (y1 - y0)    

def save_to_cache_optimized(ticker, spot, flip_level, call_wall, cw_gex, put_wall, pw_gex,history_df):
    """Saves historical levels, magnitudes, and the critical flip point."""
    file_path = get_data_path()
    new_data = pd.DataFrame([{
        'Date': datetime.now().strftime('%m/%d/%Y'),
        'timestamp': datetime.now(),
        'Ticker': ticker,
        'Spot': np.round(spot,2),
        'Flip': np.round(flip_level,2),
        'Call_Wall_Price': np.round(call_wall,2),
        'Call_Wall_GEX': np.round(cw_gex,2),
        'Put_Wall_Price': np.round(put_wall,2),
        'Put_Wall_GEX': np.round(pw_gex,2)
    }])

    header = not file_path.exists()
    now = datetime.now()
    stale_cutoff = now - timedelta(hours=8)

    if history_df is None:
        latest_ts = pd.NaT
    else:
        latest_ts = history_df.loc[history_df['Ticker'] == ticker, 'timestamp'].max()

    if (
        history_df is None
        or ticker not in history_df['Ticker'].values 
        or pd.isna(latest_ts) #if new ticker
        or latest_ts < stale_cutoff
    ):
        try:
            new_data.to_csv(file_path, mode='a', index=False, header=header)
        except Exception as e:
            logger.error(f"IO Error for {ticker}: {e}")

def load_history(ticker, days=None):
    """Loads history for a ticker, optionally limited to the last N days."""
    file_path = get_data_path()
    if not file_path.exists():
        return None
    try:
        df = pd.read_csv(file_path)
        df = df.dropna(subset=['Ticker'])
        df = df[df['Ticker'] == ticker].copy()
        if df.empty:
            return None
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
        if days is not None:
            # Use business-day offset so `days` counts weekdays, not calendar days.
            # e.g. if today is Monday Jan 12, history_days=5 → cutoff = Mon Jan 5
            cutoff = (pd.Timestamp.now() - pd.offsets.BDay(int(days))).to_pydatetime()
            df = df[df['timestamp'] >= cutoff]
        # Keep weekday rows only (Mon=0 … Fri=4)
        df = df[df['timestamp'].dt.dayofweek < 5]
        if df.empty:
            return None
        return df.sort_values('timestamp')
    except Exception as e:
        logger.error(f"Failed to load history: {e}")
        return None


def save_oi_snapshot(ticker: str, df_opt: 'pd.DataFrame') -> None:
    """Saves a per-strike OI snapshot (summed across all expiries) once per trading day."""
    file_path = get_data_path('oi_history.parquet')
    today = pd.Timestamp.now().normalize()

    # Skip if we already saved today's snapshot for this ticker
    if file_path.exists():
        try:
            existing = pd.read_parquet(file_path, columns=['date', 'ticker'])
            if not existing[
                (existing['ticker'] == ticker) & (existing['date'] == today)
            ].empty:
                return
        except Exception:
            pass

    try:
        grouped = (
            df_opt.groupby(['strike', 'is_call'])['oi']
            .sum()
            .reset_index()
        )
        grouped['date'] = today
        grouped['ticker'] = ticker
        new_data = grouped[['date', 'ticker', 'strike', 'is_call', 'oi']]

        if file_path.exists():
            existing = pd.read_parquet(file_path)
            existing['date'] = pd.to_datetime(existing['date'], errors='coerce')
            new_data = pd.concat([existing, new_data], ignore_index=True)

        new_data['date'] = pd.to_datetime(new_data['date'], errors='coerce')
        new_data.to_parquet(file_path, index=False)
    except Exception as exc:
        logger.error(f"OI snapshot save error for {ticker}: {exc}")


def load_oi_history(ticker: str, days: int = 30) -> 'pd.DataFrame | None':
    """Loads per-strike OI history (last N calendar days) for the OI heatmap."""
    file_path = get_data_path('oi_history.parquet')
    if not file_path.exists():
        return None
    try:
        df = pd.read_parquet(file_path)
        df = df[df['ticker'] == ticker].copy()
        if df.empty:
            return None
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
        df = df[df['date'] >= cutoff]
        if df.empty:
            return None
        df['is_call'] = df['is_call'].astype(bool)
        return df.sort_values('date')
    except Exception as exc:
        logger.error(f"OI history load error for {ticker}: {exc}")
        return None

def _bucket_oi_df(df: 'pd.DataFrame', spot: float,
                  bucket_pct: float = 0.02) -> 'pd.DataFrame':
    """
    Groups raw per-strike OI into fixed-size price buckets.

    bucket_size = spot * bucket_pct  (default 2 % of spot price).
    Each strike is assigned to the bucket whose lower boundary is
    floor(strike / bucket_size) * bucket_size.
    OI is summed within each bucket.

    Returns a DataFrame with extra columns:
        bucket_lo     – lower boundary of the bucket ($)
        bucket_center – midpoint used as the Y coordinate on the heatmap
        bucket_label  – human-readable label e.g. "$98-$100"
    The original 'strike' column is dropped and replaced by bucket_center.
    """
    bucket_size = spot * bucket_pct
    df = df.copy()
    df['bucket_lo']     = np.floor(df['strike'] / bucket_size) * bucket_size
    df['bucket_center'] = df['bucket_lo'] + bucket_size / 2.0
    df['bucket_label']  = (
        '$' + df['bucket_lo'].map('{:.0f}'.format) +
        '-$' + (df['bucket_lo'] + bucket_size).map('{:.0f}'.format)
    )
    # columns present differ between oi_history (has 'date','is_call') and others
    grp = [c for c in ['date', 'bucket_center', 'bucket_label', 'is_call']
           if c in df.columns]
    return df.groupby(grp, as_index=False)['oi'].sum()


# --- 2. THE MATH ENGINE (Vectorized) ---
def calculate_gamma_exposure(prices, df_opt):
    """Calculates Dealer Gamma Exposure for a given set of prices."""
    total_gex = np.zeros_like(prices)
    for _, row in df_opt.iterrows():
        # Standard GEX formula: 0.5 * Spot * Gamma * OpenInterest
        # Note: Gamma should be adjusted for Call (+) vs Put (-)
        is_call = 1 if row['is_call'] else -1
        total_gex += is_call * (0.5 * row['strike'] * row['gamma'] * row['oi'])
    return total_gex

def black_scholes_metrics_vectorized(S, K, T, r, sigma, is_call):
    """Vectorized calculation of Gamma, Delta, and Vanna."""
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        pdf_d1 = norm.pdf(d1)
        cdf_d1 = norm.cdf(d1)

        gamma = pdf_d1 / (S * sigma * np.sqrt(T))
        delta = np.where(is_call, cdf_d1, cdf_d1 - 1)
        vanna = (pdf_d1 * d2) / sigma

    return np.nan_to_num(gamma), np.nan_to_num(delta), np.nan_to_num(vanna)

def calculate_gex(prices, df_opt, r=0.045):
    """
    Canonical vectorized GEX calculator. Output in $Billions.
    This is the single source of truth — calculate_gamma_exposure (loop version)
    and the gex_matrix inside calculate_advanced_metrics must produce identical
    values at every price point.
    """
    S_matrix = np.array(prices)[:, np.newaxis]
    K      = df_opt['strike'].values
    T      = df_opt['T'].values
    sigma  = df_opt['iv'].values
    oi     = df_opt['oi'].values
    is_call = df_opt['is_call'].values
 
    with np.errstate(divide='ignore', invalid='ignore'):
        d1    = (np.log(S_matrix / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        gamma = norm.pdf(d1) / (S_matrix * sigma * np.sqrt(T))
 
    cp_mult = np.where(is_call, 1, -1)
    return np.sum(np.nan_to_num(gamma) * oi * 100 * (S_matrix**2) * 0.01 * cp_mult, axis=1) ##Not Scaling to Billions  
 

# --- 3. DATA REFINEMENT ---
def build_refined_options_chain(ticker_obj, max_days=45, min_oi=5):
    all_expiries = ticker_obj.options
    now = datetime.now().strftime('%m/%d/%Y')
    valid_expiries = [e for e in all_expiries if 0 <= (datetime.strptime(e, '%Y-%m-%d') - datetime.strptime(now, '%m/%d/%Y')).days <= max_days]

    if not valid_expiries: return None

    all_chains = []
    for exp in valid_expiries:
        chain = ticker_obj.option_chain(exp)
        c, p = chain.calls.copy(), chain.puts.copy()
        c['is_call'], p['is_call'] = True, False
        c['expiry_str'], p['expiry_str'] = exp, exp
        all_chains.extend([c, p])

    df = pd.concat(all_chains).rename(columns={'impliedVolatility': 'iv', 'openInterest': 'oi'})
    df = df[df['oi'] >= min_oi].dropna(subset=['iv', 'oi'])
    df['T'] = df['expiry_str'].apply(get_market_seconds_remaining)
    return df


# --- 3b. IV INDEX (IVX) ---
def calculate_iv_index(ticker_obj, spot, target_days=30, option_type='both', n_strikes=4, r=0.045):
    """
    Calculates an IV Index (IVX) using the Livevol/CBOE methodology.

    Finds the two expiries that bracket `target_days`, computes a vega-weighted
    average IV for each using the `n_strikes` options nearest to spot, then
    interpolates between the two using a sqrt(days) weighting.

    Parameters
    ----------
    ticker_obj   : yfinance Ticker object
    spot         : current stock price
    target_days  : target days window for the index (default 30)
    option_type  : 'calls', 'puts', or 'both' — which side(s) to use (default 'both')
    n_strikes    : number of strikes nearest to spot per expiry (default 4)
    r            : risk-free rate (default 0.045)

    Returns
    -------
    float : annualized IVX as a percentage, or np.nan if not calculable
    """
    today = datetime.now().date()
    all_expiries = ticker_obj.options

    expiry_days = []
    for e in all_expiries:
        d = (datetime.strptime(e, '%Y-%m-%d').date() - today).days
        if d > 0:
            expiry_days.append((d, e))

    if len(expiry_days) < 1:
        return np.nan

    near_list = [(d, e) for d, e in expiry_days if d <= target_days]
    far_list  = [(d, e) for d, e in expiry_days if d >= target_days]

    # Edge cases: target_days outside the available expiry range
    if not near_list and not far_list:
        return np.nan
    if not near_list:
        near_days, near_exp = expiry_days[0]
        far_days,  far_exp  = expiry_days[1] if len(expiry_days) > 1 else expiry_days[0]
    elif not far_list:
        near_days, near_exp = expiry_days[-2] if len(expiry_days) > 1 else expiry_days[-1]
        far_days,  far_exp  = expiry_days[-1]
    else:
        near_days, near_exp = max(near_list, key=lambda x: x[0])
        far_days,  far_exp  = min(far_list,  key=lambda x: x[0])

    def _vega_weighted_iv(option_df, T_years):
        """Vega-weighted average IV for the n_strikes closest to spot."""
        df = option_df.copy()
        df['_dist'] = (df['strike'] - spot).abs()
        nearest = df.nsmallest(n_strikes, '_dist')
        K     = nearest['strike'].values
        sigma = nearest['impliedVolatility'].values
        sigma = np.where(sigma <= 0, 1e-6, sigma)
        with np.errstate(divide='ignore', invalid='ignore'):
            d1   = (np.log(spot / K) + (r + 0.5 * sigma**2) * T_years) / (sigma * np.sqrt(T_years))
            vega = spot * np.sqrt(T_years) * norm.pdf(d1)
        vega = np.nan_to_num(vega)
        total_vega = vega.sum()
        if total_vega == 0:
            return float(np.nanmean(sigma)) * 100
        return float(np.sum(vega * sigma) / total_vega) * 100

    def _expiry_iv(expiry_str, days):
        T_years = days / 365.0
        chain = ticker_obj.option_chain(expiry_str)
        if option_type == 'calls':
            return _vega_weighted_iv(chain.calls, T_years)
        elif option_type == 'puts':
            return _vega_weighted_iv(chain.puts, T_years)
        else:  # 'both'
            iv_c = _vega_weighted_iv(chain.calls, T_years)
            iv_p = _vega_weighted_iv(chain.puts,  T_years)
            return (iv_c + iv_p) / 2.0

    ivx_near = _expiry_iv(near_exp, near_days)
    ivx_far  = _expiry_iv(far_exp,  far_days)

    if near_days == far_days:
        return ivx_near

    # Square-root-of-days interpolation (Livevol methodology)
    sqrt_near, sqrt_far, sqrt_target = np.sqrt(near_days), np.sqrt(far_days), np.sqrt(target_days)
    denom = sqrt_far - sqrt_near
    if denom == 0:
        return ivx_near
    w_near = (sqrt_far   - sqrt_target) / denom
    w_far  = (sqrt_target - sqrt_near)  / denom
    return w_near * ivx_near + w_far * ivx_far


# --- 3c. CONSTANT MATURITY IV & BS IV HELPERS ---

def _fill_iv_gaps(iv_series):
    """
    Fill interior NaN gaps in an IV Series indexed by strike.
    Uses linear interpolation over interior gaps only; edge NaNs are left as-is.
    """
    return iv_series.interpolate(method='index', limit_area='inside')


def calc_constant_maturity_iv(df_opt, spot, target_days=30):
    """
    Compute a 30-DTE Constant Maturity IV smile using sqrt(T) interpolation
    between the two expiries that bracket target_days (VIX/CBOE methodology).

    Parameters
    ----------
    df_opt       : refined options chain DataFrame (has 'expiry_str', 'iv', 'is_call', 'strike')
    spot         : current stock price
    target_days  : synthetic target DTE (default 30)

    Returns
    -------
    cm_calls, cm_puts : DataFrames with columns ['strike', 'iv_cm'], sorted by strike.
    """
    today = datetime.now().date()
    df = df_opt.copy()
    df['iv'] = pd.to_numeric(df['iv'], errors='coerce')
    df = df[df['iv'] > 0].dropna(subset=['iv'])

    # Build sorted (expiry, dte) list
    dte_map = (
        df[['expiry_str']].drop_duplicates()
        .assign(dte=lambda x: x['expiry_str'].apply(
            lambda e: (datetime.strptime(e, '%Y-%m-%d').date() - today).days))
        .sort_values('dte')
    )
    expiry_dtes = [(r['expiry_str'], r['dte']) for _, r in dte_map.iterrows() if r['dte'] >= 0]

    if not expiry_dtes:
        return pd.DataFrame(columns=['strike', 'iv_cm']), pd.DataFrame(columns=['strike', 'iv_cm'])

    near_list = [(e, d) for e, d in expiry_dtes if d <= target_days]
    far_list  = [(e, d) for e, d in expiry_dtes if d >= target_days]

    if not near_list and not far_list:
        return pd.DataFrame(columns=['strike', 'iv_cm']), pd.DataFrame(columns=['strike', 'iv_cm'])
    if not near_list:
        near_exp, near_dte = expiry_dtes[0]
        far_exp,  far_dte  = expiry_dtes[1] if len(expiry_dtes) > 1 else expiry_dtes[0]
    elif not far_list:
        near_exp, near_dte = expiry_dtes[-2] if len(expiry_dtes) > 1 else expiry_dtes[-1]
        far_exp,  far_dte  = expiry_dtes[-1]
    else:
        near_exp, near_dte = max(near_list, key=lambda x: x[1])
        far_exp,  far_dte  = min(far_list,  key=lambda x: x[1])

    # sqrt(T) weight toward far expiry
    sqrt_near = np.sqrt(max(near_dte, 1))
    sqrt_far  = np.sqrt(max(far_dte,  1))
    sqrt_tgt  = np.sqrt(target_days)
    denom     = sqrt_far - sqrt_near
    weight    = float(np.clip((sqrt_tgt - sqrt_near) / denom, 0, 1)) if denom != 0 else 0.5

    def _interpolate_side(is_call_val):
        near_df = df[(df['expiry_str'] == near_exp) & (df['is_call'] == is_call_val)].groupby('strike')['iv'].mean()
        far_df  = df[(df['expiry_str'] == far_exp)  & (df['is_call'] == is_call_val)].groupby('strike')['iv'].mean()
        all_strikes = near_df.index.union(far_df.index)
        near_iv = _fill_iv_gaps(near_df.reindex(all_strikes))
        far_iv  = _fill_iv_gaps(far_df.reindex(all_strikes))
        cm_iv = ((1 - weight) * near_iv + weight * far_iv).dropna()
        return cm_iv.reset_index().rename(columns={'iv': 'iv_cm'}).sort_values('strike')

    return _interpolate_side(True), _interpolate_side(False)


def calc_bs_iv_chain(df_opt, spot, r=0.045):
    """
    Back-solve IV from market option prices using Black-Scholes (Brent's method).

    Assumptions:
      - Market price = (bid + ask) / 2; fallback to lastPrice if mid unavailable / <= 0.
      - r = 0.045 (risk-free rate, matches existing hardcoded value).
      - q = 0 (no continuous dividend yield).
      - Options where market_price < intrinsic value are skipped (return NaN).

    Returns df_opt with an added 'iv_bs' column.
    """
    from scipy.optimize import brentq

    def _bs_price(S, K, T, r, sigma, is_call):
        if T <= 1e-6 or sigma <= 1e-6:
            return max(S - K, 0.0) if is_call else max(K - S, 0.0)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if is_call:
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    def _solve_iv(S, K, T, r, mkt_price, is_call):
        intrinsic = max(S - K, 0.0) if is_call else max(K - S, 0.0)
        if mkt_price <= 0 or mkt_price < intrinsic * 0.99:
            return np.nan
        try:
            f = lambda sig: _bs_price(S, K, T, r, sig, is_call) - mkt_price
            if f(1e-4) * f(10.0) > 0:
                return np.nan
            return brentq(f, 1e-4, 10.0, xtol=1e-6, maxiter=100)
        except Exception:
            return np.nan

    df = df_opt.copy()
    df['bid']       = pd.to_numeric(df.get('bid',       np.nan), errors='coerce')
    df['ask']       = pd.to_numeric(df.get('ask',       np.nan), errors='coerce')
    df['lastPrice'] = pd.to_numeric(df.get('lastPrice', np.nan), errors='coerce')
    mid = (df['bid'] + df['ask']) / 2.0
    df['market_price'] = np.where(mid.notna() & (mid > 0), mid, df['lastPrice'])

    df['iv_bs'] = [
        _solve_iv(spot, row['strike'], row['T'], r, row['market_price'], bool(row['is_call']))
        for _, row in df.iterrows()
    ]
    return df


def plot_master_dashboard(ticker_symbol, prices, gex_array, spot, flip_level, history,gex_ex_next, gex_only_next, additional_data):
    """
    Combines Current GEX Profile, Historical Corridor, and Wall Intensity 
    into a single 3-panel Master Dashboard.
    """
    if history is not None:
        df = history[history['Ticker'] == ticker_symbol].copy()
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        df = pd.DataFrame()

    # Create a 3-row layout
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 14), gridspec_kw={'height_ratios': [1, 1, 0.6]})
    
    # --- PANEL 1: CURRENT GEX PROFILE (SNAPSHOT) ---
    ax1.plot(prices, gex_array/1e9, color='blue', linestyle='-', lw=2, label=f'Total GEX {ticker_symbol}')
    ax1.plot(prices, gex_ex_next/1e9, color='magenta', linestyle=':', lw=2, label=f'GEX Ex-{additional_data["next_expiry"]}')
    ax1.plot(prices, gex_only_next/1e9, color='orange', linestyle='--', lw=1.5, label=f'GEX Only {additional_data["next_expiry"]}')
    ax1.axhline(0, color='black', alpha=0.5)
    ax1.axvline(spot, color='green', linestyle=':', label=f'Spot: {spot:.0f}')
    idx_call, idx_put = np.argmax(gex_array), np.argmin(gex_array)
    ax1.scatter(prices[idx_call], gex_array[idx_call]/1e9, color='darkgreen', s=100, label=f'Call Wall  {prices[idx_call]:.0f}', zorder=5)
    ax1.scatter(prices[idx_put], gex_array[idx_put]/1e9, color='darkred', s=100, label=f'Put Wall {prices[idx_put]:.0f}', zorder=5)
    flip_level = find_flip_level(prices, gex_array)

    if flip_level:
        ax1.axvline(flip_level, color='red', linestyle='--', label=f'Flip: {flip_level:.0f}')
        ax1.fill_between(prices, gex_array/1e9, 0, where=(gex_array > 0), color='green', alpha=0.1)
        ax1.fill_between(prices, gex_array/1e9, 0, where=(gex_array < 0), color='red', alpha=0.1)
        if (max(gex_array)-min(gex_array))/1e9 >1:
            ax1.text(0.02, 0.95, f"Flip Level: ${flip_level:.2f}", transform=ax1.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax1.set_title(f"GEX Profile: {ticker_symbol} Dealer Gex ${(max(gex_array)-min(gex_array))/1e6:.0f}M", fontsize=14, fontweight='bold')
    ax1.set_xlim(min(prices), max(prices))
    ax1.legend(loc='upper left')
    ax1.set_xlabel("Price ($)")
    ax1.set_ylabel("Dealer Gex ($B)")

    if len(df) < 2:
        print(f"Not enough history to plot {ticker_symbol} trends (Panel 2 & 3 skipped).")
        ax2.set_visible(False)
        ax3.set_visible(False)
        plt.tight_layout()
        return

    # Deduplicate to one row per trading day (keep last snapshot of each day).
    # Then use integer x-positions so no gaps appear for weekends/overnight.
    df_daily = df.copy()
    df_daily['_date_key'] = df_daily['timestamp'].dt.date
    df_daily = df_daily.sort_values('timestamp').drop_duplicates(subset='_date_key', keep='last')
    x = np.arange(len(df_daily))
    xlabels = df_daily['timestamp'].dt.strftime('%m/%d')

    # --- PANEL 2: STRUCTURAL CORRIDOR (WHERE PRICE LIVES) ---
    ax2.plot(x, df_daily['Spot'], color='black', lw=3, label='Spot Price', zorder=5)
    ax2.plot(x, df_daily['Call_Wall_Price'], color='green', linestyle='--', alpha=0.6, label='Call Wall')
    ax2.plot(x, df_daily['Put_Wall_Price'], color='red', linestyle='--', alpha=0.6, label='Put Wall')
    ax2.plot(x, df_daily['Flip'], color='orange', linestyle=':', lw=2, label='Gamma Flip')
    ax2.fill_between(x, df_daily['Flip'], df_daily['Put_Wall_Price'], color='red', alpha=0.08, label='Volatility Zone')
    ax2.set_xticks(x)
    ax2.set_xticklabels(xlabels, rotation=45)
    
    ax2.set_title("PRICE CORRIDOR MIGRATION", fontsize=12)
    ax2.set_ylabel("Price ($)")
    ax2.set_xlabel("Date")
    ax2.legend(loc='upper left', ncol=2)
    ax2.grid(axis='y', alpha=0.3)
 
    # --- PANEL 3: WALL INTENSITY (CONVICTION BARS) ---
    ax3.bar(x, df_daily['Call_Wall_GEX'], color='green', alpha=0.5, label='Call Wall Strength')
    ax3.bar(x, df_daily['Put_Wall_GEX'], color='red', alpha=0.5, label='Put Wall Strength')
    ax3.set_xticks(x)
    ax3.set_xticklabels(xlabels, rotation=45)
    ax3.set_title("DEALER CONVICTION (GEX MAGNITUDE)", fontsize=12)
    ax3.axhline(0, color='black', lw=1)
    ax3.legend(loc='upper left')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.952, hspace=0.32)
    ax1.grid(axis='both', alpha=0.2)
    ax2.grid(axis='both', alpha=0.2)
    ax3.grid(axis='both', alpha=0.2)
    #plt.show() # One single blocking call per ticker
    plt.show(block=False)  # Non-blocking so second figure can open immediately

def plot_options_analysis(ticker_symbol, spot, prices, gex_total, gex_ex_next, gex_only_next, additional_data, history):
    fig, axes = plt.subplots(3, 1, figsize=(7, 18))
    (ax2, ax3, ax4) = axes
    price_min, price_max = min(prices), max(prices)

    # 1. OI Concentration
    calls, puts = additional_data['calls'], additional_data['puts']
    idx_callW, idx_putW = np.argmax(calls['openInterest']), np.argmax(puts['openInterest'])
    peak_oi = max(calls['openInterest'].max(), puts['openInterest'].max())

    ax2.bar(calls['strike'], calls['openInterest'], color='blue', alpha=0.4, label='Calls OI')
    ax2.bar(puts['strike'], puts['openInterest'], color='red', alpha=0.4, label='Puts OI')
    ax2.axvline(spot, color='black', label=f'Spot: {spot:.2f}')
    ax2.axvline(calls['strike'][idx_callW], color='green', linestyle='--', label=f'Call Wall: {calls["strike"][idx_callW]:.2f}')
    ax2.axvline(puts['strike'][idx_putW], color='red', linestyle='--', label=f'Put Wall: {puts["strike"][idx_putW]:.2f}')
    ax2.set_title(f"OI Concentration {ticker_symbol} (Expiry: {additional_data['expiry']})")
    ax2.set_xlim(price_min, price_max)
    ax2.set_ylim(0, peak_oi * 1.1)
    ax2.legend()

    # 2. Volatility Profile — 30-DTE Constant Maturity IV
    cm_calls = additional_data.get('cm_calls', pd.DataFrame())
    cm_puts  = additional_data.get('cm_puts',  pd.DataFrame())
    if not cm_calls.empty and not cm_puts.empty:
        ax3.plot(cm_calls['strike'], cm_calls['iv_cm'] * 100,
                 color='steelblue', lw=2, marker='o', markersize=3, alpha=0.8,
                 label='Calls IV (30-DTE CM)')
        ax3.plot(cm_puts['strike'],  cm_puts['iv_cm']  * 100,
                 color='tomato',    lw=2, marker='x', markersize=3, alpha=0.8,
                 label='Puts IV (30-DTE CM)')
    ax3.axvline(spot, color='black', label=f'Spot: {spot:.2f}')
    for window, color in [('7d', 'blue'), ('14d', 'orange'), ('30d', 'purple')]:
        ax3.axhline(additional_data[f'hv_{window}'], color=color, linestyle='--', linewidth=1.2, label=f'HV {window}')
    ax3.set_title(f"Volatility Profile — {ticker_symbol}  [30-DTE Constant Maturity IV]")
    ax3.set_xlim(price_min, price_max)
    ax3.set_ylim(0, max(80, max(additional_data['hv_7d'], additional_data['hv_14d'], additional_data['hv_30d']) + 5))
    ax3.legend()

    # 3. HV Comparison — clustered bar chart (now / 1w ago / 1mo ago) per window
    windows = ['7d', '10d', '14d', '20d', '30d']
    vals_now   = [additional_data[f'hv_{w}']         for w in windows]
    vals_1w    = [additional_data[f'hv_{w}_1w_ago']  for w in windows]
    vals_1mo   = [additional_data[f'hv_{w}_1mo_ago'] for w in windows]

    x = np.arange(len(windows))
    bar_w = 0.25
    ax4.bar(x - bar_w, vals_now, bar_w, label='Now',      color='steelblue',   alpha=0.9)
    ax4.bar(x,         vals_1w,  bar_w, label='1 Wk Ago', color='darkorange',  alpha=0.8)
    ax4.bar(x + bar_w, vals_1mo, bar_w, label='1 Mo Ago', color='mediumpurple', alpha=0.7)
    ax4.set_xticks(x)
    ax4.set_xticklabels(windows)
    ax4.set_title(f"HV Comparison by Window — {ticker_symbol}")
    ax4.set_ylabel("Annualized HV (%)")
    ax4.set_xlabel("HV Window")
    ax4.hlines(additional_data['ivx_both_7d'],  color='crimson', linestyle='-', xmin=-0.5, xmax=4.5, linewidth=1.0, alpha=0.8, label=f'IVX-7d Mean: {additional_data["ivx_both_7d"]:.1f}%')
    ax4.hlines(additional_data['ivx_both_30d'],  color='crimson', linestyle='--', xmin=-0.5, xmax=4.5, linewidth=1.0, alpha=0.8, label=f'IVX-30d Mean: {additional_data["ivx_both_30d"]:.1f}%')

    ax4.set_xlim(-0.5,4.5)
    ax4.legend()

    ax2.grid(axis='both', alpha=0.2)
    ax3.grid(axis='both', alpha=0.2)
    ax4.grid(axis='y', alpha=0.2)
    plt.tight_layout()
    plt.subplots_adjust(top=0.952, hspace=0.35)
    plt.show(block=False)

def calculate_price_bounds(df_opt,spot):
    prices = np.linspace(spot*0.5, spot*1.5, 100)
    gex_array = calculate_gex(prices, df_opt)
    """Identifies key price levels: Call Wall, Put Wall, and Gamma Flip."""
    idx_call_wall = np.argmax(gex_array)
    idx_put_wall = np.argmin(gex_array)
    call_wall_price = prices[idx_call_wall]
    put_wall_price = prices[idx_put_wall]
    flip_level = find_flip_level(prices, gex_array)
    return int(call_wall_price), int(put_wall_price), flip_level


def plot_iv_skew(ticker_symbol, spot, additional_data):
    """
    IV Skew chart using 30-DTE Constant Maturity IV.
    OTM call IV vs OTM put IV plotted against % distance from spot.
    Green shading where call IV > put IV (positive skew / upside expected).
    Red shading where put IV > call IV (negative skew / downside expected).
    """
    cm_calls = additional_data.get('cm_calls', pd.DataFrame())
    cm_puts  = additional_data.get('cm_puts',  pd.DataFrame())

    if cm_calls.empty or cm_puts.empty:
        print(f"{ticker_symbol}: CM IV not available — run analyze_ticker_suite first.")
        return

    otm_calls = cm_calls[cm_calls['strike'] > spot].copy()
    otm_puts  = cm_puts[cm_puts['strike']   < spot].copy()

    if otm_calls.empty or otm_puts.empty:
        print(f"{ticker_symbol}: Not enough OTM data to plot IV skew.")
        return

    otm_calls['pct_move'] = (otm_calls['strike'] - spot) / spot * 100
    otm_puts['pct_move']  = (spot - otm_puts['strike']) / spot * 100
    otm_calls = otm_calls.sort_values('pct_move')
    otm_puts  = otm_puts.sort_values('pct_move')

    pct_max = min(otm_calls['pct_move'].max(), otm_puts['pct_move'].max())
    if pct_max <= 0:
        return
    grid     = np.linspace(0, pct_max, 300)
    iv_calls = np.interp(grid, otm_calls['pct_move'].values, otm_calls['iv_cm'].values * 100)
    iv_puts  = np.interp(grid, otm_puts['pct_move'].values,  otm_puts['iv_cm'].values  * 100)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.fill_between(grid, iv_calls, iv_puts,
                    where=(iv_calls >= iv_puts), interpolate=True,
                    color='green', alpha=0.25, label='Positive Skew (Upside Expected)')
    ax.fill_between(grid, iv_calls, iv_puts,
                    where=(iv_puts > iv_calls), interpolate=True,
                    color='red', alpha=0.25, label='Negative Skew (Downside Expected)')

    ax.plot(otm_calls['pct_move'], otm_calls['iv_cm'] * 100,
            color='green', lw=2, marker='o', markersize=4, label='OTM Calls IV [30-DTE CM]')
    ax.plot(otm_puts['pct_move'],  otm_puts['iv_cm']  * 100,
            color='red',   lw=2, marker='x', markersize=4, label='OTM Puts IV [30-DTE CM]')

    ax.set_title(f"IV Skew — {ticker_symbol}  [30-DTE Constant Maturity]", fontsize=13, fontweight='bold')
    ax.set_xlabel('% Move from Spot')
    ax.set_ylabel('Implied Volatility (%)')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.1f}%'))
    ax.set_xlim(0, 25)
    ax.legend()
    ax.grid(axis='both', alpha=0.25)
    plt.tight_layout()
    plt.show(block=False)


def plot_iv_skew_bs(ticker_symbol, spot, additional_data):
    """
    Figure 1 of 2 new IV figures — 2 sub-panels:
      Panel 1: Volatility Profile (strike vs IV) using Black-Scholes back-solved IV, 30-DTE CM.
      Panel 2: IV Skew (% move from spot vs IV) using same BS IVs.

    BS assumptions:
      - r = 0.045, q = 0 (no dividend), market price = mid (bid+ask)/2 fallback lastPrice.
      - Intrinsic floor applied; options below floor are skipped.
    """
    df_opt = additional_data.get('df_opt')
    if df_opt is None or df_opt.empty:
        print(f"{ticker_symbol}: No df_opt available for BS IV calculation.")
        return

    df_bs = calc_bs_iv_chain(df_opt, spot)
    df_bs_cm = df_bs.copy()
    df_bs_cm['iv'] = df_bs_cm['iv_bs']
    df_bs_cm = df_bs_cm.dropna(subset=['iv'])

    if df_bs_cm.empty:
        print(f"{ticker_symbol}: BS IV inversion returned no valid results.")
        return

    cm_calls, cm_puts = calc_constant_maturity_iv(df_bs_cm, spot)

    if cm_calls.empty or cm_puts.empty:
        print(f"{ticker_symbol}: Not enough BS IV data for 30-DTE CM calculation.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 10))

    # --- Panel 1: Volatility Profile ---
    ax1.plot(cm_calls['strike'], cm_calls['iv_cm'] * 100,
             color='green', lw=2, marker='o', markersize=3, alpha=0.85,
             label='Calls IV [BS Back-Solved, 30-DTE CM]')
    ax1.plot(cm_puts['strike'],  cm_puts['iv_cm']  * 100,
             color='red',   lw=2, marker='x', markersize=3, alpha=0.85,
             label='Puts IV [BS Back-Solved, 30-DTE CM]')
    ax1.axvline(spot, color='black', linestyle=':', lw=1.5, label=f'Spot: {spot:.2f}')
    for window, color in [('7d', 'blue'), ('14d', 'orange'), ('30d', 'purple')]:
        ax1.axhline(additional_data[f'hv_{window}'], color=color, linestyle='--',
                    linewidth=1.2, label=f'HV {window}')
    ax1.set_title(f"Volatility Profile — {ticker_symbol}  [BS Back-Solved IV, 30-DTE Constant Maturity]",
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('Strike')
    ax1.set_ylabel('Implied Volatility (%)')
    ax1.legend(fontsize=8)
    ax1.grid(axis='both', alpha=0.25)

    # --- Panel 2: IV Skew ---
    otm_calls = cm_calls[cm_calls['strike'] > spot].copy()
    otm_puts  = cm_puts[cm_puts['strike']   < spot].copy()

    if not otm_calls.empty and not otm_puts.empty:
        otm_calls['pct_move'] = (otm_calls['strike'] - spot) / spot * 100
        otm_puts['pct_move']  = (spot - otm_puts['strike']) / spot * 100
        otm_calls = otm_calls.sort_values('pct_move')
        otm_puts  = otm_puts.sort_values('pct_move')

        pct_max = min(otm_calls['pct_move'].max(), otm_puts['pct_move'].max())
        if pct_max > 0:
            grid     = np.linspace(0, pct_max, 300)
            iv_calls = np.interp(grid, otm_calls['pct_move'].values, otm_calls['iv_cm'].values * 100)
            iv_puts  = np.interp(grid, otm_puts['pct_move'].values,  otm_puts['iv_cm'].values  * 100)
            ax2.fill_between(grid, iv_calls, iv_puts, where=(iv_calls >= iv_puts),
                             interpolate=True, color='green', alpha=0.25,
                             label='Positive Skew (Upside Expected)')
            ax2.fill_between(grid, iv_calls, iv_puts, where=(iv_puts > iv_calls),
                             interpolate=True, color='red', alpha=0.25,
                             label='Negative Skew (Downside Expected)')
        ax2.plot(otm_calls['pct_move'], otm_calls['iv_cm'] * 100,
                 color='green', lw=2, marker='o', markersize=4,
                 label='OTM Calls IV [BS, 30-DTE CM]')
        ax2.plot(otm_puts['pct_move'],  otm_puts['iv_cm']  * 100,
                 color='red',   lw=2, marker='x', markersize=4,
                 label='OTM Puts IV [BS, 30-DTE CM]')

    ax2.set_title(f"IV Skew — {ticker_symbol}  [BS Back-Solved IV, 30-DTE Constant Maturity]",
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('% Move from Spot')
    ax2.set_ylabel('Implied Volatility (%)')
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.1f}%'))
    ax2.set_xlim(0, 25)
    ax2.legend(fontsize=8)
    ax2.grid(axis='both', alpha=0.25)

    plt.tight_layout()
    plt.show(block=False)


def plot_iv_comparison(ticker_symbol, spot, additional_data):
    """
    4-panel figure (2 rows × 2 cols) comparing BS Back-Solved vs yfinance 30-DTE CM IV.
      Top-left:     Volatility Profile — BS Back-Solved, 30-DTE CM
      Top-right:    Volatility Profile — yfinance, 30-DTE CM
      Bottom-left:  IV Skew            — BS Back-Solved, 30-DTE CM
      Bottom-right: IV Skew            — yfinance, 30-DTE CM
    """
    # yfinance CM IV (already computed in additional_data)
    cm_calls_yf = additional_data.get('cm_calls', pd.DataFrame())
    cm_puts_yf  = additional_data.get('cm_puts',  pd.DataFrame())

    # BS back-solved CM IV
    cm_calls_bs = cm_puts_bs = pd.DataFrame()
    df_opt = additional_data.get('df_opt')
    if df_opt is not None and not df_opt.empty:
        try:
            df_bs = calc_bs_iv_chain(df_opt, spot)
            df_bs_cm = df_bs.copy()
            df_bs_cm['iv'] = df_bs_cm['iv_bs']
            df_bs_cm = df_bs_cm.dropna(subset=['iv'])
            if not df_bs_cm.empty:
                cm_calls_bs, cm_puts_bs = calc_constant_maturity_iv(df_bs_cm, spot)
        except Exception as _e:
            logger.warning(f"{ticker_symbol}: BS IV computation failed for comparison: {_e}")

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    ax_vbs, ax_vyf, ax_sbs, ax_syf = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    def _draw_vol_profile(ax, cm_c, cm_p, suffix):
        if not cm_c.empty:
            ax.plot(cm_c['strike'], cm_c['iv_cm'] * 100,
                    color='green', lw=2, marker='o', markersize=3, alpha=0.85,
                    label=f'Calls IV [{suffix}]')
        if not cm_p.empty:
            ax.plot(cm_p['strike'], cm_p['iv_cm'] * 100,
                    color='red', lw=2, marker='x', markersize=3, alpha=0.85,
                    label=f'Puts IV [{suffix}]')
        ax.axvline(spot, color='black', linestyle=':', lw=1.5, label=f'Spot: {spot:.2f}')
        for w, c in [('7d', 'blue'), ('14d', 'orange'), ('30d', 'purple')]:
            ax.axhline(additional_data.get(f'hv_{w}', 0), color=c,
                       linestyle='--', lw=1.2, label=f'HV {w}')
        ax.set_xlabel('Strike')
        ax.set_ylabel('IV (%)')
        ax.legend(fontsize=7)
        ax.grid(axis='both', alpha=0.25)

    def _draw_iv_skew(ax, cm_c, cm_p, suffix):
        if cm_c.empty or cm_p.empty:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=11, color='grey')
            return
        otm_c = cm_c[cm_c['strike'] > spot].copy()
        otm_p = cm_p[cm_p['strike']  < spot].copy()
        if otm_c.empty or otm_p.empty:
            ax.text(0.5, 0.5, 'No OTM data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=11, color='grey')
            return
        otm_c['pct_move'] = (otm_c['strike'] - spot) / spot * 100
        otm_p['pct_move'] = (spot - otm_p['strike']) / spot * 100
        otm_c = otm_c.sort_values('pct_move')
        otm_p = otm_p.sort_values('pct_move')
        pct_max = min(otm_c['pct_move'].max(), otm_p['pct_move'].max())
        if pct_max > 0:
            g   = np.linspace(0, pct_max, 300)
            ivc = np.interp(g, otm_c['pct_move'].values, otm_c['iv_cm'].values * 100)
            ivp = np.interp(g, otm_p['pct_move'].values, otm_p['iv_cm'].values * 100)
            ax.fill_between(g, ivc, ivp, where=(ivc >= ivp), interpolate=True,
                            color='green', alpha=0.25, label='Positive Skew (Upside Expected)')
            ax.fill_between(g, ivc, ivp, where=(ivp > ivc), interpolate=True,
                            color='red', alpha=0.25, label='Negative Skew (Downside Expected)')
        ax.plot(otm_c['pct_move'], otm_c['iv_cm'] * 100,
                color='green', lw=2, marker='o', markersize=4,
                label=f'OTM Calls [{suffix}]')
        ax.plot(otm_p['pct_move'], otm_p['iv_cm'] * 100,
                color='red', lw=2, marker='x', markersize=4,
                label=f'OTM Puts [{suffix}]')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.1f}%'))
        ax.set_xlabel('% Move from Spot')
        ax.set_ylabel('IV (%)')
        ax.legend(fontsize=7)
        ax.grid(axis='both', alpha=0.25)

    _draw_vol_profile(ax_vbs, cm_calls_bs, cm_puts_bs, 'BS, 30-DTE CM')
    ax_vbs.set_title(f'Volatility Profile — {ticker_symbol}\n[BS Back-Solved, 30-DTE CM]',
                     fontsize=11, fontweight='bold')

    _draw_vol_profile(ax_vyf, cm_calls_yf, cm_puts_yf, 'yfinance, 30-DTE CM')
    ax_vyf.set_title(f'Volatility Profile — {ticker_symbol}\n[yfinance, 30-DTE CM]',
                     fontsize=11, fontweight='bold')

    _draw_iv_skew(ax_sbs, cm_calls_bs, cm_puts_bs, 'BS, 30-DTE CM')
    ax_sbs.set_title(f'IV Skew — {ticker_symbol}\n[BS Back-Solved, 30-DTE CM]',
                     fontsize=11, fontweight='bold')

    _draw_iv_skew(ax_syf, cm_calls_yf, cm_puts_yf, 'yfinance, 30-DTE CM')
    ax_syf.set_title(f'IV Skew — {ticker_symbol}\n[yfinance, 30-DTE CM]',
                     fontsize=11, fontweight='bold')

    # ---- Sync axes: use yfinance panel limits as the shared baseline ----
    # Vol Profile row: sync X (strike) and Y (IV %) from ax_vyf → ax_vbs
    _vp_xlim = ax_vyf.get_xlim()
    _vp_ylim = ax_vyf.get_ylim()
    ax_vbs.set_xlim(_vp_xlim)
    ax_vbs.set_ylim(_vp_ylim)

    # IV Skew row: X always capped at 25%; Y from yfinance OTM data within that range
    _SKW_X_MAX = 25.0
    _yf_otm_c = cm_calls_yf[cm_calls_yf['strike'] > spot].copy() if not cm_calls_yf.empty else pd.DataFrame()
    _yf_otm_p = cm_puts_yf[cm_puts_yf['strike']  < spot].copy() if not cm_puts_yf.empty else pd.DataFrame()
    if not _yf_otm_c.empty:
        _yf_otm_c['pct_move'] = (_yf_otm_c['strike'] - spot) / spot * 100
    if not _yf_otm_p.empty:
        _yf_otm_p['pct_move'] = (spot - _yf_otm_p['strike']) / spot * 100
    _skew_ivs_yf: list[float] = []
    for _df_sk in [_yf_otm_c, _yf_otm_p]:
        if not _df_sk.empty:
            _in_range = _df_sk[_df_sk['pct_move'] <= _SKW_X_MAX]
            _skew_ivs_yf += (_in_range['iv_cm'] * 100).tolist()
    for _ax_sk in (ax_sbs, ax_syf):
        _ax_sk.set_xlim(0, _SKW_X_MAX)
        if _skew_ivs_yf:
            _pad_sk = (max(_skew_ivs_yf) - min(_skew_ivs_yf)) * 0.10 or 1.0
            _ax_sk.set_ylim(max(0.0, min(_skew_ivs_yf) - _pad_sk),
                            max(_skew_ivs_yf) + _pad_sk)

    fig.suptitle(
        f'{ticker_symbol} — IV Comparison: BS Back-Solved vs yfinance  [30-DTE Constant Maturity]',
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()
    plt.show(block=False)


def plot_iv_skew_multi_expiry(ticker_symbol, spot, additional_data):
    """
    Figure 2 of 2 new IV figures — 2 sub-panels showing 3 nearest expiries.
      Panel 1: Volatility Profile (strike vs yfinance IV) — solid/dashed/dotted by expiry.
      Panel 2: IV Skew (% move from spot vs yfinance IV) — same line styles.
                Shading (green/red) only for the nearest expiry.

    IV gaps within an expiry smile are filled via linear interpolation (interior only).
    """
    ticker_obj = additional_data.get('ticker_obj')
    if ticker_obj is None:
        print(f"{ticker_symbol}: ticker_obj missing from additional_data.")
        return

    today = datetime.now().date()
    all_expiries = ticker_obj.options
    valid = [e for e in all_expiries
             if (datetime.strptime(e, '%Y-%m-%d').date() - today).days >= 1]
    selected = valid[:3]
    if not selected:
        print(f"{ticker_symbol}: No valid expiries available.")
        return

    linestyles  = ['-',       '--',       ':']
    call_colors = ['#00aa44', '#44bb88', '#88ddaa']
    put_colors  = ['#cc0000', '#dd6666', '#eeaaaa']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 11))

    first_otm_calls = first_otm_puts = None
    all_skew_ivs: list[float] = []  # IV values within 0–25% for Y-axis sizing

    for idx, exp in enumerate(selected):
        try:
            chain = ticker_obj.option_chain(exp)
        except Exception as e:
            logger.warning(f"{ticker_symbol}: Could not fetch chain for {exp}: {e}")
            continue

        calls = chain.calls.copy().sort_values('strike')
        puts  = chain.puts.copy().sort_values('strike')

        for df in (calls, puts):
            df['impliedVolatility'] = pd.to_numeric(df['impliedVolatility'], errors='coerce')

        # Fill interior NaN gaps; on persistent gaps (same on retry) interpolation handles it
        calls['impliedVolatility'] = _fill_iv_gaps(
            calls.set_index('strike')['impliedVolatility']).values
        puts['impliedVolatility']  = _fill_iv_gaps(
            puts.set_index('strike')['impliedVolatility']).values

        dte          = (datetime.strptime(exp, '%Y-%m-%d').date() - today).days
        label_suffix = f"({exp}, {dte}d)"
        ls           = linestyles[idx]
        ccol         = call_colors[idx]
        pcol         = put_colors[idx]
        mrkr_c       = 'o' if idx == 0 else None
        mrkr_p       = 'x' if idx == 0 else None

        # Panel 1: Volatility Profile
        ax1.plot(calls['strike'], calls['impliedVolatility'] * 100,
                 color=ccol, lw=2, ls=ls, marker=mrkr_c, markersize=3,
                 label=f'Calls {label_suffix}')
        ax1.plot(puts['strike'],  puts['impliedVolatility'] * 100,
                 color=pcol, lw=2, ls=ls, marker=mrkr_p, markersize=3,
                 label=f'Puts {label_suffix}')

        # Panel 2: IV Skew
        otm_c = calls[calls['strike'] > spot].dropna(subset=['impliedVolatility']).copy()
        otm_p = puts[puts['strike']   < spot].dropna(subset=['impliedVolatility']).copy()
        if otm_c.empty or otm_p.empty:
            continue

        otm_c['pct_move'] = (otm_c['strike'] - spot) / spot * 100
        otm_p['pct_move'] = (spot - otm_p['strike']) / spot * 100
        otm_c = otm_c.sort_values('pct_move')
        otm_p = otm_p.sort_values('pct_move')

        ax2.plot(otm_c['pct_move'], otm_c['impliedVolatility'] * 100,
                 color=ccol, lw=2, ls=ls, marker=mrkr_c, markersize=3,
                 label=f'OTM Calls {label_suffix}')
        ax2.plot(otm_p['pct_move'], otm_p['impliedVolatility'] * 100,
                 color=pcol, lw=2, ls=ls, marker=mrkr_p, markersize=3,
                 label=f'OTM Puts {label_suffix}')

        # Collect IVs within 0–25% for Y-axis sizing
        for _otm_df in [otm_c, otm_p]:
            _in = _otm_df[_otm_df['pct_move'] <= 25.0]
            all_skew_ivs += (_in['impliedVolatility'] * 100).tolist()

        if idx == 0:
            first_otm_calls, first_otm_puts = otm_c, otm_p

    # Shading only for nearest expiry
    if first_otm_calls is not None and first_otm_puts is not None:
        pct_max_s = min(first_otm_calls['pct_move'].max(), first_otm_puts['pct_move'].max())
        if pct_max_s > 0:
            g   = np.linspace(0, pct_max_s, 300)
            ivc = np.interp(g, first_otm_calls['pct_move'].values,
                            first_otm_calls['impliedVolatility'].values * 100)
            ivp = np.interp(g, first_otm_puts['pct_move'].values,
                            first_otm_puts['impliedVolatility'].values * 100)
            ax2.fill_between(g, ivc, ivp, where=(ivc >= ivp), interpolate=True,
                             color='green', alpha=0.2, label='Positive Skew (Nearest Expiry)')
            ax2.fill_between(g, ivc, ivp, where=(ivp > ivc), interpolate=True,
                             color='red', alpha=0.2, label='Negative Skew (Nearest Expiry)')

    ax1.axvline(spot, color='black', linestyle=':', lw=1.5, label=f'Spot: {spot:.2f}')
    ax1.set_title(f"Volatility Profile — {ticker_symbol}  [3 Nearest Expiries, yfinance IV]",
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('Strike')
    ax1.set_ylabel('Implied Volatility (%)')
    ax1.legend(ncol=2, fontsize=8)
    ax1.grid(axis='both', alpha=0.25)

    ax2.set_title(f"IV Skew — {ticker_symbol}  [3 Nearest Expiries, yfinance IV]",
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('% Move from Spot')
    ax2.set_ylabel('Implied Volatility (%)')
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.1f}%'))
    ax2.set_xlim(0, 25)
    if all_skew_ivs:
        _pad_sk = (max(all_skew_ivs) - min(all_skew_ivs)) * 0.10 or 1.0
        ax2.set_ylim(max(0.0, min(all_skew_ivs) - _pad_sk), max(all_skew_ivs) + _pad_sk)
    ax2.legend(ncol=2, fontsize=8)
    ax2.grid(axis='both', alpha=0.25)

    plt.tight_layout()
    plt.show(block=False)


def plot_conviction_signals(ticker_symbol, spot, additional_data):
    """
    Conviction Signals Dashboard — three panels:
      1. Volume / Open Interest ratio per strike (> 1 → new positions being opened aggressively).
      2. IVX vs HV divergence bar chart for 7d and 30d windows.
      3. Text summary of signals (IV/HV relationship, IVX trend, OTM flow).
    """
    calls = additional_data['calls'].copy()
    puts  = additional_data['puts'].copy()
    for df in (calls, puts):
        df['openInterest']      = pd.to_numeric(df['openInterest'],      errors='coerce').fillna(0)
        df['volume']            = pd.to_numeric(df['volume'],            errors='coerce').fillna(0)
        df['impliedVolatility'] = pd.to_numeric(df['impliedVolatility'], errors='coerce').fillna(0)

    calls['vol_oi'] = (calls['volume'] / calls['openInterest'].replace(0, np.nan)).fillna(0)
    puts['vol_oi']  = (puts['volume']  / puts['openInterest'].replace(0, np.nan)).fillna(0)

    ivx_7d  = float(additional_data['ivx_both_7d'])
    ivx_30d = float(additional_data['ivx_both_30d'])
    hv_7d   = float(additional_data['hv_7d'])
    hv_30d  = float(additional_data['hv_30d'])

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 14),
                                         gridspec_kw={'height_ratios': [1.8, 1.2, 0.9]})

    # Compute strike range to share with OI Concentration plot
    all_strikes = pd.concat([calls['strike'], puts['strike']])
    x_min_strike = all_strikes.min()
    x_max_strike = all_strikes.max()

    # --- Panel 1: Vol / OI by strike ---
    ax1.bar(calls['strike'], calls['vol_oi'], color='green', alpha=0.55, label='Calls Vol/OI', width=0.8)
    ax1.bar(puts['strike'],  puts['vol_oi'],  color='red',   alpha=0.55, label='Puts Vol/OI',  width=0.8)
    ax1.axhline(1.0, color='gold', linestyle='--', lw=1.5, label='Vol/OI = 1.0  (new positions threshold)')
    ax1.axvline(spot, color='black', linestyle=':', lw=1.5, label=f'Spot: {spot:.2f}')
    ax1.set_xlim(x_min_strike, x_max_strike)   # sync with OI Concentration x-axis
    ax1.set_title(f"Conviction Signals — {ticker_symbol}", fontsize=13, fontweight='bold')
    ax1.set_ylabel('Volume / Open Interest Ratio')
    ax1.set_xlabel('Strike')
    ax1.legend(ncol=2)
    ax1.grid(axis='y', alpha=0.25)

    # --- Panel 2: IVX vs HV comparison ---
    cmp_labels = ['7d', '30d']
    hv_vals    = [hv_7d,  hv_30d]
    ivx_vals   = [ivx_7d, ivx_30d]
    x_cmp      = np.arange(len(cmp_labels))
    bar_w      = 0.3
    ivx_colors = ['green' if iv < hv else 'red' for iv, hv in zip(ivx_vals, hv_vals)]
    ax2.bar(x_cmp - bar_w / 2, hv_vals,  bar_w, label='HV (Realized)',  color='steelblue',  alpha=0.9)
    ax2.bar(x_cmp + bar_w / 2, ivx_vals, bar_w, color=ivx_colors, alpha=0.85,
            label='IVX (Implied)  — green=IV<HV healthy  /  red=IV>HV nervous')
    ax2.set_xticks(x_cmp)
    ax2.set_xticklabels(cmp_labels)
    ax2.set_ylabel('Volatility (%)')
    ax2.set_title('Implied vs Historical Volatility Divergence')
    ax2.legend(fontsize=8)
    ax2.grid(axis='y', alpha=0.25)

    # --- Panel 3: Signal text summary ---
    ax3.axis('off')
    iv_hv_ratio = ivx_30d / hv_30d if hv_30d > 0 else float('nan')
    if np.isnan(iv_hv_ratio):
        iv_hv_sig = 'IV/HV N/A'
    elif iv_hv_ratio < 0.85:
        iv_hv_sig = f'IV << HV ({iv_hv_ratio:.2f}x) — Market comfortable; steady trend, no reversal priced in'
    elif iv_hv_ratio > 1.20:
        iv_hv_sig = f'IV >> HV ({iv_hv_ratio:.2f}x) — Uncertainty HIGH; big players hedging, reversal risk elevated'
    else:
        iv_hv_sig = f'IV ≈ HV ({iv_hv_ratio:.2f}x) — Neutral; monitor for divergence'

    ivx_delta = ivx_7d - ivx_30d
    if ivx_delta > 3:
        ivx_sig = f'▲ IV Rising short-term (+{ivx_delta:.1f}%) — FOMO / parabolic risk; Vol Crush possible'
    elif ivx_delta < -3:
        ivx_sig = f'▼ IV Falling short-term ({ivx_delta:.1f}%) — Healthy rally or post-event crush'
    else:
        ivx_sig = f'→ IV Stable ({ivx_delta:+.1f}%) — No extreme near-term signal'

    otm_calls_hi = int((calls[calls['strike'] > spot]['vol_oi'] > 1).sum())
    otm_puts_hi  = int((puts[puts['strike']   < spot]['vol_oi'] > 1).sum())
    call_sig = (f'{otm_calls_hi} OTM call strike(s) Vol>OI — '
                + ('Aggressive call buying; uptrend likely continuing' if otm_calls_hi >= 3
                   else 'Moderate call activity'))
    put_sig  = (f'{otm_puts_hi} OTM put strike(s) Vol>OI — '
                + ('Aggressive put buying; downside hedge / bearish conviction' if otm_puts_hi >= 3
                   else 'Moderate put activity'))

    summary = '\n'.join([
        f'  IV/HV  :  {iv_hv_sig}',
        f'  IVX    :  {ivx_sig}',
        f'  Calls  :  {call_sig}',
        f'  Puts   :  {put_sig}',
    ])
    ax3.text(0.01, 0.95, summary, transform=ax3.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f0f4f8', alpha=0.85))
    ax3.set_title('Signal Summary', fontsize=10, loc='left')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.35)
    plt.show(block=False)


def plot_oi_heatmap(ticker_symbol, spot, oi_history,
                    bucket_pct: float = 0.02,
                    hide_percentile: float = 50.0,
                    price_history=None):
    """
    OI Heatmap — three figures: individual Call, individual Put, then a combined side-by-side.
    Y axis = price bucket (centred around spot ±25 %).
    X axis = date of snapshot (skips days with no data).
    Colour intensity ∝ open interest; green shades for calls, red shades for puts.
    White/yellow line connects peak-OI bucket per date to show migration.

    Args:
        bucket_pct: width of each price bucket as a fraction of spot (default 0.02 = 2 %).
        hide_percentile: cells whose OI is below this percentile are blank (default 50).
    """
    import matplotlib as _mpl
    import numpy.ma as ma

    if oi_history is None or oi_history.empty:
        print(f"{ticker_symbol}: No OI history available for heatmap (data accumulates over time).")
        return

    price_low  = spot * 0.75
    price_high = spot * 1.25
    df = oi_history[
        (oi_history['strike'] >= price_low) & (oi_history['strike'] <= price_high)
    ].copy()
    if df.empty:
        print(f"{ticker_symbol}: No OI history within ±25 % of spot for heatmap.")
        return

    df_bucketed = _bucket_oi_df(df, spot, bucket_pct=bucket_pct)
    bucket_size = spot * bucket_pct

    # Build {date → close_price} lookup from price history for spot migration line
    spot_by_date: dict = {}
    if price_history is not None:
        ph = price_history.copy()
        ph.index = pd.to_datetime(ph.index).normalize()
        for ts, row in ph.iterrows():
            spot_by_date[ts.date()] = float(row['Close'])

    configs = [
        (True,  f'{ticker_symbol} — Call OI Heatmap', 'YlGn',   'white'),
        (False, f'{ticker_symbol} — Put OI Heatmap',  'YlOrRd', 'yellow'),
    ]

    # ---- 1. Build per-side raw data (masking applied after global threshold is known) ----
    side_data = {}
    for is_call_val, title, cmap_name, peak_lc in configs:
        sub = df_bucketed[df_bucketed['is_call'] == is_call_val].copy()
        if sub.empty:
            continue

        sub_agg = sub.groupby(['date', 'bucket_center'])['oi'].sum().reset_index()
        pivot   = sub_agg.pivot_table(index='bucket_center', columns='date', values='oi', aggfunc='sum')
        pivot   = pivot.sort_index(ascending=True)
        dates_sorted = sorted(pivot.columns)
        pivot        = pivot[dates_sorted]
        date_labels  = [d.strftime('%m/%d') for d in dates_sorted]
        strikes_arr  = pivot.index.to_numpy(dtype=float)
        z_raw        = pivot.values.astype(float)

        peak_x_idx, peak_y = [], []
        for col_idx, d in enumerate(dates_sorted):
            col = pivot[d].dropna()
            if not col.empty:
                peak_x_idx.append(col_idx)
                peak_y.append(float(col.idxmax()))

        label_map = sub.drop_duplicates('bucket_center').set_index('bucket_center')['bucket_label']
        ytick_labels = [label_map.get(s, f'${s:.0f}') for s in strikes_arr]
        y_extent = (strikes_arr[0] - bucket_size / 2, strikes_arr[-1] + bucket_size / 2)

        cmap_obj = _mpl.colormaps.get_cmap(cmap_name).copy()
        cmap_obj.set_bad(color='white')

        side_data[is_call_val] = dict(
            title=title, cmap_name=cmap_name, cmap_obj=cmap_obj,
            peak_lc=peak_lc, pivot=pivot, z_raw=z_raw,
            dates_sorted=dates_sorted, date_labels=date_labels,
            strikes_arr=strikes_arr, ytick_labels=ytick_labels,
            y_extent=y_extent, peak_x_idx=peak_x_idx, peak_y=peak_y,
            bucket_labels_map=label_map,
        )

    # ---- 2. Compute global percentile threshold across BOTH call and put OI ----
    all_valid = np.concatenate([
        sd['z_raw'][~np.isnan(sd['z_raw']) & (sd['z_raw'] > 0)]
        for sd in side_data.values()
    ]) if side_data else np.array([])

    global_threshold = float(np.nanpercentile(all_valid, hide_percentile)) \
        if all_valid.size > 0 and hide_percentile > 0 else 0.0

    for sd in side_data.values():
        z = sd['z_raw']
        sd['z_masked'] = np.where((z > 0) & (z < global_threshold), np.nan, z) \
            if global_threshold > 0 else z.copy()

    # ---- 3. Side-by-side figure (calls left, puts right, shared Y axis) ----
    if not side_data:
        return

    # Align both sides to a common bucket set for shared Y axis
    all_buckets = sorted(set().union(*[set(sd['strikes_arr'].tolist()) for sd in side_data.values()]))
    y_lo = all_buckets[0]  - bucket_size / 2
    y_hi = all_buckets[-1] + bucket_size / 2

    panels = [(True, 1), (False, 2)] if len(side_data) == 2 else [(list(side_data.keys())[0], 1)]
    n_dates_max = max(len(sd['dates_sorted']) for sd in side_data.values())
    fig_w = max(16, n_dates_max * 0.8 + 4)

    fig_side, axes = plt.subplots(
        1, len(panels),
        figsize=(fig_w, 8),
        sharey=True,
    )
    if len(panels) == 1:
        axes = [axes]

    def _draw_panel(ax, sd):
        im = ax.imshow(
            ma.masked_invalid(sd['z_masked']),
            aspect='auto', cmap=sd['cmap_obj'], origin='lower',
            extent=[-0.5, len(sd['dates_sorted']) - 0.5,
                    sd['y_extent'][0], sd['y_extent'][1]],
            interpolation='nearest',
        )
        plt.colorbar(im, ax=ax, label='Open Interest', shrink=0.80)
        ax.set_yticks(sd['strikes_arr'])
        ax.set_yticklabels(sd['ytick_labels'], fontsize=7)
        ax.set_ylim(y_lo, y_hi)
        if len(sd['peak_x_idx']) > 1:
            ax.plot(sd['peak_x_idx'], sd['peak_y'], color=sd['peak_lc'],
                    lw=2.5, marker='D', markersize=6, label='Peak OI Migration')
        spot_xi, spot_yi = [], []
        for col_idx, d in enumerate(sd['dates_sorted']):
            px = spot_by_date.get(d.date() if hasattr(d, 'date') else d)
            if px is not None:
                spot_xi.append(col_idx)
                spot_yi.append(px)
        if len(spot_xi) > 1:
            ax.plot(spot_xi, spot_yi, color='cyan', lw=2, linestyle='-',
                    marker='o', markersize=5, label='Spot Price')
        elif len(spot_xi) == 1:
            ax.axhline(spot_yi[0], color='cyan', linestyle='--', lw=1.5,
                       label=f'Spot ${spot:.2f}')
        else:
            ax.axhline(spot, color='cyan', linestyle='--', lw=1.5,
                       label=f'Spot ${spot:.2f}')
        ax.set_xticks(range(len(sd['dates_sorted'])))
        ax.set_xticklabels(sd['date_labels'], rotation=45, ha='right', fontsize=8)
        ax.set_xlabel('Date')
        ax.set_ylabel('Strike Bucket ($)')
        ax.set_title(sd['title'], fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=7)
        ax.grid(False)

    for (is_call_val, _), ax in zip(panels, axes):
        if is_call_val in side_data:
            _draw_panel(ax, side_data[is_call_val])

    fig_side.suptitle(
        f'{ticker_symbol} — Call OI (left) & Put OI (right) Heatmaps'
        f'  [top {100 - hide_percentile:.0f}% of OI shown]',
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()
    plt.show(block=False)


# --- 4. EXECUTION & VISUALIZATION ---
def analyze_ticker_suite(symbol, graphs='',history_days=5):
    """
    Consolidated main execution.
    Maintains original naming while adding requested analytical depth.
    """
    # try:
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period='1y')
    if hist.empty: return
    spot = ticker.fast_info['lastPrice']

    # 1. Fetch data using the refined logic
    df_opt = build_refined_options_chain(ticker)
    if df_opt is None or df_opt.empty:
        print(f"{symbol} Yahoo Finance returned no valid options data after filtering. Skipping.")
        return

    # 2. Identify Next Expiry for subsetting
    expiries = ticker.options
    next_expiry = expiries[0]
    df_next = df_opt[df_opt['expiry_str'] == next_expiry]
    df_ex_next = df_opt[df_opt['expiry_str'] != next_expiry]

    # 3. Establish Price Range for GEX profiles
    atm_iv = df_opt.iloc[(df_opt['strike'] - spot).abs().idxmin()]['iv']
    move = spot * atm_iv * np.sqrt(1/12) * 2
    [cwp, pwp, fl] = calculate_price_bounds(df_opt, spot)
    price_minbnd = np.min([spot - move, pwp*0.9, fl*0.9])
    price_maxbnd = np.max([spot + move, cwp*1.1, fl*1.1])

    #prices = np.linspace(spot - move, spot + move, 100) ## THis is more dynamic but can lead to different x-axis ranges across tickers, making visual comparison harder. Alternative is to use a fixed % range around spot for consistency.
    # prices = np.linspace(spot*0.9, spot*1.1, 100) # Alternative fixed range for consistency across tickers
    prices = np.linspace(price_minbnd, price_maxbnd, 200) # Ensure prices are within calculated bounds

    # 4. Calculate GEX Profiles
    gex_total = np.round(calculate_gex(prices, df_opt),2)
    gex_only_next = np.round(calculate_gex(prices, df_next),2)
    gex_ex_next = np.round(calculate_gex(prices, df_ex_next),2)

    # 5. Locate Gamma Flip (where GEX crosses zero)
    flips = np.where(np.diff(np.sign(gex_total)))[0]
    gamma_flip = prices[flips[0]] if len(flips) > 0 else np.nan

    # 6. Extract Walls for History
    if not np.isnan(gamma_flip):
        above_mask = prices > gamma_flip
        below_mask = prices < gamma_flip
        idx_cw = np.where(above_mask)[0][np.argmax(gex_total[above_mask])] if np.any(above_mask) else None
        idx_pw = np.where(below_mask)[0][np.argmin(gex_total[below_mask])] if np.any(below_mask) else None
    else:
        idx_cw, idx_pw = None, None

    cw_p = prices[idx_cw] if idx_cw is not None else np.nan
    cw_g = gex_total[idx_cw] if idx_cw is not None else np.nan

    pw_p = prices[idx_pw] if idx_pw is not None else np.nan
    pw_g = gex_total[idx_pw] if idx_pw is not None else np.nan

    # 7. Persistence
    history_df = load_history(symbol,history_days) # Load last 5 days of history for this ticker
    save_to_cache_optimized(symbol, spot, gamma_flip, cw_p, cw_g, pw_p, pw_g,history_df)

    # Save per-strike OI snapshot (once per day) and load 30-day history for heatmap
    save_oi_snapshot(symbol, df_opt)
    oi_history = load_oi_history(symbol, days=30)
    
    def _hv(returns, n): return returns.tail(n).std() * np.sqrt(252) * 100

    daily_returns      = hist['Close'].pct_change()
    hist_1w_ago        = hist.iloc[:-5]  if len(hist) > 5  else hist
    hist_1mo_ago       = hist.iloc[:-21] if len(hist) > 21 else hist
    daily_ret_1w_ago   = hist_1w_ago['Close'].pct_change()
    daily_ret_1mo_ago  = hist_1mo_ago['Close'].pct_change()

    # IV Index — day window, calls / puts / both
    ivx_both_7d  = calculate_iv_index(ticker, spot, target_days=7, option_type='both')
    # ivx_both_10d  = calculate_iv_index(ticker, spot, target_days=10, option_type='both')
    # ivx_both_14d  = calculate_iv_index(ticker, spot, target_days=14, option_type='both')
    # ivx_both_20d  = calculate_iv_index(ticker, spot, target_days=7, option_type='both')
    ivx_both_30d  = calculate_iv_index(ticker, spot, target_days=30, option_type='both')

    analysis_expiry = expiries[1] if len(expiries) > 1 else expiries[0]

    additional_data = {
        'calls': ticker.option_chain(analysis_expiry).calls,
        'puts':  ticker.option_chain(analysis_expiry).puts,
        'expiry': analysis_expiry, 'next_expiry': next_expiry,
        # Full multi-expiry options chain (for skew & OI heatmap)
        'df_opt': df_opt,
        # Historical per-strike OI (last 30 days) for heatmap
        'oi_history': oi_history,
        # IV Index (30-day)
        'ivx_both_7d':  ivx_both_7d,
        'ivx_both_30d': ivx_both_30d,
        # current HV windows
        'hv_7d':  _hv(daily_returns, 7),
        'hv_10d': _hv(daily_returns, 10),
        'hv_14d': _hv(daily_returns, 14),
        'hv_20d': _hv(daily_returns, 20),
        'hv_30d': _hv(daily_returns, 30),
        # same windows as of 1 week ago
        'hv_7d_1w_ago':  _hv(daily_ret_1w_ago, 7),
        'hv_10d_1w_ago': _hv(daily_ret_1w_ago, 10),
        'hv_14d_1w_ago': _hv(daily_ret_1w_ago, 14),
        'hv_20d_1w_ago': _hv(daily_ret_1w_ago, 20),
        'hv_30d_1w_ago': _hv(daily_ret_1w_ago, 30),
        # same windows as of 1 month ago
        'hv_7d_1mo_ago':  _hv(daily_ret_1mo_ago, 7),
        'hv_10d_1mo_ago': _hv(daily_ret_1mo_ago, 10),
        'hv_14d_1mo_ago': _hv(daily_ret_1mo_ago, 14),
        'hv_20d_1mo_ago': _hv(daily_ret_1mo_ago, 20),
        'hv_30d_1mo_ago': _hv(daily_ret_1mo_ago, 30),
        # Daily close prices (DatetimeIndex → float) for spot migration on OI heatmap
        'price_history': hist[['Close']],
        # Ticker object (for multi-expiry IV plots that need on-demand chain fetches)
        'ticker_obj': ticker,
    }

    # Compute 30-DTE Constant Maturity IV smile (used by vol profile & iv skew)
    try:
        cm_calls, cm_puts = calc_constant_maturity_iv(df_opt, spot)
        additional_data['cm_calls'] = cm_calls
        additional_data['cm_puts']  = cm_puts
    except Exception as _e:
        logger.warning(f"{symbol}: CM IV computation failed: {_e}")
        additional_data['cm_calls'] = pd.DataFrame(columns=['strike', 'iv_cm'])
        additional_data['cm_puts']  = pd.DataFrame(columns=['strike', 'iv_cm'])
    # 8. Call plotting function
    history_safe = history_df if history_df is not None else pd.DataFrame(columns=['Ticker', 'Date', 'timestamp', 'Spot', 'Flip', 'Call_Wall_Price', 'Call_Wall_GEX', 'Put_Wall_Price', 'Put_Wall_GEX'])
    if graphs == 'all' or 'master_dashboard' in graphs:
        plot_master_dashboard(symbol, prices, gex_total, spot, gamma_flip, history_safe, gex_ex_next, gex_only_next, additional_data)
    if graphs == 'all' or 'options_analysis' in graphs:
        plot_options_analysis(symbol, spot, prices, gex_total, gex_ex_next, gex_only_next, additional_data, history_safe)
    if graphs == 'all' or 'iv_comparison' in graphs:
        plot_iv_comparison(symbol, spot, additional_data)
    if 'iv_skew' in graphs and graphs != 'all':
        plot_iv_skew(symbol, spot, additional_data)
    if 'iv_skew_bs' in graphs and graphs != 'all':
        plot_iv_skew_bs(symbol, spot, additional_data)
    if graphs == 'all' or 'iv_skew_multi' in graphs:
        plot_iv_skew_multi_expiry(symbol, spot, additional_data)
    if graphs == 'all' or 'conviction' in graphs:
        plot_conviction_signals(symbol, spot, additional_data)
    if graphs == 'all' or 'oi_heatmap' in graphs:
        plot_oi_heatmap(symbol, spot, oi_history,
                        price_history=additional_data.get('price_history'))
    return symbol,prices,gex_total,spot,gamma_flip,history_safe,gex_ex_next,gex_only_next,additional_data,history_safe
    
    
    # except Exception as e:
    #     logger.error(f"Error in analyze_ticker_suite for {symbol}: {e}")

if __name__ == "__main__":
    for ticker in ["SPY","QQQ","AMZN","NVDA","META","MSFT","GOOGL","INTU","NOW","NFLX","PANW","CRWD","UBER","MDT","PGR","TMUS","AMZN","PLTR","LMT","NOC","BA","LHX","RTX","GD","AXP","COF","V","MA","AVGO","QCOM"]:
        try:
            #analyze_ticker_suite(ticker, graphs='master_dashboard',history_days=5)
            result = analyze_ticker_suite(ticker)
            if result is None:
                continue
            [symbol,prices,gex_total,spot,gamma_flip, history_df,gex_ex_next,gex_only_next,additional_data,history_df] = result
            if spot < gamma_flip:
                print(f"{symbol} - SPOT BELOW FLIP: {spot:.2f} < {gamma_flip:.2f}")
                plot_master_dashboard(symbol, prices, gex_total, spot, gamma_flip, history_df,gex_ex_next, gex_only_next, additional_data)
                plt.show(block=False)
            else:
                print(f"{symbol} - SPOT ABOVE FLIP: {spot:.2f} > {gamma_flip:.2f}")
        except Exception as e:
            plt.show(block=False)
            print(f"Error processing {ticker}: {e}")
    plt.show()
    ''' Analyse SPY separately with all graphs to visually inspect the outputs and ensure no errors in the plotting functions, especially since SPY is a widely followed ETF with rich options data. '''
    # try:
    #     analyze_ticker_suite("SPY", graphs='all', history_days=5)
    #     plt.show()
    # except Exception as e:        print(f"Error processing SPY: {e}")
    # plt.show()
    '''Analyse one stock with all graphs to visually inspect the outputs and ensure no errors in the plotting functions, especially since some graphs involve more complex data manipulations (like the IV skew multi-expiry).'''
    #analyze_ticker_suite("V", graphs='all', history_days=5)