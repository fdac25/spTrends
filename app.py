import os
import json
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

ROOT = os.path.abspath(os.path.dirname(__file__))
EDA_RESULTS_DIR = os.path.join(ROOT, "EDA", "results")
TABLES_DIRS = [
    os.path.join(EDA_RESULTS_DIR, "tables"),
    os.path.join(ROOT, "EDA", "tables"),
]

def get_tables_directory():
    """Get the tables directory path, preferring EDA/results/tables"""
    for path in TABLES_DIRS:
        if os.path.isdir(path):
            return path
    logger.error("No tables directory found")
    return None

TABLES_DIR = get_tables_directory()

def load_csv(filename):
    """Load CSV or JSON file from tables directory with proper error handling"""
    if not TABLES_DIR:
        logger.error(f"Cannot load {filename}: no tables directory")
        return None
    
    filepath = os.path.join(TABLES_DIR, filename)
    if not os.path.exists(filepath):
        logger.warning(f"File not found: {filename}")
        return None
    
    try:
        if filename.endswith('.json'):
            with open(filepath, 'r') as f:
                data = json.load(f)
            logger.info(f"Loaded {filename}: JSON data")
            return data
        else:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {filename}: {len(df)} rows, {len(df.columns)} columns")
            return df
    except Exception as e:
        logger.error(f"Error loading {filename}: {str(e)}")
        return None

def load_eda_data():
    """Load EDA analysis data from results directory"""
    try:
        # Load summary data
        summaries = {}
        reports_dir = os.path.join(EDA_RESULTS_DIR, "reports")
        if os.path.exists(reports_dir):
            for file in os.listdir(reports_dir):
                if file.endswith('_summary.json'):
                    analysis_name = file.replace('_summary.json', '')
                    filepath = os.path.join(reports_dir, file)
                    with open(filepath, 'r') as f:
                        summaries[analysis_name] = json.load(f)
        
        # Load table data
        tables = {}
        if TABLES_DIR:
            for file in os.listdir(TABLES_DIR):
                if file.endswith('.csv'):
                    table_name = file.replace('.csv', '')
                    tables[table_name] = load_csv(file)
        
        # Also load processed dataset from processed_data directory
        processed_data_dir = os.path.join(EDA_RESULTS_DIR, "processed_data")
        if os.path.exists(processed_data_dir):
            for file in os.listdir(processed_data_dir):
                if file.endswith('.csv'):
                    table_name = file.replace('.csv', '')
                    filepath = os.path.join(processed_data_dir, file)
                    tables[table_name] = load_csv(filepath)
        
        # Load plot data for regeneration
        plot_data = {}
        # Check both data and processed_data directories
        for data_dir_name in ["data", "processed_data"]:
            data_dir = os.path.join(EDA_RESULTS_DIR, data_dir_name)
            if os.path.exists(data_dir):
                for file in os.listdir(data_dir):
                    if file.endswith('_plot_data.json'):
                        analysis_name = file.replace('_plot_data.json', '')
                        filepath = os.path.join(data_dir, file)
                        with open(filepath, 'r') as f:
                            plot_data[analysis_name] = json.load(f)
        
        logger.info(f"Loaded EDA data: {len(summaries)} summaries, {len(tables)} tables, {len(plot_data)} plot datasets")
        return {
            'summaries': summaries,
            'tables': tables,
            'plot_data': plot_data
        }
    except Exception as e:
        logger.error(f"Error loading EDA data: {str(e)}")
        return None

# Load EDA data on startup
EDA_DATA = load_eda_data()

# Country code to country name mapping
COUNTRY_MAPPING = {
    'US': 'United States', 'GB': 'United Kingdom', 'BR': 'Brazil', 'DE': 'Germany', 'JP': 'Japan',
    'FR': 'France', 'CA': 'Canada', 'AU': 'Australia', 'IT': 'Italy', 'ES': 'Spain', 'MX': 'Mexico',
    'IN': 'India', 'RU': 'Russia', 'CN': 'China', 'KR': 'South Korea', 'NL': 'Netherlands',
    'SE': 'Sweden', 'NO': 'Norway', 'DK': 'Denmark', 'FI': 'Finland', 'PL': 'Poland', 'CZ': 'Czech Republic',
    'HU': 'Hungary', 'AT': 'Austria', 'CH': 'Switzerland', 'BE': 'Belgium', 'PT': 'Portugal',
    'GR': 'Greece', 'TR': 'Turkey', 'IL': 'Israel', 'AE': 'United Arab Emirates', 'SA': 'Saudi Arabia',
    'EG': 'Egypt', 'ZA': 'South Africa', 'NG': 'Nigeria', 'KE': 'Kenya', 'MA': 'Morocco',
    'TN': 'Tunisia', 'DZ': 'Algeria', 'GH': 'Ghana', 'UG': 'Uganda', 'ZW': 'Zimbabwe',
    'BW': 'Botswana', 'ZM': 'Zambia', 'MW': 'Malawi', 'SG': 'Singapore', 'TH': 'Thailand',
    'MY': 'Malaysia', 'ID': 'Indonesia', 'PH': 'Philippines', 'VN': 'Vietnam', 'TW': 'Taiwan',
    'HK': 'Hong Kong', 'NZ': 'New Zealand', 'AR': 'Argentina', 'CL': 'Chile', 'CO': 'Colombia',
    'PE': 'Peru', 'VE': 'Venezuela', 'UY': 'Uruguay', 'PY': 'Paraguay', 'BO': 'Bolivia',
    'EC': 'Ecuador', 'CR': 'Costa Rica', 'PA': 'Panama', 'GT': 'Guatemala', 'HN': 'Honduras',
    'SV': 'El Salvador', 'NI': 'Nicaragua', 'CU': 'Cuba', 'DO': 'Dominican Republic',
    'JM': 'Jamaica', 'TT': 'Trinidad and Tobago', 'BB': 'Barbados'
}

# Cache for expensive operations
from functools import lru_cache
import time

# Simple in-memory cache with TTL
class TTLCache:
    def __init__(self, ttl_seconds=300):  # 5 minute default TTL
        self.cache = {}
        self.ttl = ttl_seconds
    
    def get(self, key):
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            else:
                del self.cache[key]
        return None
    
    def set(self, key, value):
        self.cache[key] = (value, time.time())
    
    def clear(self):
        self.cache.clear()

# Global cache instance
cache = TTLCache(ttl_seconds=300)  # 5 minute cache

def add_release_date_column(df):
    """Create a release_date column from existing date fields."""
    if df is None or df.empty:
        return df

    working_df = df.copy()
    release_source = None

    if 'album_release_date' in working_df.columns:
        release_source = working_df['album_release_date']
    elif 'snapshot_date' in working_df.columns:
        release_source = working_df['snapshot_date']

    if release_source is not None:
        working_df['release_date'] = pd.to_datetime(release_source, errors='coerce')
        working_df['release_date'] = working_df['release_date'].fillna(method='ffill')
    else:
        working_df['release_date'] = pd.NaT

    return working_df

def generate_charts_from_eda():
    """Generate meaningful business intelligence charts from cached EDA summaries."""
    charts_data = {}
    
    # Use real EDA data directly instead of relying on filtered_df
    if EDA_DATA is None:
        logger.error("EDA data not available")
        return charts_data
    
    # 1. Audio Feature Correlation Network - Use real EDA correlations
    audio_summary = EDA_DATA['summaries'].get('audio_features_analysis', {})
    correlations = audio_summary.get('strongest_correlations', [])
    
    if correlations:
        charts_data['audio_correlations'] = {
            'type': 'network',
            'features': audio_summary.get('available_features', []),
            'correlations': correlations,
            'title': 'Audio Feature Correlation Network',
            'insight': f'Identifies {len(correlations)} significant relationships between audio features'
        }
    
    # 2. Country Music Activity Analysis - Use real geographical data
    geo_summary = EDA_DATA['summaries'].get('geographical_analysis', {})
    top_countries = geo_summary.get('top_10_countries', {})
    
    if top_countries:
        # Create country activity chart
        countries = list(top_countries.keys())
        song_counts = list(top_countries.values())
        
        charts_data['country_activity'] = {
            'type': 'bar',
            'countries': [get_country_name(c) for c in countries],
            'song_counts': song_counts,
            'title': 'Top Countries by Music Activity',
            'insight': f'Shows music activity across {len(countries)} most active countries'
        }
    
    # 3. Temporal Trend Analysis - Create seasonal pattern analysis
    processed_df = EDA_DATA['tables'].get('processed_spotify_dataset')
    if processed_df is not None and 'snapshot_date' in processed_df.columns:
        # Convert snapshot_date to datetime and extract month
        processed_df['snapshot_date'] = pd.to_datetime(processed_df['snapshot_date'], errors='coerce')
        processed_df = processed_df.dropna(subset=['snapshot_date'])
        
        # Extract month (1-12) for seasonal analysis
        processed_df['month'] = processed_df['snapshot_date'].dt.month
        
        # Count songs per month across all years
        monthly_counts = processed_df.groupby('month').size()
        
        # Calculate average releases per month across all years
        monthly_averages = {}
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        for month_num in range(1, 13):
            count = monthly_counts.get(month_num, 0)
            monthly_averages[month_names[month_num - 1]] = int(count)
        
        # Find peak and low months
        sorted_months = sorted(monthly_averages.items(), key=lambda x: x[1], reverse=True)
        peak_month = sorted_months[0]
        low_month = sorted_months[-1]
        
        charts_data['temporal_analysis'] = {
            'type': 'seasonal_bar',
            'monthly_averages': monthly_averages,
            'title': 'Seasonal Music Release Patterns',
            'insight': f'Peak release month: {peak_month[0]} ({peak_month[1]:,} avg), Low month: {low_month[0]} ({low_month[1]:,} avg)'
        }
    else:
        # Fallback to EDA data if no release_date column
        monthly_data = EDA_DATA['tables'].get('monthly_song_counts')
        if monthly_data is not None and not monthly_data.empty:
            monthly_values = monthly_data.iloc[:, 0].tolist()
            # Create a realistic timeline assuming recent years
            monthly_timeline = {}
            for i, count in enumerate(monthly_values):
                # Assume data starts from 2020-01
                year = 2020 + (i // 12)
                month = (i % 12) + 1
                month_label = f"{year}-{month:02d}"
                monthly_timeline[month_label] = int(count)
            
            charts_data['temporal_analysis'] = {
                'type': 'single_line',
                'monthly': monthly_timeline,
                'title': 'Music Release Patterns Over Time',
                'insight': f'Shows monthly song releases across {len(monthly_timeline)} months'
            }
    
    # 5. Top Artists Analysis - Use processed dataset to get unique song counts
    processed_df = EDA_DATA['tables'].get('processed_spotify_dataset')
    if processed_df is not None and 'artists' in processed_df.columns:
        # Count unique songs per artist (not total entries)
        artist_counts = processed_df.groupby('artists')['spotify_id'].nunique().sort_values(ascending=False).head(10)
        
        charts_data['top_artists'] = {
            'type': 'bar',
            'artists': artist_counts.index.tolist(),
            'song_counts': artist_counts.values.tolist(),
            'title': 'Top Artists by Unique Song Count',
            'insight': f'Shows artists with the most unique songs (not duplicate entries)'
        }
    
    # 7. Audio Feature Distribution Analysis
    if processed_df is not None and 'danceability' in processed_df.columns:
        # Sample audio features for distribution analysis
        sample_size = min(5000, len(processed_df))
        sample_df = processed_df.sample(n=sample_size)
        
        audio_features = ['danceability', 'energy', 'valence', 'acousticness']
        feature_stats = {}
        
        for feature in audio_features:
            if feature in sample_df.columns:
                feature_stats[feature] = {
                    'mean': round(sample_df[feature].mean(), 3),
                    'std': round(sample_df[feature].std(), 3),
                    'min': round(sample_df[feature].min(), 3),
                    'max': round(sample_df[feature].max(), 3)
                }
        
        charts_data['audio_feature_distribution'] = {
            'type': 'box_plot',
            'features': list(feature_stats.keys()),
            'data': feature_stats,
            'title': 'Audio Feature Distribution Analysis',
            'insight': f'Statistical distribution of {len(feature_stats)} key audio features'
        }
    
    # 4. Explicitness Impact Analysis - Use real EDA data
    explicitness_summary = EDA_DATA['summaries'].get('explicitness_analysis', {})
    
    if explicitness_summary:
        charts_data['explicitness_impact'] = {
            'type': 'comparison',
            'explicit': {
                'mean': round(explicitness_summary.get('group2_mean', 0), 1),
                'median': round(explicitness_summary.get('explicitness_groups_summary', {}).get('median', {}).get('True', 0), 1),
                'count': explicitness_summary.get('explicit_songs_count', 0),
                'percentage': round(explicitness_summary.get('explicit_songs_percentage', 0), 1)
            },
            'clean': {
                'mean': round(explicitness_summary.get('group1_mean', 0), 1),
                'median': round(explicitness_summary.get('explicitness_groups_summary', {}).get('median', {}).get('False', 0), 1),
                'count': explicitness_summary.get('non_explicit_songs_count', 0),
                'percentage': round(explicitness_summary.get('non_explicit_songs_percentage', 0), 1)
            },
            'title': 'Explicitness vs Popularity Analysis',
            'insight': f'Explicit songs make up {round(explicitness_summary.get("explicit_songs_percentage", 0), 1)}% of the dataset'
        }
    
    return charts_data


def build_charts_from_filtered_data(filtered_df, filters=None):
    """Build chart payloads that respond directly to the active filters."""
    charts_data = {}
    if filtered_df is None or filtered_df.empty:
        return charts_data
    
    df = filtered_df.replace([np.inf, -np.inf], np.nan)
    
    # 1. Country activity driven by filtered dataset
    if 'country' in df.columns:
        # Count unique songs per country (not total entries) to avoid duplicates
        if 'spotify_id' in df.columns:
            country_counts = df.groupby('country')['spotify_id'].nunique().sort_values(ascending=False).head(10)
        else:
            # Fallback to row count if spotify_id not available
            country_counts = df['country'].value_counts().head(10)
        
        if not country_counts.empty:
            charts_data['country_activity'] = {
                'type': 'bar',
                'countries': [get_country_name(code) for code in country_counts.index],
                'song_counts': country_counts.astype(int).tolist(),
                'title': 'Top Countries (Filtered View)',
                'insight': f"{country_counts.sum():,} tracks across {len(country_counts)} highlighted markets"
            }
    
    # 2. Temporal trends using release dates
    if 'release_date' in df.columns:
        release_df = df.dropna(subset=['release_date']).copy()
        if not release_df.empty:
            release_df['release_date'] = pd.to_datetime(release_df['release_date'], errors='coerce')
            release_df = release_df.dropna(subset=['release_date'])
            if not release_df.empty:
                release_df['month_label'] = release_df['release_date'].dt.to_period('M').astype(str)
                monthly_counts = release_df.groupby('month_label').size().sort_index()
                recent_months = monthly_counts.tail(24)
                if not recent_months.empty:
                    charts_data['temporal_analysis'] = {
                        'type': 'single_line',
                        'monthly': recent_months.to_dict(),
                        'title': 'Release Volume Over Time',
                        'insight': f"Shows last {len(recent_months)} months after applying filters"
                    }
    
    # 4. Audio feature stats for filtered sample
    audio_features = ['danceability', 'energy', 'valence', 'acousticness', 'speechiness', 'instrumentalness', 'liveness']
    available_audio_features = [feature for feature in audio_features if feature in df.columns]
    feature_stats = {}
    for feature in available_audio_features:
        series = df[feature].dropna()
        if series.empty:
            continue
        std_value = round(series.std(), 3) if len(series) > 1 else 0.0
        feature_stats[feature] = {
            'mean': round(series.mean(), 3),
            'std': std_value if not np.isnan(std_value) else 0.0,
            'min': round(series.min(), 3),
            'max': round(series.max(), 3)
        }
    if feature_stats:
        charts_data['audio_feature_distribution'] = {
            'type': 'box_plot',
            'features': list(feature_stats.keys()),
            'data': feature_stats,
            'title': 'Audio Features (Filtered Sample)',
            'insight': f"{len(feature_stats)} feature distributions recalculated with active filters"
        }
    
    # 5. Explicit vs clean comparison
    if {'is_explicit', 'popularity'}.issubset(df.columns):
        explicit_df = df[['is_explicit', 'popularity']].dropna()
        if not explicit_df.empty:
            explicit_df['is_explicit'] = explicit_df['is_explicit'].astype(bool)
            stats = explicit_df.groupby('is_explicit')['popularity'].agg(['mean', 'median', 'count'])
            total = stats['count'].sum()
            explicit_count = int(stats.loc[True, 'count']) if True in stats.index else 0
            clean_count = int(stats.loc[False, 'count']) if False in stats.index else 0
            charts_data['explicitness_impact'] = {
                'type': 'comparison',
                'explicit': {
                    'mean': round(stats.loc[True, 'mean'], 1) if True in stats.index else 0,
                    'median': round(stats.loc[True, 'median'], 1) if True in stats.index else 0,
                    'count': explicit_count,
                    'percentage': round((explicit_count / total) * 100, 1) if total else 0
                },
                'clean': {
                    'mean': round(stats.loc[False, 'mean'], 1) if False in stats.index else 0,
                    'median': round(stats.loc[False, 'median'], 1) if False in stats.index else 0,
                    'count': clean_count,
                    'percentage': round((clean_count / total) * 100, 1) if total else 0
                },
                'title': 'Explicitness vs Popularity (Filtered)',
                'insight': f"{explicit_count:,} explicit tracks vs {clean_count:,} clean tracks in current view"
            }
    
    # 5. Correlation network recalculated from filtered data
    corr_features = [f for f in audio_features if f in df.columns]
    if len(corr_features) >= 3:
        corr_matrix = df[corr_features].corr().stack().reset_index()
        corr_matrix.columns = ['feature1', 'feature2', 'correlation']
        corr_matrix = corr_matrix[corr_matrix['feature1'] != corr_matrix['feature2']]
        corr_matrix['abs_corr'] = corr_matrix['correlation'].abs()
        corr_matrix = corr_matrix[corr_matrix['abs_corr'] >= 0.3].sort_values('abs_corr', ascending=False)
        
        correlations = []
        seen_pairs = set()
        for _, row in corr_matrix.iterrows():
            pair = tuple(sorted([row['feature1'], row['feature2']]))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            correlations.append({
                'feature1': row['feature1'],
                'feature2': row['feature2'],
                'correlation': round(row['correlation'], 3),
                'strength': 'strong' if row['abs_corr'] >= 0.6 else 'moderate'
            })
            if len(correlations) >= 15:
                break
        
        if correlations:
            charts_data['audio_correlations'] = {
                'type': 'network',
                'features': corr_features,
                'correlations': correlations,
                'title': 'Audio Feature Network (Filtered)',
                'insight': f"{len(correlations)} strongest relationships recalculated from current data slice"
            }
    
    return charts_data


def generate_insightful_charts(filtered_df, filters=None):
    """Return charts that respect filters, falling back to cached EDA summaries when necessary."""
    if filtered_df is None or filtered_df.empty:
        logger.warning("Filtered dataset empty; falling back to EDA charts")
        return generate_charts_from_eda()
    
    try:
        charts = build_charts_from_filtered_data(filtered_df, filters=filters)
        if charts and len(charts) > 0:
            return charts
        logger.warning("No charts built from filtered data; using EDA fallback")
    except Exception as exc:
        logger.error(f"Error building filtered charts: {exc}", exc_info=True)
    
    return generate_charts_from_eda()

def apply_filters_to_dataset(dataset, countries=None, pop_min=None, pop_max=None,
                             release_start=None, release_end=None, trend=None):
    """Apply filters to dataset and return filtered DataFrame"""
    if dataset is None or dataset.empty:
        return dataset
    
    filtered_df = dataset.copy()
    if 'release_date' in filtered_df.columns:
        filtered_df['release_date'] = pd.to_datetime(filtered_df['release_date'], errors='coerce')
    
    # Apply filters
    if countries and 'country' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['country'].isin(countries)]
    
    if pop_min is not None and 'popularity' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['popularity'] >= pop_min]
    
    if pop_max is not None and 'popularity' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['popularity'] <= pop_max]
    
    if release_start and 'release_date' in filtered_df.columns:
        start_dt = pd.to_datetime(release_start, errors='coerce')
        if pd.notna(start_dt):
            filtered_df = filtered_df[filtered_df['release_date'] >= start_dt]
    
    if release_end and 'release_date' in filtered_df.columns:
        end_dt = pd.to_datetime(release_end, errors='coerce')
        if pd.notna(end_dt):
            filtered_df = filtered_df[filtered_df['release_date'] <= end_dt]
    
    # Apply trend filter
    if trend == 'growing' and {'popularity', 'release_date'}.issubset(filtered_df.columns):
        filtered_df = filtered_df[
            (filtered_df['popularity'] > 70) & 
            (filtered_df['release_date'] >= '2023-01-01')
        ]
    elif trend == 'declining' and {'popularity', 'release_date'}.issubset(filtered_df.columns):
        filtered_df = filtered_df[
            (filtered_df['popularity'] < 30) | 
            (filtered_df['release_date'] < '2022-01-01')
        ]
    elif trend == 'steady' and 'popularity' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['popularity'] >= 30) & 
            (filtered_df['popularity'] <= 70)
        ]
    
    return filtered_df

def validate_filter_params(params):
    """Validate and sanitize filter parameters"""
    validated = {}
    
    # Validate countries (should be 2-letter codes)
    if params.get('countries'):
        countries = params['countries']
        if isinstance(countries, str):
            countries = [countries]
        validated['countries'] = [c.upper() for c in countries if len(c) == 2 and c.isalpha()]
    
    # Validate popularity range
    pop_min = params.get('pop_min')
    pop_max = params.get('pop_max')
    if pop_min is not None:
        try:
            pop_min = int(pop_min)
            validated['pop_min'] = max(0, min(100, pop_min))
        except (ValueError, TypeError):
            validated['pop_min'] = None
    if pop_max is not None:
        try:
            pop_max = int(pop_max)
            validated['pop_max'] = max(0, min(100, pop_max))
        except (ValueError, TypeError):
            validated['pop_max'] = None
    
    # Validate date range
    release_start = params.get('release_start')
    release_end = params.get('release_end')
    if release_start:
        try:
            pd.to_datetime(release_start)
            validated['release_start'] = release_start
        except (ValueError, TypeError):
            validated['release_start'] = None
    if release_end:
        try:
            pd.to_datetime(release_end)
            validated['release_end'] = release_end
        except (ValueError, TypeError):
            validated['release_end'] = None
    
    # Validate trend
    trend = params.get('trend')
    if trend and trend in ['growing', 'steady', 'declining']:
        validated['trend'] = trend
    
    return validated

def format_api_response(data, status='success', message=None, metadata=None):
    """Format API response with consistent structure"""
    response = {
        'status': status,
        'data': data,
        'timestamp': datetime.now().isoformat()
    }
    
    if message:
        response['message'] = message
    
    if metadata:
        response['metadata'] = metadata
    
    return response

def get_country_name(country_code):
    """Convert country code to full country name"""
    return COUNTRY_MAPPING.get(country_code, country_code)

def extract_filter_params(request):
    """Extract and validate filter parameters from request"""
    raw_params = {
        'countries': request.args.getlist('country'),
        'pop_min': request.args.get('popMin', type=int),
        'pop_max': request.args.get('popMax', type=int),
        'release_start': request.args.get('releaseStart'),
        'release_end': request.args.get('releaseEnd'),
        'trend': request.args.get('trend')
    }
    
    return validate_filter_params(raw_params)

# Load the main dataset for filtering using real EDA data
def get_filtered_dataset(filters=None):
    """Get filtered dataset using actual Spotify data instead of synthetic data"""
    try:
        if EDA_DATA is None:
            logger.error("EDA data not available")
            return None
        
        # Try to load the actual processed dataset first
        processed_df = EDA_DATA['tables'].get('processed_spotify_dataset')
        if processed_df is not None and len(processed_df) > 0:
            logger.info(f"Using actual processed dataset with {len(processed_df)} records")
            # Sample the data for performance (take every 200th record to get ~10k records)
            sample_df = processed_df.iloc[::200].copy()
            logger.info(f"Sampled dataset to {len(sample_df)} records for performance")
            return add_release_date_column(sample_df)
        
        # If no processed dataset, try to load the original dataset
        try:
            # Look for the original dataset file
            dataset_path = os.path.join(os.path.dirname(__file__), 'spotify_dataset.csv')
            if os.path.exists(dataset_path):
                logger.info("Loading original Spotify dataset")
                df = pd.read_csv(dataset_path)
                # Sample for performance
                sample_df = df.iloc[::200].copy()
                logger.info(f"Loaded and sampled original dataset to {len(sample_df)} records")
                return add_release_date_column(sample_df)
        except Exception as e:
            logger.warning(f"Could not load original dataset: {str(e)}")
        
        logger.error("Processed Spotify dataset not available")
        return None
    except Exception as e:
        logger.error(f"Could not create filtered dataset: {str(e)}")
        return None

# Remove the large MAIN_DATASET - we'll load data on-demand
# MAIN_DATASET = load_main_dataset()  # This was loading 2M+ records into memory

def build_manifest():
    """Build a simplified manifest with only essential static charts"""
    charts = []
    
    # Only include charts that provide real value and aren't redundant with dynamic charts
    # Remove the complex static chart generation that doesn't use filtered data
    return charts



def compute_kpis(filtered_data=None):
    # Use filtered data if provided, otherwise get a sample dataset
    data_to_use = filtered_data if filtered_data is not None else get_filtered_dataset()
    
    # Calculate total tracks from the actual dataset
    total_tracks = len(data_to_use) if data_to_use is not None and not data_to_use.empty else 0
    
    # Calculate countries covered and averages from the dataset
    countries_covered = 0
    avg_energy = 0.0
    avg_danceability = 0.0
    
    if data_to_use is not None and not data_to_use.empty:
        countries_covered = data_to_use['country'].nunique()
        avg_energy = round(data_to_use['energy'].mean(), 3)
        avg_danceability = round(data_to_use['danceability'].mean(), 3)
    
    # Use real EDA data for accurate KPIs when no filtered data is provided
    if filtered_data is None and EDA_DATA is not None:
        # Get accurate numbers from EDA analysis
        basic_summary = EDA_DATA['summaries'].get('basic_data', {})
        geo_summary = EDA_DATA['summaries'].get('geographical_analysis', {})
        
        # Use real total tracks from EDA
        total_tracks = basic_summary.get('total_records', total_tracks)
        
        # Use real country count from EDA
        countries_covered = geo_summary.get('total_countries', countries_covered)
        
        # Use real audio features from EDA if available
        audio_summary = EDA_DATA['summaries'].get('audio_features_analysis', {})
        if audio_summary:
            # These would need to be calculated from the actual data, but for now use the synthetic values
            pass
    
    return {
        "total_tracks": total_tracks,
        "avg_energy": avg_energy,
        "avg_danceability": avg_danceability,
        "countries_covered": countries_covered,
        "last_ingest": datetime.now().strftime("%Y-%m-%d %H:%M")
    }

@app.route("/")
def index():
    manifest = build_manifest()
    kpis = compute_kpis()
    return render_template("index.html", kpis=kpis, manifest=manifest)

@app.route("/api/filter-options")
def get_filter_options():
    """Get available filter options with proper error handling and caching"""
    try:
        # Check cache first
        cache_key = "filter_options"
        cached_options = cache.get(cache_key)
        if cached_options:
            logger.info("Returning cached filter options")
            return jsonify(format_api_response(cached_options, message="Filter options loaded from cache"))
        
        # Get dataset for filter options
        dataset = get_filtered_dataset()
        if dataset is None or dataset.empty:
            logger.error("Dataset not available")
            return jsonify(format_api_response(None, status='error', message="Dataset not loaded")), 500
        
        # Create country options with both codes and names
        country_codes = sorted(dataset['country'].dropna().unique().tolist()) if 'country' in dataset.columns else []
        country_options = [{"code": code, "name": get_country_name(code)} for code in country_codes]
        
        # Build filter options based on available columns
        filter_options = {
            "countries": country_options,
        }
        if 'popularity' in dataset.columns and not dataset['popularity'].dropna().empty:
            filter_options["popularity_range"] = {
                "min": int(dataset['popularity'].min()),
                "max": int(dataset['popularity'].max())
            }
        else:
            filter_options["popularity_range"] = {"min": 0, "max": 100}
        
        if 'release_date' in dataset.columns and not dataset['release_date'].dropna().empty:
            filter_options["date_range"] = {
                "start": dataset['release_date'].min().strftime('%Y-%m-%d'),
                "end": dataset['release_date'].max().strftime('%Y-%m-%d')
            }
        else:
            filter_options["date_range"] = {"start": "2020-01-01", "end": "2024-12-31"}
        
        # Cache the result
        cache.set(cache_key, filter_options)
        
        metadata = {
            "total_countries": len(country_options),
            "dataset_size": len(dataset)
        }
        
        return jsonify(format_api_response(filter_options, message="Filter options loaded successfully", metadata=metadata))
    except Exception as e:
        logger.error(f"Error getting filter options: {str(e)}")
        return jsonify(format_api_response(None, status='error', message="Internal server error")), 500

@app.route("/api/filtered-data")
def get_filtered_data():
    """Get filtered data based on query parameters with proper error handling"""
    try:
        # Get dataset
        dataset = get_filtered_dataset()
        if dataset is None or dataset.empty:
            logger.error("Dataset not available")
            return jsonify(format_api_response(None, status='error', message="Dataset not loaded")), 500
        
        # Extract and validate filter parameters
        filters = extract_filter_params(request)
        
        # Apply filters using helper function
        filtered_df = apply_filters_to_dataset(dataset, **filters)
    
        logger.debug(f"Applied filters, resulting in {len(filtered_df)} records")
        
        # Calculate KPIs from filtered data
        filtered_kpis = compute_kpis(filtered_df)
        
        # Prepare response data
        response_data = {
            "total_records": len(filtered_df),
            "filtered_data": filtered_df.to_dict('records')[:1000],  # Limit to 1000 records for performance
            "kpis": filtered_kpis
        }
        
        metadata = {
            "filters_applied": {k: v for k, v in filters.items() if v},
            "original_dataset_size": len(dataset),
            "filtered_dataset_size": len(filtered_df),
            "data_limit": 1000
        }
        
        return jsonify(format_api_response(response_data, message="Filtered data retrieved successfully", metadata=metadata))
    except Exception as e:
        logger.error(f"Error processing filtered data: {str(e)}")
        return jsonify(format_api_response(None, status='error', message="Internal server error")), 500

@app.route("/api/chart-data")
def get_chart_data():
    """Get chart data based on current filters with proper error handling"""
    try:
        # Get dataset
        dataset = get_filtered_dataset()
        if dataset is None or dataset.empty:
            logger.error("Dataset not available")
            return jsonify({"error": "Dataset not loaded"}), 500
        
        # Extract filter parameters and apply filters
        filters = extract_filter_params(request)
        filtered_df = apply_filters_to_dataset(dataset, **filters)
        
        logger.debug(f"Applied chart filters, resulting in {len(filtered_df)} records")
        
        # Generate insightful charts instead of basic ones
        # Pass filters so we can use full dataset for accurate country counts
        charts_data = generate_insightful_charts(filtered_df, filters=filters)
        
        # Add metadata about the analysis
        metadata = {
            "filters_applied": {k: v for k, v in filters.items() if v},
            "dataset_size": len(filtered_df),
            "charts_generated": len(charts_data),
            "chart_types": list(charts_data.keys())
        }
        
        return jsonify(format_api_response(charts_data, message="Insightful charts generated successfully", metadata=metadata))
    except Exception as e:
        logger.error(f"Error processing chart data: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/api/eda-data")
def get_eda_data():
    """Get EDA analysis data for frontend visualization"""
    try:
        if EDA_DATA is None:
            return jsonify({"error": "EDA data not available"}), 500
        
        # Return structured EDA data
        return jsonify({
            "summaries": EDA_DATA['summaries'],
            "available_tables": list(EDA_DATA['tables'].keys()),
            "available_plots": list(EDA_DATA['plot_data'].keys()),
            "status": "success"
        })
    except Exception as e:
        logger.error(f"Error serving EDA data: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/api/eda-table/<table_name>")
def get_eda_table(table_name):
    """Get specific EDA table data"""
    try:
        if EDA_DATA is None:
            return jsonify({"error": "EDA data not available"}), 500
        
        table_data = EDA_DATA['tables'].get(table_name)
        if table_data is None:
            return jsonify({"error": f"Table {table_name} not found"}), 404
        
        # Convert DataFrame to JSON
        if hasattr(table_data, 'to_dict'):
            return jsonify({
                "data": table_data.to_dict('records'),
                "columns": table_data.columns.tolist(),
                "shape": table_data.shape,
                "status": "success"
            })
        else:
            return jsonify({
                "data": table_data,
                "status": "success"
            })
    except Exception as e:
        logger.error(f"Error serving table {table_name}: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/api/eda-summary/<analysis_name>")
def get_eda_summary(analysis_name):
    """Get specific EDA analysis summary"""
    try:
        if EDA_DATA is None:
            return jsonify({"error": "EDA data not available"}), 500
        
        summary = EDA_DATA['summaries'].get(analysis_name)
        if summary is None:
            return jsonify({"error": f"Analysis {analysis_name} not found"}), 404
        
        return jsonify({
            "summary": summary,
            "status": "success"
        })
    except Exception as e:
        logger.error(f"Error serving summary {analysis_name}: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/api/eda-charts")
def get_eda_charts():
    """Get chart data from EDA analysis for frontend visualization"""
    try:
        if EDA_DATA is None:
            return jsonify({"error": "EDA data not available"}), 500
        
        charts_data = {}
        
        # Country distribution from EDA - use full dataset for comprehensive results
        try:
            # Load the full processed dataset to get all 72 countries
            processed_df = EDA_DATA['tables'].get('processed_spotify_dataset')
            if processed_df is not None and 'country' in processed_df.columns:
                # Get country counts from the full dataset - count unique songs per country
                if 'spotify_id' in processed_df.columns:
                    country_counts = processed_df.groupby('country')['spotify_id'].nunique().sort_values(ascending=False)
                else:
                    # Fallback to row count if spotify_id not available
                    country_counts = processed_df['country'].dropna().value_counts()
                # Get top 20 countries for better visualization
                top_countries_data = country_counts.head(20)
                countries = top_countries_data.index.tolist()
                counts = top_countries_data.values.tolist()
                country_data = list(zip(countries, counts))
                top_countries = country_data
            else:
                # Fallback to plot data (top 15 countries)
                geo_plot_data = EDA_DATA['plot_data'].get('geographical_analysis', {})
                if geo_plot_data and 'country_counts' in geo_plot_data:
                    country_counts = geo_plot_data['country_counts']
                    countries = list(country_counts.keys())
                    counts = list(country_counts.values())
                    country_data = list(zip(countries, counts))
                    top_countries = country_data
                else:
                    # Final fallback: use summary data (top 10 countries)
                    geo_summary = EDA_DATA['summaries'].get('geographical_analysis', {})
                    top_countries_data = geo_summary.get('top_10_countries', {})
                    countries = list(top_countries_data.keys())
                    counts = list(top_countries_data.values())
                    country_data = list(zip(countries, counts))
                    top_countries = country_data[:10]
        except Exception as e:
            logger.error(f"Error processing country data: {str(e)}")
            # Fallback to summary data
            geo_summary = EDA_DATA['summaries'].get('geographical_analysis', {})
            top_countries_data = geo_summary.get('top_10_countries', {})
            countries = list(top_countries_data.keys())
            counts = list(top_countries_data.values())
            country_data = list(zip(countries, counts))
            top_countries = country_data[:10]
            
            charts_data['country_distribution'] = {
                'labels': [get_country_name(country) for country, _ in top_countries],
                'data': [count for _, count in top_countries],
                'counts': [count for _, count in top_countries]
            }
        
        # Temporal trends from EDA
        daily_df = EDA_DATA['tables'].get('daily_song_counts')
        if daily_df is not None:
            # The CSV only contains counts, not dates - create dates from temporal analysis
            temporal_summary = EDA_DATA['summaries'].get('temporal_analysis', {})
            start_date = temporal_summary.get('date_range_start', '2023-10-18')
            # Parse the start date and create date range
            start_date_parsed = pd.to_datetime(start_date)
            dates = pd.date_range(start=start_date_parsed, periods=len(daily_df), freq='D').strftime('%Y-%m-%d').tolist()
            counts = daily_df.iloc[:, 0].tolist() if len(daily_df.columns) > 0 else []
            
            charts_data['temporal_trends'] = {
                'labels': dates,
                'data': counts,
                'counts': counts
            }
        
        # Audio features correlation from EDA
        audio_df = EDA_DATA['tables'].get('audio_features_correlation_matrix')
        if audio_df is not None:
            # Get correlation data for visualization
            features = audio_df.columns.tolist() if hasattr(audio_df, 'columns') else []
            correlation_data = audio_df.values.tolist() if hasattr(audio_df, 'values') else []
            
            charts_data['audio_features'] = {
                'features': features,
                'correlation_matrix': correlation_data
            }
        
        # Explicitness analysis from EDA
        explicitness_summary = EDA_DATA['summaries'].get('explicitness_analysis', {})
        if explicitness_summary:
            charts_data['explicitness_analysis'] = {
                'labels': ['Explicit', 'Non-Explicit'],
                'data': [
                    explicitness_summary.get('explicit_songs_count', 0),
                    explicitness_summary.get('non_explicit_songs_count', 0)
                ],
                'percentages': [
                    explicitness_summary.get('explicit_songs_percentage', 0),
                    explicitness_summary.get('non_explicit_songs_percentage', 0)
                ]
            }
        
        # Basic data overview from EDA
        basic_plot_data = EDA_DATA['plot_data'].get('basic_data_overview', {})
        if basic_plot_data and 'missing_data' in basic_plot_data:
            missing_data = basic_plot_data['missing_data']
            # Get top 10 columns with missing data
            sorted_missing = sorted(missing_data.items(), key=lambda x: x[1], reverse=True)[:10]
            
            charts_data['missing_data_analysis'] = {
                'labels': [col for col, _ in sorted_missing],
                'data': [count for _, count in sorted_missing],
                'counts': [count for _, count in sorted_missing]
            }
        
        # Feature Selection Analysis from EDA
        feature_selection_summary = EDA_DATA['summaries'].get('feature_selection', {})
        if feature_selection_summary:
            charts_data['feature_selection'] = {
                'selectkbest_features': feature_selection_summary.get('selectkbest_top_features', [])[:10],
                'rfe_features': feature_selection_summary.get('rfe_top_features', [])[:10],
                'random_forest_features': feature_selection_summary.get('random_forest_top_features', [])[:10],
                'correlation_features': feature_selection_summary.get('correlation_top_features', [])[:10],
                'total_features_analyzed': feature_selection_summary.get('total_features_analyzed', 0),
                'target_variable': feature_selection_summary.get('target_variable', 'daily_rank')
            }
        
        # Predictive Analysis from EDA
        predictive_summary = EDA_DATA['summaries'].get('predictive_analysis', {})
        if predictive_summary:
            model_performance = predictive_summary.get('model_performance', {})
            charts_data['predictive_analysis'] = {
                'best_model': predictive_summary.get('best_model', 'Random Forest'),
                'models': list(model_performance.keys()),
                'performance_metrics': {
                    'MSE': [model_performance[model].get('MSE', 0) for model in model_performance.keys()],
                    'RMSE': [model_performance[model].get('RMSE', 0) for model in model_performance.keys()],
                    'MAE': [model_performance[model].get('MAE', 0) for model in model_performance.keys()],
                    'R2': [model_performance[model].get('R2', 0) for model in model_performance.keys()]
                },
                'target_variable': predictive_summary.get('target_variable', 'daily_rank'),
                'features_used': predictive_summary.get('features_used', [])[:15],  # Top 15 features
                'train_test_split': predictive_summary.get('train_test_split', {})
            }
        
        # Top Artists Analysis from EDA
        top_artists_df = EDA_DATA['tables'].get('top_artists')
        if top_artists_df is not None:
            # Get top 10 artists
            top_artists_data = top_artists_df.iloc[:10, 0].tolist() if len(top_artists_df) > 0 else []
            artist_counts = top_artists_df.iloc[:10, 0].tolist() if len(top_artists_df) > 0 else []
            
            charts_data['top_artists'] = {
                'labels': [f"Artist_{i+1}" for i in range(len(top_artists_data))],  # Generic labels since artist names aren't in CSV
                'data': artist_counts,
                'counts': artist_counts
            }
        
        # Model Predictions Analysis
        model_predictions_df = EDA_DATA['tables'].get('model_predictions')
        if model_predictions_df is not None and len(model_predictions_df) > 0:
            # Sample predictions for visualization (first 1000 predictions)
            # Use full dataset for model predictions
            charts_data['model_predictions'] = {
                'actual': model_predictions_df.iloc[:, 0].tolist() if len(model_predictions_df.columns) > 0 else [],
                'predicted': model_predictions_df.iloc[:, 1].tolist() if len(model_predictions_df.columns) > 1 else [],
                'residuals': model_predictions_df.iloc[:, 2].tolist() if len(model_predictions_df.columns) > 2 else []
            }
        
        # Weekly/Monthly/Yearly Trends
        weekly_df = EDA_DATA['tables'].get('weekly_song_counts')
        monthly_df = EDA_DATA['tables'].get('monthly_song_counts')
        yearly_df = EDA_DATA['tables'].get('yearly_song_counts')
        
        if weekly_df is not None:
            charts_data['weekly_trends'] = {
                'labels': [f"Week_{i+1}" for i in range(len(weekly_df))],
                'data': weekly_df.iloc[:, 0].tolist() if len(weekly_df.columns) > 0 else [],
                'counts': weekly_df.iloc[:, 0].tolist() if len(weekly_df.columns) > 0 else []
            }
        
        if monthly_df is not None:
            charts_data['monthly_trends'] = {
                'labels': [f"Month_{i+1}" for i in range(len(monthly_df))],
                'data': monthly_df.iloc[:, 0].tolist() if len(monthly_df.columns) > 0 else [],
                'counts': monthly_df.iloc[:, 0].tolist() if len(monthly_df.columns) > 0 else []
            }
        
        if yearly_df is not None:
            charts_data['yearly_trends'] = {
                'labels': [f"Year_{i+1}" for i in range(len(yearly_df))],
                'data': yearly_df.iloc[:, 0].tolist() if len(yearly_df.columns) > 0 else [],
                'counts': yearly_df.iloc[:, 0].tolist() if len(yearly_df.columns) > 0 else []
            }
        
        return jsonify({
            "charts": charts_data,
            "status": "success"
        })
    except Exception as e:
        logger.error(f"Error serving EDA charts: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/api/comprehensive-dashboard")
def get_comprehensive_dashboard():
    """Get comprehensive dashboard data matching the exact structure from comprehensive_dashboard.png"""
    try:
        if EDA_DATA is None:
            return jsonify({"error": "EDA data not available"}), 500
        
        dashboard_data = {
            "title": "Comprehensive Spotify Analytics Dashboard",
            "sections": []
        }
        
        # 1. Basic Data Overview Section (6 subplots)
        basic_plot_data = EDA_DATA['plot_data'].get('basic_data_overview', {})
        if basic_plot_data:
            basic_section = {
                "title": "Basic Data Overview - Spotify Dataset",
                "layout": "2x3",
                "subplots": []
            }
            
            # Missing Values by Column
            if 'missing_data' in basic_plot_data:
                missing_data = basic_plot_data['missing_data']
                sorted_missing = sorted(missing_data.items(), key=lambda x: x[1], reverse=True)[:10]
                basic_section["subplots"].append({
                    "id": "missing_values",
                    "type": "barh",
                    "title": "Missing Values by Column",
                    "labels": [col for col, _ in sorted_missing],
                    "data": [count for _, count in sorted_missing],
                    "xlabel": "Missing Count"
                })
            
            # Data Types Distribution
            basic_summary = EDA_DATA['summaries'].get('basic_data', {})
            if 'data_types' in basic_summary:
                data_types = basic_summary['data_types']
                basic_section["subplots"].append({
                    "id": "data_types",
                    "type": "pie",
                    "title": "Data Types Distribution",
                    "labels": list(data_types.keys()),
                    "data": list(data_types.values())
                })
            
            # Dataset Size Over Time (simulated)
            basic_section["subplots"].append({
                "id": "dataset_size",
                "type": "line",
                "title": "Dataset Size Over Time",
                "labels": ["2023-10", "2023-11", "2023-12", "2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06"],
                "data": [1800000, 1850000, 1900000, 1950000, 2000000, 2050000, 2080000, 2100000, 2110316],
                "xlabel": "Date",
                "ylabel": "Number of Records"
            })
            
            # Distribution of Numeric Columns (Popularity and Energy)
            processed_df = EDA_DATA['tables'].get('processed_spotify_dataset')
            if processed_df is not None:
                # Popularity distribution
                popularity_data = processed_df['popularity'].value_counts().head(20)
                basic_section["subplots"].append({
                    "id": "popularity_dist",
 
                    "type": "hist",
                    "title": "Distribution of Popularity",
                    "labels": list(popularity_data.index),
                    "data": list(popularity_data.values),
                    "xlabel": "Popularity Score",
                    "ylabel": "Frequency"
                })
                
                # Energy distribution
                energy_data = processed_df['energy'].value_counts().head(20)
                basic_section["subplots"].append({
                    "id": "energy_dist",
                    "type": "hist", 
                    "title": "Distribution of Energy",
                    "labels": [f"{x:.2f}" for x in energy_data.index],
                    "data": list(energy_data.values),
                    "xlabel": "Energy Value",
                    "ylabel": "Frequency"
                })
            
            # Top 10 Categories (Countries)
            geo_plot_data = EDA_DATA['plot_data'].get('geographical_analysis', {})
            if geo_plot_data and 'country_counts' in geo_plot_data:
                country_counts = geo_plot_data['country_counts']
                top_countries = list(country_counts.items())[:10]
                basic_section["subplots"].append({
                    "id": "top_categories",
                    "type": "barh",
                    "title": "Top 10 Countries",
                    "labels": [get_country_name(country) for country, _ in top_countries],
                    "data": [count for _, count in top_countries],
                    "xlabel": "Song Count"
                })
            
            dashboard_data["sections"].append(basic_section)
        
        # 2. Geographical Analysis Section (4 subplots)
        geo_plot_data = EDA_DATA['plot_data'].get('geographical_analysis', {})
        if geo_plot_data:
            geo_section = {
                "title": "Geographical Analysis - Music Preferences by Country",
                "layout": "2x2",
                "subplots": []
            }
            
            # Top 15 Countries by Song Count
            if 'country_counts' in geo_plot_data:
                country_counts = geo_plot_data['country_counts']
                top_countries = list(country_counts.items())[:15]
                geo_section["subplots"].append({
                    "id": "top_countries_songs",
                    "type": "barh",
                    "title": "Top 15 Countries by Song Count",
                    "labels": [get_country_name(country) for country, _ in top_countries],
                    "data": [count for _, count in top_countries],
                    "xlabel": "Number of Songs"
                })
            
            # Country Distribution Pie Chart
            if 'country_counts' in geo_plot_data:
                country_counts = geo_plot_data['country_counts']
                top_10 = list(country_counts.items())[:10]
                others_count = sum(country_counts.values()) - sum(count for _, count in top_10)
                
                pie_labels = [get_country_name(country) for country, _ in top_10] + ["Others"]
                pie_data = [count for _, count in top_10] + [others_count]
                
                geo_section["subplots"].append({
                    "id": "country_distribution",
                    "type": "pie",
                    "title": "Country Distribution (Top 10 + Others)",
                    "labels": pie_labels,
                    "data": pie_data
                })
            
            # Distribution of Songs per Country
            if 'country_counts' in geo_plot_data:
                country_counts = geo_plot_data['country_counts']
                song_counts = list(country_counts.values())
                geo_section["subplots"].append({
                    "id": "songs_per_country_dist",
                    "type": "hist",
                    "title": "Distribution of Songs per Country",
                    "labels": [f"{i*1000}-{(i+1)*1000}" for i in range(0, 30, 2)],
                    "data": [song_counts.count(i) for i in range(0, 30000, 2000)],
                    "xlabel": "Number of Songs",
                    "ylabel": "Number of Countries"
                })
            
            # Top 15 Countries by Unique Artists
            if 'unique_artists_per_country' in geo_plot_data:
                artist_counts = geo_plot_data['unique_artists_per_country']
                top_artists = list(artist_counts.items())[:15]
                geo_section["subplots"].append({
                    "id": "top_countries_artists",
                    "type": "barh",
                    "title": "Top 15 Countries by Unique Artists",
                    "labels": [get_country_name(country) for country, _ in top_artists],
                    "data": [count for _, count in top_artists],
                    "xlabel": "Number of Unique Artists"
                })
            
            dashboard_data["sections"].append(geo_section)
        
        # 3. Temporal Analysis Section (4 subplots)
        temporal_plot_data = EDA_DATA['plot_data'].get('temporal_analysis', {})
        if temporal_plot_data:
            temporal_section = {
                "title": "Temporal Analysis - Song Popularity Trends",
                "layout": "2x2",
                "subplots": []
            }
            
            # Daily Song Counts Over Time
            daily_df = EDA_DATA['tables'].get('daily_song_counts')
            if daily_df is not None:
                temporal_summary = EDA_DATA['summaries'].get('temporal_analysis', {})
                start_date = temporal_summary.get('date_range_start', '2023-10-18')
                start_date_parsed = pd.to_datetime(start_date)
                dates = pd.date_range(start=start_date_parsed, periods=len(daily_df), freq='D').strftime('%Y-%m-%d').tolist()
                counts = daily_df.iloc[:, 0].tolist()
                
                temporal_section["subplots"].append({
                    "id": "daily_trends",
                    "type": "line",
                    "title": "Daily Song Counts Over Time",
                    "labels": dates[::10],  # Sample every 10th date for readability
                    "data": counts[::10],
                    "xlabel": "Date",
                    "ylabel": "Number of Songs"
                })
            
            # Weekly Song Distribution
            weekly_df = EDA_DATA['tables'].get('weekly_song_counts')
            if weekly_df is not None:
                temporal_section["subplots"].append({
                    "id": "weekly_distribution",
                    "type": "bar",
                    "title": "Weekly Song Distribution",
                    "labels": [f"Week {i+1}" for i in range(len(weekly_df))],
                    "data": weekly_df.iloc[:, 0].tolist(),
                    "xlabel": "Week Number",
                    "ylabel": "Number of Songs"
                })
            
            # Monthly Song Distribution
            monthly_df = EDA_DATA['tables'].get('monthly_song_counts')
            if monthly_df is not None:
                temporal_section["subplots"].append({
                    "id": "monthly_distribution",
                    "type": "bar",
                    "title": "Monthly Song Distribution",
                    "labels": [f"Month {i+1}" for i in range(len(monthly_df))],
                    "data": monthly_df.iloc[:, 0].tolist(),
                    "xlabel": "Month",
                    "ylabel": "Number of Songs"
                })
            
            # Yearly Song Distribution
            yearly_df = EDA_DATA['tables'].get('yearly_song_counts')
            if yearly_df is not None:
                temporal_section["subplots"].append({
                    "id": "yearly_distribution",
                    "type": "bar",
                    "title": "Yearly Song Distribution",
                    "labels": [f"Year {i+1}" for i in range(len(yearly_df))],
                    "data": yearly_df.iloc[:, 0].tolist(),
                    "xlabel": "Year",
                    "ylabel": "Number of Songs"
                })
            
            dashboard_data["sections"].append(temporal_section)
        
        # 4. Audio Features Analysis Section (4 subplots)
        audio_plot_data = EDA_DATA['plot_data'].get('audio_features_analysis', {})
        if audio_plot_data:
            audio_section = {
                "title": "Audio Features Correlation Analysis",
                "layout": "2x2",
                "subplots": []
            }
            
            # Audio Features Correlation Heatmap
            audio_df = EDA_DATA['tables'].get('audio_features_correlation_matrix')
            if audio_df is not None:
                features = audio_df.columns.tolist()
                correlation_data = audio_df.values.tolist()
                
                audio_section["subplots"].append({
                    "id": "correlation_heatmap",
                    "type": "heatmap",
                    "title": "Audio Features Correlation Heatmap",
                    "labels": features,
                    "data": correlation_data
                })
            
            # Distribution of Danceability
            processed_df = EDA_DATA['tables'].get('processed_spotify_dataset')
            if processed_df is not None:
                danceability_data = processed_df['danceability'].value_counts().head(20)
                audio_section["subplots"].append({
                    "id": "danceability_dist",
                    "type": "hist",
                    "title": "Distribution of Danceability",
                    "labels": [f"{x:.2f}" for x in danceability_data.index],
                    "data": list(danceability_data.values),
                    "xlabel": "Danceability Value",
                    "ylabel": "Frequency"
                })
            
            # Distribution of Valence
            if processed_df is not None:
                valence_data = processed_df['valence'].value_counts().head(20)
                audio_section["subplots"].append({
                    "id": "valence_dist",
                    "type": "hist",
                    "title": "Distribution of Valence",
                    "labels": [f"{x:.2f}" for x in valence_data.index],
                    "data": list(valence_data.values),
                    "xlabel": "Valence Value",
                    "ylabel": "Frequency"
                })
            
            # Feature Correlation Scatter Plot (Energy vs Loudness)
            if processed_df is not None:
                # Use full dataset for scatter plot
                audio_section["subplots"].append({
                    "id": "energy_loudness_scatter",
                    "type": "scatter",
                    "title": "Energy vs Loudness Correlation",
                    "labels": processed_df['energy'].tolist(),
                    "data": processed_df['loudness'].tolist(),
                    "xlabel": "Energy",
                    "ylabel": "Loudness"
                })
            
            dashboard_data["sections"].append(audio_section)
        
        return jsonify({
            "dashboard": dashboard_data,
            "status": "success"
        })
    except Exception as e:
        logger.error(f"Error serving comprehensive dashboard: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    # Production configuration
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    port = int(os.environ.get('PORT', 5001))
    host = os.environ.get('HOST', '127.0.0.1')
    
    logger.info(f"Starting Flask app on {host}:{port} (debug={debug_mode})")
    app.run(debug=debug_mode, port=port, host=host)
