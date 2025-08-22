import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
import os
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Create results directory
def create_results_directory():
    """
    Create a results directory to store all outputs
    """
    results_dir = "eda_full_dataset_results"
    plots_dir = os.path.join(results_dir, "plots")
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"Results will be saved in: {results_dir}")
    return results_dir, plots_dir

def load_full_dataset(file_path):
    """
    Load the full CSV file without sampling
    and drop the 'answer_start_my' column if it exists
    """
    print(f"Loading full dataset from {file_path}...")
    
    df = pd.read_csv(file_path)
    print(f"Dataset shape before dropping: {df.shape}")
    
    # Drop 'answer_start_my' if present
    if 'answer_start_my' in df.columns:
        df = df.drop(columns=['answer_start_my'])
        print("Column 'answer_start_my' dropped.")
    
    print(f"Dataset shape after dropping: {df.shape}")
    return df


def analyze_text_lengths(df, column_name, title, plots_dir, dataset_name=""):
    """
    Analyze text length distribution for a specific column and save plots
    """
    # Calculate text lengths
    lengths = df[column_name].astype(str).str.len()
    
    # Create subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histogram
    ax1.hist(lengths, bins=100, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Text Length (characters)')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'{title} - Length Distribution ({dataset_name})')
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2.boxplot(lengths, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
    ax2.set_ylabel('Text Length (characters)')
    ax2.set_title(f'{title} - Length Box Plot ({dataset_name})')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = os.path.join(plots_dir, f"{dataset_name.lower().replace(' ', '_')}_{title.lower().replace(' ', '_')}_length_analysis.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()  # Close the plot to free memory
    
    print(f"Plot saved: {plot_filename}")
    
    # Print statistics
    print(f"\n{title} Length Statistics ({dataset_name}):")
    print(f"Mean: {lengths.mean():.2f}")
    print(f"Median: {lengths.median():.2f}")
    print(f"Std: {lengths.std():.2f}")
    print(f"Min: {lengths.min()}")
    print(f"Max: {lengths.max()}")
    print(f"25th percentile: {lengths.quantile(0.25):.2f}")
    print(f"75th percentile: {lengths.quantile(0.75):.2f}")
    
    return lengths

def analyze_answers(df, plots_dir, dataset_name=""):
    """
    Analyze answer patterns and characteristics and save plots
    """
    print(f"\n=== Answer Analysis ({dataset_name}) ===")
    
    # Analyze answer text lengths
    answer_lengths = []
    for answers in df['answers']:
        try:
            # Parse the answers string to extract text
            if isinstance(answers, str) and 'text' in answers:
                # Extract text from the answers string
                text_match = re.search(r"'text': array\(\[(.*?)\], dtype=object\)", answers)
                if text_match:
                    text_content = text_match.group(1)
                    # Remove quotes and split by comma
                    text_parts = [t.strip().strip("'\"") for t in text_content.split(',')]
                    for text in text_parts:
                        if text and text != '':  # Skip empty strings
                            answer_lengths.append(len(text))
        except:
            continue
    
    if answer_lengths:
        print(f"Number of valid answers analyzed: {len(answer_lengths)}")
        print(f"Answer length statistics:")
        print(f"  Mean: {np.mean(answer_lengths):.2f}")
        print(f"  Median: {np.median(answer_lengths):.2f}")
        print(f"  Min: {np.min(answer_lengths)}")
        print(f"  Max: {np.max(answer_lengths)}")
        
        # Plot answer length distribution and save
        plt.figure(figsize=(10, 6))
        plt.hist(answer_lengths, bins=50, alpha=0.7, color='orange', edgecolor='black')
        plt.xlabel('Answer Length (characters)')
        plt.ylabel('Frequency')
        plt.title(f'Answer Length Distribution ({dataset_name})')
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        plot_filename = os.path.join(plots_dir, f"{dataset_name.lower().replace(' ', '_')}_answer_length_distribution.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Answer length plot saved: {plot_filename}")
        
        return answer_lengths
    else:
        print("No valid answers found for analysis")
        return []

def analyze_myanmar_text(df, dataset_name=""):
    """
    Analyze Myanmar text characteristics
    """
    print(f"\n=== Myanmar Text Analysis ({dataset_name}) ===")
    
    # Check for Myanmar text columns
    myanmar_columns = [col for col in df.columns if 'my' in col.lower()]
    print(f"Myanmar text columns found: {myanmar_columns}")
    
    myanmar_stats = {}
    
    for col in myanmar_columns:
        if col in df.columns:
            # Count non-empty Myanmar text
            non_empty = df[col].notna() & (df[col].astype(str).str.strip() != '')
            print(f"\n{col}:")
            print(f"  Non-empty entries: {non_empty.sum()}")
            print(f"  Empty entries: {(~non_empty).sum()}")
            print(f"  Coverage: {non_empty.sum()/len(df)*100:.2f}%")
            
            # Store stats for reporting
            myanmar_stats[col] = {
                'non_empty': non_empty.sum(),
                'empty': (~non_empty).sum(),
                'coverage': non_empty.sum()/len(df)*100
            }
            
            # Sample some Myanmar text
            sample_texts = df[non_empty][col].head(3)
            print(f"  Sample texts:")
            for i, text in enumerate(sample_texts, 1):
                print(f"    {i}: {str(text)[:100]}...")
    
    return myanmar_stats

def perform_comprehensive_eda(df, plots_dir, dataset_name=""):
    """
    Perform comprehensive exploratory data analysis and save plots
    """
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE EDA - {dataset_name.upper()}")
    print(f"{'='*60}")
    
    # Filter out answer_start column since it won't be used
    columns_to_analyze = [col for col in df.columns if col != 'answer_start_my']
    print(f"\nNote: 'answer_start' column excluded from analysis as it won't be used, since it will not be used in the model trainingg")
    
    # Basic dataset info
    print(f"\nDataset Overview:")
    print(f"Shape: {df.shape}")
    print(f"Columns (excluding answer_start): {columns_to_analyze}")
    print(f"Data types:\n{df[columns_to_analyze].dtypes}")
    
    # Memory usage (excluding answer_start)
    memory_usage = df[columns_to_analyze].memory_usage(deep=True)
    print(f"\nMemory Usage (excluding answer_start):")
    print(f"Total memory: {memory_usage.sum() / 1024**2:.2f} MB")
    print(f"Memory per column:")
    for col, mem in memory_usage.items():
        print(f"  {col}: {mem / 1024**2:.2f} MB")
    
    # Missing values (excluding answer_start)
    print(f"\nMissing Values:")
    missing_data = df[columns_to_analyze].isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing_data,
        'Missing Percentage': missing_percent
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0]
    if len(missing_df) > 0:
        print(missing_df)
    else:
        print("No missing values found!")
    
    # Duplicate analysis
    print(f"\nDuplicate Analysis:")
    print(f"Total duplicates: {df.duplicated().sum()}")
    print(f"Duplicate percentage: {(df.duplicated().sum() / len(df)) * 100:.2f}%")
    
    # Text length analysis for key columns
    print(f"\nText Length Analysis:")
    
    length_stats = {}
    
    # Analyze context lengths
    if 'context' in df.columns:
        length_stats['context'] = analyze_text_lengths(df, 'context', 'Context', plots_dir, dataset_name)
    
    # Analyze question lengths
    if 'question' in df.columns:
        length_stats['question'] = analyze_text_lengths(df, 'question', 'Question', plots_dir, dataset_name)
    
    # Analyze Myanmar context lengths
    if 'context_my' in df.columns:
        length_stats['context_my'] = analyze_text_lengths(df, 'context_my', 'Myanmar Context', plots_dir, dataset_name)
    
    # Analyze Myanmar question lengths
    if 'question_my' in df.columns:
        length_stats['question_my'] = analyze_text_lengths(df, 'question_my', 'Myanmar Question', plots_dir, dataset_name)
    
    # Answer analysis
    answer_lengths = analyze_answers(df, plots_dir, dataset_name)
    if answer_lengths:
        length_stats['answers'] = answer_lengths
    
    # Myanmar text analysis
    myanmar_stats = analyze_myanmar_text(df, dataset_name)
    
    # Title analysis (if available)
    if 'title' in df.columns:
        print(f"\nTitle Analysis:")
        unique_titles = df['title'].nunique()
        print(f"Unique titles: {unique_titles}")
        print(f"Most common titles:")
        title_counts = df['title'].value_counts().head(10)
        for title, count in title_counts.items():
            print(f"  {title}: {count}")
    
    # Correlation analysis for numerical columns (excluding answer_start)
    numerical_cols = df[columns_to_analyze].select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 1:
        print(f"\nNumerical Columns Correlation:")
        correlation_matrix = df[numerical_cols].corr()
        print(correlation_matrix)
        
        # Plot correlation heatmap and save
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title(f'Correlation Matrix of Numerical Columns ({dataset_name})')
        plt.tight_layout()
        
        # Save the plot
        plot_filename = os.path.join(plots_dir, f"{dataset_name.lower().replace(' ', '_')}_correlation_matrix.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Correlation matrix plot saved: {plot_filename}")
    
    return {
        'length_stats': length_stats,
        'myanmar_stats': myanmar_stats,
        'missing_data': missing_data,
        'duplicates': df.duplicated().sum()
    }

def create_comparative_analysis(train_stats, validation_stats, results_dir):
    """
    Create a comparative analysis between train and validation datasets
    """
    print(f"\nCreating comparative analysis...")
    
    report_filename = os.path.join(results_dir, "comparative_analysis.txt")
    
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write("SQuAD Myanmar Dataset - Comparative Analysis Report\n")
        f.write("=" * 60 + "\n\n")
        f.write("Note: 'answer_start' column excluded from analysis as it won't be used\n\n")
        
        f.write("DATASET COMPARISON\n")
        f.write("-" * 30 + "\n")
        
        # Compare basic stats
        f.write("Dataset Sizes:\n")
        f.write(f"  Train: {train_stats.get('dataset_size', 'N/A')}\n")
        f.write(f"  Validation: {validation_stats.get('dataset_size', 'N/A')}\n\n")
        
        # Compare missing values
        f.write("Missing Values Comparison:\n")
        if 'missing_data' in train_stats and 'missing_data' in validation_stats:
            train_missing = train_stats['missing_data']
            val_missing = validation_stats['missing_data']
            
            for col in set(train_missing.index) | set(val_missing.index):
                if col != 'answer_start':  # Skip answer_start column
                    train_count = train_missing.get(col, 0)
                    val_count = val_missing.get(col, 0)
                    f.write(f"  {col}:\n")
                    f.write(f"    Train: {train_count}\n")
                    f.write(f"    Validation: {val_count}\n")
        f.write("\n")
        
        # Compare duplicates
        f.write("Duplicate Analysis:\n")
        f.write(f"  Train duplicates: {train_stats.get('duplicates', 'N/A')}\n")
        f.write(f"  Validation duplicates: {validation_stats.get('duplicates', 'N/A')}\n\n")
        
        # Compare Myanmar text coverage
        f.write("Myanmar Text Coverage Comparison:\n")
        if 'myanmar_stats' in train_stats and 'myanmar_stats' in validation_stats:
            train_my = train_stats['myanmar_stats']
            val_my = validation_stats['myanmar_stats']
            
            for col in set(train_my.keys()) | set(val_my.keys()):
                if col in train_my and col in val_my:
                    f.write(f"  {col}:\n")
                    f.write(f"    Train: {train_my[col]['coverage']:.2f}%\n")
                    f.write(f"    Validation: {val_my[col]['coverage']:.2f}%\n")
        f.write("\n")
        
        # Compare text length statistics
        f.write("Text Length Statistics Comparison:\n")
        if 'length_stats' in train_stats and 'length_stats' in validation_stats:
            train_lengths = train_stats['length_stats']
            val_lengths = validation_stats['length_stats']
            
            for col in set(train_lengths.keys()) | set(val_lengths.keys()):
                if col in train_lengths and col in val_lengths:
                    f.write(f"  {col}:\n")
                    f.write(f"    Train - Mean: {np.mean(train_lengths[col]):.2f}, Median: {np.median(train_lengths[col]):.2f}\n")
                    f.write(f"    Validation - Mean: {np.mean(val_lengths[col]):.2f}, Median: {np.median(val_lengths[col]):.2f}\n")
    
    print(f"Comparative analysis saved: {report_filename}")

def create_summary_report(train_df, validation_df, train_stats, validation_stats, results_dir):
    """
    Create a comprehensive summary report for both datasets
    """
    print(f"\nCreating summary report...")
    
    report_filename = os.path.join(results_dir, "full_dataset_eda_summary.txt")
    
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write("SQuAD Myanmar Dataset - Full Dataset EDA Summary Report\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("DATASET OVERVIEW\n")
        f.write("-" * 20 + "\n")
        f.write(f"Train dataset size: {len(train_df)} rows\n")
        f.write(f"Validation dataset size: {len(validation_df)} rows\n")
        f.write(f"Total combined size: {len(train_df) + len(validation_df)} rows\n")
        f.write(f"Note: 'answer_start' column excluded from analysis as it won't be used\n\n")
        
        f.write("COLUMNS\n")
        f.write("-" * 20 + "\n")
        f.write("Train columns (excluding answer_start):\n")
        for col in train_df.columns:
            if col != 'answer_start':
                f.write(f"  - {col}\n")
        f.write("\nValidation columns (excluding answer_start):\n")
        for col in validation_df.columns:
            if col != 'answer_start':
                f.write(f"  - {col}\n")
        f.write("\n")
        
        f.write("MISSING VALUES\n")
        f.write("-" * 20 + "\n")
        f.write("Train dataset:\n")
        missing_train = train_df.isnull().sum()
        for col in train_df.columns:
            if col != 'answer_start' and missing_train[col] > 0:
                f.write(f"  {col}: {missing_train[col]} ({(missing_train[col]/len(train_df)*100):.2f}%)\n")
        
        f.write("\nValidation dataset:\n")
        missing_val = validation_df.isnull().sum()
        for col in validation_df.columns:
            if col != 'answer_start' and missing_val[col] > 0:
                f.write(f"  {col}: {missing_val[col]} ({(missing_val[col]/len(validation_df)*100):.2f}%)\n")
        f.write("\n")
        
        f.write("TEXT LENGTH STATISTICS\n")
        f.write("-" * 20 + "\n")
        for col in ['context', 'question', 'context_my', 'question_my']:
            if col in train_df.columns:
                f.write(f"{col} - Train:\n")
                lengths = train_df[col].astype(str).str.len()
                f.write(f"  Mean: {lengths.mean():.2f}\n")
                f.write(f"  Median: {lengths.median():.2f}\n")
                f.write(f"  Min: {lengths.min()}\n")
                f.write(f"  Max: {lengths.max()}\n")
            
            if col in validation_df.columns:
                f.write(f"{col} - Validation:\n")
                lengths = validation_df[col].astype(str).str.len()
                f.write(f"  Mean: {lengths.mean():.2f}\n")
                f.write(f"  Median: {lengths.median():.2f}\n")
                f.write(f"  Min: {lengths.min()}\n")
                f.write(f"  Max: {lengths.max()}\n")
            f.write("\n")
        
        f.write("MYANMAR TEXT COVERAGE\n")
        f.write("-" * 20 + "\n")
        myanmar_columns = [col for col in train_df.columns if 'my' in col.lower()]
        for col in myanmar_columns:
            if col in train_df.columns:
                non_empty_train = train_df[col].notna() & (train_df[col].astype(str).str.strip() != '')
                f.write(f"{col} - Train: {non_empty_train.sum()}/{len(train_df)} ({non_empty_train.sum()/len(train_df)*100:.1f}%)\n")
            
            if col in validation_df.columns:
                non_empty_val = validation_df[col].notna() & (validation_df[col].astype(str).str.strip() != '')
                f.write(f"{col} - Validation: {non_empty_val.sum()}/{len(validation_df)} ({non_empty_val.sum()/len(validation_df)*100:.1f}%)\n")
            f.write("\n")
    
    print(f"Summary report saved: {report_filename}")

def main():
    """
    Main function to execute EDA on the full dataset
    """
    print("Starting SQuAD Myanmar Dataset Full Dataset EDA")
    print("="*70)
    
    # Create results directory
    results_dir, plots_dir = create_results_directory()
    
    # File paths
    train_file = "squad_myanmar_train_aligned.csv"
    validation_file = "squad_myanmar_validation_aligned.csv"
    
    try:
        # Step 1: Load full datasets
        print("Loading full datasets...")
        train_df = load_full_dataset(train_file)
        validation_df = load_full_dataset(validation_file)
        
        # Step 2: Perform comprehensive EDA on train dataset
        print(f"\nPerforming EDA on train dataset...")
        train_stats = perform_comprehensive_eda(train_df, plots_dir, "Train Dataset")
        train_stats['dataset_size'] = len(train_df)
        
        # Step 3: Perform comprehensive EDA on validation dataset
        print(f"\nPerforming EDA on validation dataset...")
        validation_stats = perform_comprehensive_eda(validation_df, plots_dir, "Validation Dataset")
        validation_stats['dataset_size'] = len(validation_df)
        
        # Step 4: Create comparative analysis
        create_comparative_analysis(train_stats, validation_stats, results_dir)
        
        # Step 5: Create comprehensive summary report
        create_summary_report(train_df, validation_df, train_stats, validation_stats, results_dir)
        
        print(f"\n" + "="*70)
        print("SUCCESS: Full Dataset EDA completed!")
        print("="*70)
        print(f"Files created in {results_dir}:")
        print(f"  - full_dataset_eda_summary.txt")
        print(f"  - comparative_analysis.txt")
        print(f"  - plots/ (containing all visualization files)")
        print(f"\nTotal samples analyzed:")
        print(f"  - Train: {len(train_df):,} samples")
        print(f"  - Validation: {len(validation_df):,} samples")
        print(f"  - Combined: {len(train_df) + len(validation_df):,} samples")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
