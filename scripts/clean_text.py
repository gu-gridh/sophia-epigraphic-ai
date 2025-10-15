#!/usr/bin/env python3
"""
Text Cleaning Script for Saint Sophia Graffiti Recognition

This script cleans transcription text by:
1. Removing HTML tags
2. Removing editorial symbols: (), [], {}, <>, (...)
3. Removing punctuation: comma, ?, +, quotes, dash, underscore
4. Removing pipe notation (| means new line, but we remove it)
5. Cleaning whitespace and special characters

The goal is to extract the clean graffiti text suitable for OCR/HTR training.
"""

import pandas as pd
import re
import os
from typing import Optional


def remove_html_tags(text: str, preserve_paragraphs: bool = True) -> str:
    """
    Remove all HTML tags including <p>, <span>, etc.
    
    Args:
        text: Text with HTML tags
        preserve_paragraphs: If True, replace <p> tags with newlines to preserve line structure
    """
    if pd.isna(text) or text == '':
        return ''
    
    # Convert to string
    text = str(text)
    
    # If preserving paragraphs, replace closing </p> with newline before removing tags
    if preserve_paragraphs:
        text = re.sub(r'</p>\s*', '\n', text)
        text = re.sub(r'<p[^>]*>', '', text)  # Remove opening <p> tags (with any attributes)
    
    # Remove all other HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove HTML entities
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&quot;', '"')
    
    return text


def remove_abbreviation_markers(text: str) -> str:
    """
    Remove abbreviation markers commonly used in paleography:
    - ҃ (Cyrillic titlo - indicates abbreviation)
    - ͠ (combining double overline)
    - ⷠ, ⷡ, ⷢ, etc. (Cyrillic thousand signs)
    - Other combining diacritical marks used for abbreviations
    """
    if not text:
        return ''
    
    # Remove titlo and combining marks
    text = text.replace('҃', '')  # Cyrillic titlo
    text = text.replace('͠', '')  # Combining double overline
    text = text.replace('҆', '')  # Cyrillic psili pneumata
    text = text.replace('҇', '')  # Cyrillic pokrytie
    
    # Remove Cyrillic thousand signs
    text = re.sub(r'[ⷀ-ⷿ]', '', text)  # Cyrillic extended-C block (includes thousand signs)
    
    # Remove other combining diacritical marks (U+0300 to U+036F)
    text = re.sub(r'[\u0300-\u036F]', '', text)
    
    return text


def remove_editorial_symbols(text: str) -> str:
    """
    Remove editorial symbols and their content:
    - (...) - indicates missing/damaged text
    - (text) - editorial expansion
    - [text] - editorial correction
    - {text} - editorial deletion
    - <text> - editorial addition
    
    We remove both the brackets/parentheses AND the content inside.
    """
    if not text:
        return ''
    
    # Remove (...) pattern (ellipsis indicating damage)
    text = re.sub(r'\(\.\.\.\)', '', text)
    
    # Remove content in parentheses: (text) -> removed entirely
    text = re.sub(r'\([^)]*\)', '', text)
    
    # Remove content in square brackets: [text] -> removed entirely
    text = re.sub(r'\[[^\]]*\]', '', text)
    
    # Remove content in curly brackets: {text} -> removed entirely
    text = re.sub(r'\{[^}]*\}', '', text)
    
    # Remove content in angle brackets: <text> -> removed entirely (after HTML is removed)
    # This catches any remaining angle brackets
    text = re.sub(r'<[^>]*>', '', text)
    
    return text


def remove_punctuation(text: str) -> str:
    """
    Remove unnecessary punctuation for OCR training:
    - Commas (,)
    - Question marks (?)
    - Plus signs (+)
    - Quotes (', ", «, »)
    - Dashes (-, –, —)
    - Underscores (_)
    - Periods (.) except in abbreviations
    """
    if not text:
        return ''
    
    # Remove common punctuation
    text = text.replace(',', '')
    text = text.replace('?', '')
    text = text.replace('+', '')
    text = text.replace('_', '')
    
    # Remove various types of quotes
    text = text.replace('"', '')
    text = text.replace("'", '')
    text = text.replace('«', '')
    text = text.replace('»', '')
    text = text.replace('"', '')
    text = text.replace('"', '')
    text = text.replace("'", '')
    text = text.replace("'", '')
    
    # Remove various types of dashes (but keep regular hyphen for compound words)
    text = text.replace('–', '')  # en dash
    text = text.replace('—', '')  # em dash
    text = text.replace('―', '')  # horizontal bar
    
    # Remove periods that are likely punctuation (not abbreviation marks)
    # Keep dots that are part of abbreviation marks (like Г͠ or г͠)
    # This is tricky - for now, remove standalone periods
    text = re.sub(r'\.(?=\s|$)', '', text)
    
    return text


def remove_pipe_notation(text: str) -> str:
    """
    Remove pipe character (|) which indicates line breaks in transcriptions.
    We remove it to get continuous text.
    """
    if not text:
        return ''
    
    # Replace pipe with space to avoid concatenating words
    text = text.replace('|', ' ')
    
    return text


def remove_gap_markers(text: str) -> str:
    """
    Remove gap markers like ---, ----, etc. that indicate damaged/illegible text.
    These are sequences of 3 or more dashes used to show gaps.
    Keep single dashes (-) as they may be part of compound words like "ни-".
    """
    if not text:
        return ''
    
    # Remove sequences of 3 or more dashes
    text = re.sub(r'-{3,}', '', text)
    
    # Keep single dash and double dash (they may be intentional)
    
    return text


def normalize_whitespace(text: str, preserve_lines: bool = True) -> str:
    """
    Normalize whitespace: remove extra spaces, tabs.
    
    Args:
        text: Text to normalize
        preserve_lines: If True, keep newlines (for multi-line inscriptions)
    """
    if not text:
        return ''
    
    if preserve_lines:
        # Split by newlines, clean each line, rejoin
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            # Remove tabs and replace with space
            line = line.replace('\t', ' ')
            # Remove multiple spaces
            line = re.sub(r' +', ' ', line)
            # Strip leading/trailing whitespace from each line
            line = line.strip()
            # Only keep non-empty lines
            if line:
                cleaned_lines.append(line)
        
        # Join with newline
        text = '\n'.join(cleaned_lines)
    else:
        # Original behavior - collapse everything to single line
        text = text.replace('\n', ' ')
        text = text.replace('\r', ' ')
        text = text.replace('\t', ' ')
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
    
    return text


def clean_graffiti_text(text: str, keep_abbreviations: bool = False, preserve_lines: bool = True) -> str:
    """
    Complete cleaning pipeline for graffiti transcription text.
    
    Args:
        text: Raw transcription text with HTML, editorial symbols, etc.
        keep_abbreviations: If True, preserve abbreviation marks like ҃, ͠, etc.
                           (Default False - we remove them for clean training text)
        preserve_lines: If True, preserve line breaks from <p> tags
                       (Default True - important for multi-line inscriptions)
        
    Returns:
        Clean text suitable for OCR/HTR training
    """
    if pd.isna(text) or text == '':
        return ''
    
    # Step 1: Remove HTML tags (must be first to avoid issues with < >)
    # This also converts </p> to newlines if preserve_lines=True
    text = remove_html_tags(text, preserve_paragraphs=preserve_lines)
    
    # Step 2: Remove editorial symbols and their content
    text = remove_editorial_symbols(text)
    
    # Step 3: Remove abbreviation markers (unless we want to keep them)
    if not keep_abbreviations:
        text = remove_abbreviation_markers(text)
    
    # Step 4: Remove pipe notation (line breaks)
    text = remove_pipe_notation(text)
    
    # Step 5: Remove gap markers (---) but keep single dashes
    text = remove_gap_markers(text)
    
    # Step 6: Remove unnecessary punctuation
    text = remove_punctuation(text)
    
    # Step 7: Normalize whitespace (preserve newlines if requested)
    text = normalize_whitespace(text, preserve_lines=preserve_lines)
    
    return text


def clean_dataset(input_csv: str, output_csv: str, text_column: str = 'transcription'):
    """
    Clean transcription text in a dataset CSV file.
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file with cleaned text
        text_column: Name of column containing transcription text
    """
    print(f"="*60)
    print(f"CLEANING GRAFFITI TEXT")
    print(f"="*60)
    print(f"Input: {input_csv}")
    print(f"Output: {output_csv}")
    print(f"Text column: {text_column}")
    print(f"="*60)
    
    # Load dataset
    df = pd.read_csv(input_csv)
    print(f"\nLoaded {len(df)} inscriptions")
    
    # Count inscriptions with text
    has_text = df[text_column].notna() & (df[text_column] != '')
    print(f"With {text_column}: {has_text.sum()} ({has_text.sum()/len(df)*100:.1f}%)")
    
    # Clean the text
    print(f"\nCleaning {text_column}...")
    df[f'{text_column}_clean'] = df[text_column].apply(clean_graffiti_text)
    
    # Also clean interpretative_edition if it exists
    if 'interpretative_edition' in df.columns:
        print(f"Cleaning interpretative_edition...")
        df['interpretative_edition_clean'] = df['interpretative_edition'].apply(clean_graffiti_text)
    
    # Also clean romanisation if it exists
    if 'romanisation' in df.columns:
        print(f"Cleaning romanisation...")
        df['romanisation_clean'] = df['romanisation'].apply(clean_graffiti_text)
    
    # Statistics
    print(f"\n{'='*60}")
    print(f"CLEANING RESULTS")
    print(f"{'='*60}")
    
    # Count non-empty cleaned text
    has_clean_text = df[f'{text_column}_clean'] != ''
    print(f"Non-empty after cleaning: {has_clean_text.sum()} ({has_clean_text.sum()/len(df)*100:.1f}%)")
    
    # Show examples
    print(f"\n{'='*60}")
    print(f"BEFORE AND AFTER EXAMPLES")
    print(f"{'='*60}")
    
    sample_df = df[has_text].head(10)
    for idx, row in sample_df.iterrows():
        original = str(row[text_column])[:100]
        cleaned = str(row[f'{text_column}_clean'])[:100]
        
        print(f"\nID {row['id']}:")
        print(f"  BEFORE: {original}")
        print(f"  AFTER:  {cleaned}")
    
    # Character statistics
    print(f"\n{'='*60}")
    print(f"CHARACTER STATISTICS")
    print(f"{'='*60}")
    
    # Average length before/after
    avg_before = df[has_text][text_column].str.len().mean()
    avg_after = df[has_clean_text][f'{text_column}_clean'].str.len().mean()
    
    print(f"Average length before cleaning: {avg_before:.1f} characters")
    print(f"Average length after cleaning: {avg_after:.1f} characters")
    print(f"Average reduction: {avg_before - avg_after:.1f} characters ({(avg_before - avg_after)/avg_before*100:.1f}%)")
    
    # Save cleaned dataset
    df.to_csv(output_csv, index=False)
    print(f"\n✓ Saved cleaned dataset: {output_csv}")
    
    return df


def main():
    """Main function to clean graffiti text in dataset."""
    
    # Clean the main inscriptions dataset
    input_file = '../data/inscriptions_graffiti_20251014_124018.csv'
    output_file = '../data/inscriptions_graffiti_cleaned.csv'
    
    if os.path.exists(input_file):
        clean_dataset(input_file, output_file, text_column='transcription')
        
        print(f"\n{'='*60}")
        print(f"✓ TEXT CLEANING COMPLETE")
        print(f"{'='*60}")
        print(f"\nNext steps:")
        print(f"1. Review cleaned text in: {output_file}")
        print(f"2. Use cleaned version to create comprehensive datasets")
        print(f"3. Re-run create_datasets.py with cleaned data")
    else:
        print(f"✗ Input file not found: {input_file}")


if __name__ == '__main__':
    main()
