#!/usr/bin/env python3
"""
Text Cleaning Script for Saint Sophia Graffiti Recognition

This script cleans transcription text according to archeographic principles:

ARCHEOGRAPHIC PRINCIPLES:
1. Lost letters are marked with hyphen "-"
2. Missing beginning/end marked with ellipsis "..."
3. Four types of brackets are used:
   - [] = letters destroyed due to plaster damage
   - () = letters omitted by original authors (abbreviations or errors)
   - // = accidental letter duplications (plaster damage during writing)
   - {} = reconstructed segments/word endings (unfinished inscriptions)
4. Line breaks marked with "|"
5. Words are pre-divided, abbreviated words with titla are expanded

CLEANING STRATEGY:
For training AI models, we need to extract the ACTUAL PRESERVED TEXT:
- Keep the letters that exist on the fresco
- Remove editorial additions/reconstructions
- Remove abbreviation expansions (keep only visible letters)
- Preserve damage indicators for context
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


def extract_preserved_text(text: str, keep_damage_markers: bool = False) -> str:
    """
    Extract only the text that is actually preserved on the fresco.
    
    According to archeographic principles:
    - [] = destroyed letters (REMOVE - not visible)
    - () = omitted letters/abbreviations (REMOVE - not originally written)
    - // = duplicated letters (KEEP only one instance)
    - {} = reconstructed text (REMOVE - not original)
    - - = lost letter marker (REMOVE or KEEP based on keep_damage_markers)
    - ... = missing beginning/end (REMOVE)
    - | = line break (CONVERT to space or newline)
    
    Args:
        text: Text with archeographic notation
        keep_damage_markers: If True, keep "-" and "..." to show damage locations
    
    Returns:
        Only the preserved/visible text
    """
    if not text:
        return ''
    
    # Step 1: Remove reconstructed text in {} brackets
    # {text} -> removed (not original, inferred from context)
    text = re.sub(r'\{[^}]*\}', '', text)
    
    # Step 2: Remove destroyed letters in [] brackets
    # [text] -> removed (destroyed, not visible)
    # But if keep_damage_markers, replace with "-" for each character
    if keep_damage_markers:
        def replace_with_markers(match):
            content = match.group(1)
            return '-' * len(content) if content else '-'
        text = re.sub(r'\[([^\]]*)\]', replace_with_markers, text)
    else:
        text = re.sub(r'\[[^\]]*\]', '', text)
    
    # Step 3: Remove editorial expansions in () brackets
    # (text) -> removed (abbreviations expanded by editors, not originally written)
    text = re.sub(r'\([^)]*\)', '', text)
    
    # Step 4: Handle duplicated letters in // brackets
    # //text// -> keep only one instance (the corrected writing)
    # The text between // is the duplicated part due to plaster damage
    # We keep the text but remove the // markers
    text = re.sub(r'//([^/]+)//', r'\1', text)
    
    # Step 5: Handle damage markers
    if not keep_damage_markers:
        # Remove ellipsis (missing beginning/end)
        text = text.replace('...', '')
        
        # Remove hyphen damage markers (but keep hyphens that are part of words)
        # This is tricky - we need to distinguish between:
        # - Single hyphen as damage marker: "помѧн-"
        # - Multiple hyphens as gap: "---"
        # - Hyphen in compound words: "ни-то"
        
        # Remove sequences of 2+ hyphens (gap markers)
        text = re.sub(r'-{2,}', '', text)
        
        # Remove single hyphens at word boundaries (damage markers)
        # Keep hyphens within words (compound words)
        text = re.sub(r'\s-\s', ' ', text)  # - surrounded by spaces
        text = re.sub(r'^-', '', text)       # - at start
        text = re.sub(r'-$', '', text)       # - at end
        text = re.sub(r'\s-', ' ', text)     # - after space (damaged end)
        text = re.sub(r'-\s', ' ', text)     # - before space (damaged start)
    
    # Step 6: Convert line breaks
    # | -> space (or newline if preserve_lines)
    text = text.replace('|', ' ')
    
    return text


def remove_abbreviation_markers(text: str) -> str:
    """
    Remove abbreviation markers (titla) and other diacritical marks.
    
    According to principles: "abbreviated words marked by titla are expanded in full"
    The expanded text is in (), which we already removed.
    Now we remove the titla marks themselves:
    - ҃ (Cyrillic titlo - indicates abbreviation)
    - ͠ (combining double overline)
    - Other combining marks
    """
    if not text:
        return ''
    
    # Remove titlo and combining marks
    text = text.replace('҃', '')  # Cyrillic titlo
    text = text.replace('͠', '')  # Combining double overline
    text = text.replace('҆', '')  # Cyrillic psili pneumata
    text = text.replace('҇', '')  # Cyrillic pokrytie
    
    # Remove Cyrillic thousand signs
    text = re.sub(r'[ⷀ-ⷿ]', '', text)
    
    # Remove other combining diacritical marks (U+0300 to U+036F)
    text = re.sub(r'[\u0300-\u036F]', '', text)
    
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
        # Collapse everything to single line
        text = text.replace('\n', ' ')
        text = text.replace('\r', ' ')
        text = text.replace('\t', ' ')
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
    
    return text


def clean_graffiti_text(text: str, 
                       keep_damage_markers: bool = False,
                       keep_abbreviations: bool = False, 
                       preserve_lines: bool = True) -> str:
    """
    Clean graffiti transcription text according to archeographic principles.
    
    This extracts only the text that is actually preserved on the fresco:
    - Removes editorial additions: (), [], {}
    - Removes duplicate text: //text// -> text
    - Optionally removes damage markers: -, ...
    - Removes abbreviation markers: titla, combining marks
    - Converts line breaks: | -> space
    
    Args:
        text: Raw transcription text with archeographic notation
        keep_damage_markers: If True, preserve "-" and "..." to show damage
        keep_abbreviations: If True, preserve abbreviation marks like ҃, ͠
        preserve_lines: If True, preserve line breaks from <p> tags
        
    Returns:
        Clean text - only the preserved/visible letters from the fresco
    """
    if pd.isna(text) or text == '':
        return ''
    
    # Step 1: Remove HTML tags first
    text = remove_html_tags(text, preserve_paragraphs=preserve_lines)
    
    # Step 2: Extract only preserved text (remove editorial additions)
    text = extract_preserved_text(text, keep_damage_markers=keep_damage_markers)
    
    # Step 3: Remove abbreviation markers (unless we want to keep them)
    if not keep_abbreviations:
        text = remove_abbreviation_markers(text)
    
    # Step 4: Normalize whitespace
    text = normalize_whitespace(text, preserve_lines=preserve_lines)
    
    return text


def clean_dataset(input_csv: str, output_csv: str, text_column: str = 'transcription'):
    """
    Clean transcription text in a dataset CSV file.
    
    Creates TWO versions of cleaned text:
    1. transcription_clean: Only preserved text (for AI training)
    2. transcription_with_damage: Preserved text + damage markers (for analysis)
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file with cleaned text
        text_column: Name of column containing transcription text
    """
    print(f"="*80)
    print(f"CLEANING GRAFFITI TEXT - ARCHEOGRAPHIC PRINCIPLES")
    print(f"="*80)
    print(f"Input: {input_csv}")
    print(f"Output: {output_csv}")
    print(f"Text column: {text_column}")
    print(f"="*80)
    
    # Load dataset
    df = pd.read_csv(input_csv)
    print(f"\nLoaded {len(df)} inscriptions")
    
    # Count inscriptions with text
    has_text = df[text_column].notna() & (df[text_column] != '')
    print(f"With {text_column}: {has_text.sum()} ({has_text.sum()/len(df)*100:.1f}%)")
    
    # Clean the text - TWO versions
    print(f"\nCleaning {text_column}...")
    print(f"  Creating clean version (preserved text only)...")
    df[f'{text_column}_clean'] = df[text_column].apply(
        lambda x: clean_graffiti_text(x, keep_damage_markers=False, preserve_lines=True)
    )
    
    print(f"  Creating version with damage markers...")
    df[f'{text_column}_with_damage'] = df[text_column].apply(
        lambda x: clean_graffiti_text(x, keep_damage_markers=True, preserve_lines=True)
    )
    
    # Also clean interpretative_edition if it exists
    if 'interpretative_edition' in df.columns:
        print(f"Cleaning interpretative_edition...")
        df['interpretative_edition_clean'] = df['interpretative_edition'].apply(
            lambda x: clean_graffiti_text(x, keep_damage_markers=False, preserve_lines=True)
        )
    
    # Also clean romanisation if it exists
    if 'romanisation' in df.columns:
        print(f"Cleaning romanisation...")
        df['romanisation_clean'] = df['romanisation'].apply(
            lambda x: clean_graffiti_text(x, keep_damage_markers=False, preserve_lines=True)
        )
    
    # Statistics
    print(f"\n{'='*80}")
    print(f"CLEANING RESULTS")
    print(f"{'='*80}")
    
    # Count non-empty cleaned text
    has_clean_text = df[f'{text_column}_clean'] != ''
    print(f"Non-empty after cleaning: {has_clean_text.sum()} ({has_clean_text.sum()/len(df)*100:.1f}%)")
    
    # Show examples with archeographic notation explanation
    print(f"\n{'='*80}")
    print(f"EXAMPLES - ARCHEOGRAPHIC NOTATION")
    print(f"{'='*80}")
    print(f"Legend:")
    print(f"  [] = destroyed letters (plaster damage)")
    print(f"  () = omitted letters (abbreviations/errors)")
    print(f"  // = duplicate letters (corrected writing)")
    print(f"  {{}} = reconstructed text (editorial inference)")
    print(f"  -  = lost letter marker")
    print(f"  .. = missing beginning/end")
    print(f"  |  = line break")
    print(f"{'='*80}")
    
    sample_df = df[has_text].head(10)
    for idx, row in sample_df.iterrows():
        original = str(row[text_column])[:150]
        cleaned = str(row[f'{text_column}_clean'])[:150]
        with_damage = str(row[f'{text_column}_with_damage'])[:150]
        
        print(f"\nID {row['id']}:")
        print(f"  ORIGINAL:     {original}")
        print(f"  CLEAN:        {cleaned}")
        print(f"  WITH DAMAGE:  {with_damage}")
    
    # Character statistics
    print(f"\n{'='*80}")
    print(f"CHARACTER STATISTICS")
    print(f"{'='*80}")
    
    # Average length before/after
    avg_original = df[has_text][text_column].str.len().mean()
    avg_clean = df[has_clean_text][f'{text_column}_clean'].str.len().mean()
    avg_damage = df[has_clean_text][f'{text_column}_with_damage'].str.len().mean()
    
    print(f"Average length - original (with notation): {avg_original:.1f} characters")
    print(f"Average length - clean (preserved only):   {avg_clean:.1f} characters")
    print(f"Average length - with damage markers:      {avg_damage:.1f} characters")
    print(f"\nReduction from original to clean: {avg_original - avg_clean:.1f} chars ({(avg_original - avg_clean)/avg_original*100:.1f}%)")
    
    # Analyze what was removed
    print(f"\n{'='*80}")
    print(f"ARCHEOGRAPHIC NOTATION STATISTICS")
    print(f"{'='*80}")
    
    def count_notation(text):
        """Count instances of archeographic notation."""
        if pd.isna(text) or text == '':
            return {}
        text = str(text)
        return {
            'destroyed []': len(re.findall(r'\[[^\]]*\]', text)),
            'omitted ()': len(re.findall(r'\([^)]*\)', text)),
            'duplicated //': len(re.findall(r'//[^/]+//', text)),
            'reconstructed {}': len(re.findall(r'\{[^}]*\}', text)),
            'damage markers -': text.count('-'),
            'missing parts ...': text.count('...'),
            'line breaks |': text.count('|'),
        }
    
    notation_stats = df[has_text][text_column].apply(count_notation)
    notation_df = pd.DataFrame(notation_stats.tolist())
    
    for col in notation_df.columns:
        total = notation_df[col].sum()
        avg = notation_df[col].mean()
        percent = (notation_df[col] > 0).sum() / len(notation_df) * 100
        print(f"{col:20s}: {total:5.0f} total, {avg:4.1f} avg, {percent:4.1f}% inscriptions")
    
    # Save cleaned dataset
    df.to_csv(output_csv, index=False)
    print(f"\n✓ Saved cleaned dataset: {output_csv}")
    
    print(f"\n{'='*80}")
    print(f"COLUMNS CREATED:")
    print(f"{'='*80}")
    print(f"  {text_column}_clean         : Preserved text only (for AI training)")
    print(f"  {text_column}_with_damage   : Preserved text + damage markers (for analysis)")
    
    return df


def main():
    """Main function to clean graffiti text in dataset."""
    
    # Clean the main inscriptions dataset
    input_file = '../data/inscriptions_graffiti_20251014_124018.csv'
    output_file = '../data/inscriptions_graffiti_cleaned.csv'
    
    if os.path.exists(input_file):
        clean_dataset(input_file, output_file, text_column='transcription')
        
        print(f"\n{'='*80}")
        print(f"✓ TEXT CLEANING COMPLETE")
        print(f"{'='*80}")
        print(f"\nWhat was done:")
        print(f"  ✓ Removed HTML tags")
        print(f"  ✓ Extracted preserved text (removed editorial additions)")
        print(f"  ✓ Removed abbreviation markers (titla)")
        print(f"  ✓ Converted line breaks (|)")
        print(f"  ✓ Created TWO versions: clean and with damage markers")
        print(f"\nNext steps:")
        print(f"1. Review cleaned text in: {output_file}")
        print(f"2. Use 'transcription_clean' column for AI training")
        print(f"3. Use 'transcription_with_damage' for damage analysis")
        print(f"4. Re-run create_datasets.py with cleaned data")
    else:
        print(f"✗ Input file not found: {input_file}")
        print(f"\nPlease ensure the inscriptions CSV file exists at:")
        print(f"  {input_file}")


if __name__ == '__main__':
    main()