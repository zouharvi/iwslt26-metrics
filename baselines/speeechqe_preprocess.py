import json
import os
import pandas as pd

inputfile = "data/iwslt26/dev.jsonl"
output_file = "data/iwslt26/dev.tsv"

def jsonl_to_tsv_dynamic(input_file, output_file, split_name="iwslt26.dev"):
  """Convert JSONL file to TSV format with dynamic language pair handling"""

  # Load JSONL data
  data = []
  with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
      data.append(json.loads(line.strip()))

  print(f"Loaded {len(data)} rows from {input_file}")

  # Language-specific instructions and suffix formats
  lang_config = {
      'de': {
          'inst':
              "Given the German translation of the speech, estimate the quality of the translation as a score between 0 to 1.",
          'suffix_format':
              "\nGerman translation: {x}"
      },
      'zh': {
          'inst':
              "Given the Chinese translation of the speech, estimate the quality of the translation as a score between 0 to 1.",
          'suffix_format':
              "\nChinese translation: {x}"
      }
  }

  # Create DataFrame with new column structure
  rows = []
  for item in data:
    tgt_lang = item['tgt_lang']
    config = lang_config.get(
        tgt_lang, lang_config['de'])  # Default to German if not found

    row = {
        'path': item['audio_path'],
        'sentence': str(int(item['score'])),
        'split': split_name,
        'lang': item['src_lang'],
        'task': f"qe.{item['src_lang']}2{item['tgt_lang']}",
        'inst': config['inst'],
        'suffix': config['suffix_format'].format(x=item['tgt_text']),
        'st_system': item['tgt_system'],
        'humanda': str(int(item['score']))
    }
    rows.append(row)
    if len(rows) % 1000 == 0:
      print(
          f"Processed {len(rows)} rows. Current instruction:\n{config['inst']}\n{config['suffix_format'].format(x=item['tgt_text'])}\n"
      )

  # Create DataFrame
  df = pd.DataFrame(rows)

  # Save to TSV
  df.to_csv(output_file, sep='\t', index=False)
  print(f"✓ Saved TSV file: {output_file}")
  print(f"  Rows: {len(df)}")
  print(f"  Columns: {list(df.columns)}")
  print(f"\nFirst row sample:")
  print(df.head(1).to_string())

  return df

df_endezh = jsonl_to_tsv_dynamic(inputfile, output_file)
