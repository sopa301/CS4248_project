import csv
import ast

input_csv = 'ELCo.csv'
output_csv = 'dataset_only_true.csv'

# Function to convert emojis to Unicode
def emoji_to_unicode(emoji_str):
    return ' '.join([f"U+{ord(char):X}" for char in emoji_str])

with open(input_csv, newline='', encoding='utf-8') as fin, \
     open(output_csv, 'w', newline='', encoding='utf-8') as fout:

    reader = csv.DictReader(fin)

    # Update fieldnames to include all keys in writer.writerow()
    fieldnames = ['sent1', 'sent2', 'unicode', 'label', 'strategy', 'attribute', 'filename']
    writer = csv.DictWriter(fout, fieldnames=fieldnames)
    writer.writeheader()

    i = 0
    for row in reader:
        desc_list = ast.literal_eval(row['Description'])  # Convert string to list
        desc_processed = ' [EM] '.join(desc.strip(':') for desc in desc_list)
        
        sent1 = f"{desc_processed}."
        sent2 = row['EN']

        label = 1

        unicode_repr = emoji_to_unicode(row['EM'])

        writer.writerow({
            'sent1': sent1,
            'sent2': sent2,
            'unicode': unicode_repr,  # Added this to match fieldnames
            'label': label,
            'strategy': row['Composition strategy'],
            'attribute': row['Attribute'],
            'filename': f"{i}.png"
        })

        i += 1

print(f"Conversion complete! Output saved to {output_csv}")
