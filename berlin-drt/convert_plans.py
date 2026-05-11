import csv
import re

MAPPING_CSV = "network_mapping.csv"
INPUT_XML = "berlin-mitte_plan_sumo.xml"
OUTPUT_XML = "berlin-mitte_plan_matsim.xml"

def main():
    print(f"Loading mapping from {MAPPING_CSV}...")
    edge_mapping = {}
    with open(MAPPING_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['sumo_type'] == 'edge':
                edge_mapping[row['sumo_id']] = row['matsim_id']
                
    print(f"Loaded {len(edge_mapping)} edge mappings.")
    
    print(f"Processing {INPUT_XML}...")
    
    def replacer_from(match):
        sumo_id = match.group(1)
        matsim_id = edge_mapping.get(sumo_id, sumo_id) # fallback to original if not found
        return f'from="{matsim_id}"'
        
    def replacer_to(match):
        sumo_id = match.group(1)
        matsim_id = edge_mapping.get(sumo_id, sumo_id)
        return f'to="{matsim_id}"'

    with open(INPUT_XML, 'r', encoding='utf-8') as infile, \
         open(OUTPUT_XML, 'w', encoding='utf-8') as outfile:
         
        for line in infile:
            if '<ride ' in line:
                line = re.sub(r'from="([^"]+)"', replacer_from, line)
                line = re.sub(r'to="([^"]+)"', replacer_to, line)
            outfile.write(line)
            
    print(f"Finished writing converted plans to {OUTPUT_XML}")

if __name__ == "__main__":
    main()
