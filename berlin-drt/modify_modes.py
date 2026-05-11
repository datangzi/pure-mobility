import random

FILE = "berlin-mitte_plan_matsim_filtered.xml"

def main():
    print(f"Modifying {FILE}...")
    
    with open(FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    counts = {'drt_A': 0, 'drt_B': 0}
        
    with open(FILE, 'w', encoding='utf-8') as f:
        for line in lines:
            if 'lines="taxi"' in line:
                mode = 'drt_A' if random.random() < 0.5 else 'drt_B'
                counts[mode] += 1
                line = line.replace('lines="taxi"', f'mode="{mode}"')
            f.write(line)
            
    print(f"Done! Overwritten {FILE}.")
    print(f"Assigned drt_A to {counts['drt_A']} rides and drt_B to {counts['drt_B']} rides.")

if __name__ == "__main__":
    main()
