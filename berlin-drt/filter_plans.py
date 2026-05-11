import xml.etree.ElementTree as ET

NETWORK_FILE = "berlin-mitte_network.xml"
INPUT_PLAN = "berlin-mitte_plan_matsim.xml"
OUTPUT_PLAN = "berlin-mitte_plan_matsim_filtered.xml"

def main():
    print(f"Loading valid links from {NETWORK_FILE}...")
    valid_links = set()
    context = ET.iterparse(NETWORK_FILE, events=("end",))
    for event, elem in context:
        if elem.tag == "link":
            valid_links.add(elem.attrib["id"])
            elem.clear()
            
    print(f"Found {len(valid_links)} valid links in the network.")
    
    print(f"Filtering plans from {INPUT_PLAN}...")
    
    tree = ET.parse(INPUT_PLAN)
    root = tree.getroot()
    
    persons_to_remove = []
    
    for person in root.findall("person"):
        keep_person = True
        for ride in person.findall("ride"):
            frm = ride.attrib.get("from")
            to = ride.attrib.get("to")
            if frm not in valid_links or to not in valid_links:
                keep_person = False
                break
                
        if not keep_person:
            persons_to_remove.append(person)
            
    for p in persons_to_remove:
        root.remove(p)
        
    print(f"Removed {len(persons_to_remove)} persons due to missing links.")
    print(f"Writing filtered plans to {OUTPUT_PLAN}...")
    
    tree.write(OUTPUT_PLAN, encoding="UTF-8", xml_declaration=True)
    print("Done!")

if __name__ == "__main__":
    main()
