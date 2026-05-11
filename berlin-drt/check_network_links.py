import xml.etree.ElementTree as ET
import re

NETWORK_FILE = "berlin-mitte_network.xml"
PLAN_FILE = "berlin-mitte_plan_matsim_filtered.xml"

def main():
    print(f"Parsing {NETWORK_FILE} for link IDs...")
    network_links = set()
    context = ET.iterparse(NETWORK_FILE, events=("end",))
    for event, elem in context:
        if elem.tag == "link":
            network_links.add(elem.attrib["id"])
            elem.clear()
            
    print(f"Found {len(network_links)} links in the network.")
    
    print(f"Parsing {PLAN_FILE} for used links...")
    plan_links = set()
    with open(PLAN_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if '<ride ' in line:
                m_from = re.search(r'from="([^"]+)"', line)
                m_to = re.search(r'to="([^"]+)"', line)
                if m_from:
                    plan_links.add(m_from.group(1))
                if m_to:
                    plan_links.add(m_to.group(1))
                    
    print(f"Found {len(plan_links)} unique links used in the plans.")
    
    missing_links = plan_links - network_links
    
    if len(missing_links) == 0:
        print("SUCCESS: All links used in the plans are present in the network file!")
    else:
        print(f"WARNING: Found {len(missing_links)} links in the plans that are MISSING from the network file.")
        print("First 20 missing links:")
        for i, link in enumerate(list(missing_links)[:20]):
            print(f"  - {link}")
            
if __name__ == "__main__":
    main()
