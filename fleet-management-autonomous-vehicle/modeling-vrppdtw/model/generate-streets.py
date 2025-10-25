import pandas as pd

edges = []

def node_id(x,y):
    return 10*x + y

for x in range(1,6):
    for y in range(1,6):
        if y < 5:
            a = node_id(x, y)
            b = node_id(x, y+1)
            if x == 1 or x == 5:
                edges.append((a, b, 1, 1))
                edges.append((b, a, 1, 1))
            else:
                edges.append((a, b, 5, 1))
                edges.append((b, a, 5, 1))
        if x < 5:
            a = node_id(x, y)
            b = node_id(x+1, y)
            if y == 1 or y == 5:
                edges.append((a, b, 1, 1))
                edges.append((b, a, 1, 1))
            else:
                edges.append((a, b, 5, 1))
                edges.append((b, a, 5, 1))

df = pd.DataFrame(edges, columns=["from_node", "to_node", "travel_time", "cost"])
df.to_csv("streets_test.csv", index=False)