import json

def search(node, query, results):
    name = node.get("data", {}).get("name", "").lower()
    acronym = node.get("data", {}).get("acronym", "").lower()

    if query in name or query in acronym:
        results.append(node)

    for c in node.get("children", []):
        search(c, query, results)


def select_region(json_path):
    with open(json_path) as f:
        data = json.load(f)

    queries = input("输入脑区(英文关键词，多个请用空格隔开): ").split()
    selected = []

    for q in queries:
        results = []
        search(data["root"], q.lower(), results)

        for i, r in enumerate(results):
            d = r["data"]
            print(f"[{i}] {d['name']} ({d['acronym']}) id={r['id']}")

        ans = input("选择编号: ")
        if ans:
            idx = list(map(int,ans.split()))
            selected+=idx
        else:
            continue

    return selected