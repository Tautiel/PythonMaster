def allocation_combinations(n_asset: int, steps: int = 4, tol: float = 1e-6):
    import itertools as it
    weights = [i / steps for i in range(steps + 1)]
    for combo in it.product(weights, repeat = n_asset):
        if abs(sum(combo) -1.0) < tol:
            yield combo
            
assets = ['AAPL', 'GOOGL', 'MSFT']
allocations = list(allocation_combinations(n_asset=3, steps=4))

print(f"{len(allocations)} allocazioni valide per 3 asset:")
for alloc in allocations[:5]:
    print(" ", dict(zip(assets, alloc)))
print(f"   ...({len(allocations) - 5} altre)")
