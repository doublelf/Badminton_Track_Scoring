import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "out/demo2/scoreboard.json"
with open(path, encoding="utf-8") as handle:
    data = json.load(handle)

print(f"points={data['points']} games={data['games']} serving={data['serving_team']}")
for decision in data["decisions"]:
    court = decision["landing_court"]
    court_str = f"({court[0]:6.2f},{court[1]:5.2f})" if court else "None"
    print(
        f"frame={decision['frame']:>5} {decision['winner_name']} +1 "
        f"{decision['reason']:<22} in_bounds={decision['in_bounds']} court={court_str} "
        f"rule={decision['rule']} conf={decision['confidence']:.2f}"
    )
