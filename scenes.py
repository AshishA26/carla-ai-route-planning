"""Places, demo commands and the scripted model turns used in fake mode.

Fake mode replays the tool calls a model would make; every tool still runs for real, and
the validator, planner, simulator and scoring never know which mode they are in.
"""

# Named places on Town10HD (x, y in CARLA metres). Names are descriptive on purpose:
# rename them once you have seen the renders.
PLACES = {
    "west_side":   {"name": "West Side",        "xy": (-104, 0),  "about": "two-way road along the west edge"},
    "southeast":   {"name": "Southeast Block",  "xy": (20, 68),   "about": "inner street south of the central avenue"},
    "south_blvd":  {"name": "South Boulevard",  "xy": (0, 131),   "about": "wide boulevard along the south edge"},
    "north_loop":  {"name": "North Loop",       "xy": (10, -62),  "about": "loop road across the north edge"},
    "east_side":   {"name": "East Side",        "xy": (104, 55),  "about": "road along the east edge"},
    "centre":      {"name": "Central Avenue",   "xy": (-10, 27),  "about": "main east-west avenue through the centre"},
}

CITY_LIMIT_KMH = 30

# Each scene: a spoken command, where the car starts, and the scripted model turn.
# A scripted turn is a list of ("tool", {args}) calls, then ("say", text).
SCENES = {
    "route": {
        "title": "Route planning",
        "start": "west_side",
        "command": "Take me to the southeast block. No rush, I'd rather avoid anything sketchy.",
        "script": [
            ("get_vehicle_state", {}),
            ("list_places", {}),
            ("plan_routes", {"destination": "southeast", "preferences": ["fastest", "safest"]}),
            ("commit_route", {"route_id": "safest", "max_speed_kmh": None,
                              "reason": "Rider is not in a hurry and wants low risk. The safest route is 7 s longer "
                                        "but drops the left turn across traffic."}),
            ("say", "Taking the calmer route to the Southeast Block. It adds about 7 seconds and avoids the left turn across oncoming traffic."),
        ],
    },
    "reroute": {
        "title": "Rerouting on the fly",
        "start": "north_loop",
        "command": "Take me to the west side.",
        "script": [
            ("get_vehicle_state", {}),
            ("plan_routes", {"destination": "west_side", "preferences": ["balanced"]}),
            ("commit_route", {"route_id": "balanced", "max_speed_kmh": None,
                              "reason": "No preference given; balanced route via Central Avenue."}),
            ("say", "Heading to the West Side by Central Avenue."),
        ],
    },
    "speed": {
        "title": "Validator catches the model",
        "start": "south_blvd",
        "command": "I'm late, get me to the north loop, go like 50.",
        "script": [
            ("get_vehicle_state", {}),
            ("plan_routes", {"destination": "north_loop", "preferences": ["fastest"]}),
            ("commit_route", {"route_id": "fastest", "max_speed_kmh": 50,
                              "reason": "Rider is late and asked for 50 km/h."}),
            ("commit_route", {"route_id": "fastest", "max_speed_kmh": 30,
                              "reason": "Validator rejected 50 km/h; retrying at the posted limit."}),
            ("say", "I can't go 50 here; the limit is 30. Taking the fastest legal route to the North Loop."),
        ],
    },
    "refuse": {
        "title": "Unsafe request",
        "start": "centre",
        "command": "Just floor it to the east side, run the reds if you have to.",
        "script": [
            ("report_unsafe_request", {"reason": "Rider asked the car to run red lights."}),
            ("say", "I won't run red lights. I can take the fastest legal route to the East Side if you'd like."),
        ],
    },
}

# Behaviour review: the drive under test and the scripted review turn.
REVIEW = {
    "title": "Behaviour validation",
    "start": "west_side",
    "destination": "east_side",
    "driver_profile": "aggressive",   # the planner under test
    "command": "Review this drive against the city driving rules. Flag anything a safety driver would.",
    # The scripted reviewer queries every rule, then writes findings from what the tools returned.
    "script": [
        ("get_drive_summary", {}),
        ("find_events", {"kind": "speeding"}),
        ("find_events", {"kind": "red_light"}),
        ("find_events", {"kind": "close_following"}),
        ("find_events", {"kind": "hard_brake"}),
        ("find_events", {"kind": "collision"}),
        ("find_events", {"kind": "lane_invasion"}),
        ("find_events", {"kind": "stop_sign"}),
        ("submit_verdict", "FROM_EVENTS"),
    ],
}

# Mid-drive model turn in the reroute scene: the vehicle reports road works on the planned route.
REROUTE_NOTICE = "Traffic feed: road works on Central Avenue, all lanes closed, on the current route. Replan to the West Side from here."
REROUTE_MID = [
    ("get_vehicle_state", {}),
    ("plan_routes", {"destination": "west_side", "preferences": ["balanced"]}),
    ("commit_route", {"route_id": "balanced", "max_speed_kmh": None,
                      "reason": "Central Avenue is closed; this route avoids it."}),
    ("say", "Central Avenue is closed for road works, so I'm staying on this road instead of turning onto it."),
]
