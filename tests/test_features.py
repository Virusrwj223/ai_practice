# tests/test_features.py
import pandas as pd
from ml.train import fe_transform

def test_fe_storey_mid_bounds_bva():
    X = pd.DataFrame([{
        "month": "2025-08-01", "town":"ANG MO KIO", "flat_type":"4 ROOM", "flat_model":"IMPROVED",
        "storey_low": 5, "storey_high": 5,  # boundary: equal
        "floor_area_sqm": 90.0, "lease_commence_year": 2000, "remaining_lease_months": 240
    }])
    Y = fe_transform(X)
    assert Y["storey_mid"].iloc[0] == 5

def test_fe_age_clip_bva_negative_age_to_zero():
    X = pd.DataFrame([{
        "month": "1990-01-01",  # before lease commence
        "town":"ANG MO KIO","flat_type":"3 ROOM","flat_model":"NG",
        "storey_low":1, "storey_high":3, "floor_area_sqm":67,
        "lease_commence_year": 2000, "remaining_lease_months": 0
    }])
    Y = fe_transform(X)
    assert Y["flat_age"].iloc[0] == 0  

def test_fe_remaining_lease_years_ep_none_to_zero():
    X = pd.DataFrame([{
        "month":"2025-08-01","town":"BISHAN","flat_type":"4 ROOM","flat_model":"IMPROVED",
        "storey_low":4,"storey_high":6,"floor_area_sqm":95,
        "lease_commence_year":1987,"remaining_lease_months": None
    }])
    Y = fe_transform(X)
    assert Y["remaining_lease_years"].iloc[0] == 0
