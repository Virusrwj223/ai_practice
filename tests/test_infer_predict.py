# tests/test_infer_predict.py
import pandas as pd

def test_predict_single_and_batch_equivalence(stub_models):
    from ml.infer import predict, CONF
    rec = {
        "month":"2025-08", "town":"ANG MO KIO", "flat_type":"4 ROOM", "flat_model":"IMPROVED",
        "storey_low":4,"storey_high":6,"floor_area_sqm":90,
        "lease_commence_year":2000,"remaining_lease_months":240
    }
    one = predict(rec)[0]
    two = predict([rec])[0]
    assert set(["resale_pred","bto_proxy","required_income"]).issubset(one)
    assert one["resale_pred"] == two["resale_pred"]
    assert abs(one["bto_proxy"] - one["resale_pred"]*(1-CONF["discount"])) < 1e-6
    assert one["required_income"] > 0

def test_predict_monotonic_area_bva(stub_models):
    from ml.infer import predict
    a = predict({"month":"2025-08","town":"BISHAN","flat_type":"4 ROOM","flat_model":"IMPROVED",
                 "storey_low":4,"storey_high":6,"floor_area_sqm":80,"lease_commence_year":2000,
                 "remaining_lease_months":240})[0]["resale_pred"]
    b = predict({"month":"2025-08","town":"BISHAN","flat_type":"4 ROOM","flat_model":"IMPROVED",
                 "storey_low":4,"storey_high":6,"floor_area_sqm":100,"lease_commence_year":2000,
                 "remaining_lease_months":240})[0]["resale_pred"]
    assert b > a  # larger area ⇒ higher predicted price under stub
