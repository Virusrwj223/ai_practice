from ml.infer import predict
def test_predict_smoke():
    rec = {
        "month":"2024-05","town":"ANG MO KIO","flat_type":"4 ROOM",
        "flat_model":"IMPROVED","storey_low":4,"storey_high":6,
        "floor_area_sqm":90,"lease_commence_year":1980,"remaining_lease_months":420
    }
    out = predict(rec)[0]
    assert {"resale_pred","bto_proxy","required_income"} <= set(out)
    assert out["resale_pred"] > 0
