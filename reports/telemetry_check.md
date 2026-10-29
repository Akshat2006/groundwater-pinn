# Forecast check on telemetry wells (never used by the model)

Maps: `exports/forecast-v1` (trained on manual readings up to 2024-01). Telemetry months 2024-02 to 2025-11, shallow wells only (median depth <= 42.1 m).

- Wells: 160 of 209 telemetry wells with data in the period (2961 well-months).
- **Typical error (RMSE): map 8.33 m vs nearest-well baseline 8.62 m** (average error 6.06 vs 4.72 m). The map's average bias is +0.37 m (+ = map deeper than measured).
- **The map beats the baseline at 37% of wells.**
- **90% range coverage:** the measured depth was inside depth +- range90 in 87% of well-months.
- **Change 2024 to 2025** (158 wells, same months both years): measured median +0.75 m, map +0.99 m; the map got the direction right at 82% of wells; average error of the change 1.00 m vs 1.27 m for "no change".

Plain words: RMSE is the typical size of the error in metres. The baseline is what you would guess at that spot without the model: the last reading of the closest CGWB well. Coverage should be near 90%.

Caveats: telemetry sensors are a different network (some may sit in a deeper layer even after the depth filter), readings are preliminary, and one cell of the map is 1 km wide.
