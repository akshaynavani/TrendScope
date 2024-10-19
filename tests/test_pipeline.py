from trendscope.pipeline import run_pipeline


def test_pipeline_end_to_end(memory_backend, small_config):
    small_config.forecast.models = ["seasonal_naive", "arima"]
    result = run_pipeline(small_config, backend=memory_backend, seed=True, persist=True, plots=True)

    assert len(result.outcomes) == 2
    assert len(result.forecasts) == 2 * small_config.forecast.horizon
    assert {"opportunity_score", "opportunity_rank", "tier", "headroom"} <= set(result.headroom.columns)
    assert result.headroom["opportunity_rank"].min() == 1

    stored = memory_backend.query("SELECT COUNT(*) AS n FROM segment_headroom")["n"].iloc[0]
    assert stored == 2
    runs = memory_backend.query("SELECT run_id FROM forecast_runs")
    assert runs["run_id"].iloc[0] == result.run_id

    pngs = [p for p in result.artifacts if p.suffix == ".png"]
    csvs = [p for p in result.artifacts if p.suffix == ".csv"]
    assert pngs and all(p.exists() for p in pngs)
    assert csvs and all(p.exists() for p in csvs)


def test_pipeline_segment_filter(memory_backend, small_config):
    result = run_pipeline(small_config, backend=memory_backend, seed=True,
                          persist=False, plots=False, segment_filter="emea")
    assert [o.segment.segment_id for o in result.outcomes] == ["EMEA | Software | Direct"]
