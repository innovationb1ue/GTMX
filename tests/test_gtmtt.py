from src.gtmx import GTMTimeSeries


def test_gtmtt(example_data):
    e = GTMTimeSeries(s=2, map_shape=(12, 12), group_size=2)
    e.fit(example_data, epoch=30)
    print("vis time")

    e.plot_llh()
    e.plot(mode='mode')
    e.plot(mode='mean')
    e.plot(mode='hot')


def test_gtmtt_vis_system(example_data):
    gtmtt = GTMTimeSeries(s=2, map_shape=(12, 12), group_size=2)
    gtmtt.start_vis_server()
    gtmtt.fit(example_data, epoch=30)


def test_heat_map(gtmtt: GTMTimeSeries):
    gtmtt.plot('hot')
