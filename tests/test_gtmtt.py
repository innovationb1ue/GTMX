from src.gtmx import GTMTimeSeries
import pandas as pd


def test_gtmtt():
    dta = pd.read_excel("../data/example_data.xlsx", sheet_name=0,  header=None)
    dta = dta.to_numpy()
    dta = dta[:, :9]  # first 9 cols
    print(dta.shape)
    dta = dta.reshape([4, dta.shape[0]//4, 9])

    e = GTMTimeSeries(s=2, map_shape=(12, 12), group_size=2)
    e.fit(dta, epoch=30)
    print("vis time")

    e.plot_llh()
    e.plot(mode='mode')
    e.plot(mode='mean')
    e.plot(mode='hot')


def test_heat_map(gtmtt: GTMTimeSeries):
    gtmtt.plot('hot')
