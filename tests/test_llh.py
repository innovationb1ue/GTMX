from src.gtmx import GTMBase


def test_llh_monotonically_increase(original_gtm: GTMBase):
    llh = original_gtm.llhs
    assert all(x < y for x, y in zip(llh, llh[1:]))


def test_llh_gtmtt(gtmtt):
    llh = gtmtt.llhs
    assert all(x < y for x, y in zip(llh, llh[1:]))



