"""The setup of the tests (tests/testthat/setup.R of the R package dflimit): at most two threads."""
import dflimit

dflimit.ltratio_threads(2)
