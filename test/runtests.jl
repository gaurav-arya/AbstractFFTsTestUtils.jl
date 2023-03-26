
using AbstractFFTsTestUtils

# Load example plan implementation from AbstractFFTs' tests
import AbstractFFTs
AbstractFFTs_rootdir = dirname(dirname(pathof(AbstractFFTs)))
include(joinpath(AbstractFFTs_rootdir, "test", "TestPlans.jl"))
using .TestPlans

# Run interface tests for TestPlans 
test_fft_backend(Array) 