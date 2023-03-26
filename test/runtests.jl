using AbstractFFTsTestUtils

# Load example plan implementation from AbstractFFTs' tests
parentrepodir = dirname(dirname(dirname(@__DIR__)))
include(joinpath(parentrepodir, "test", "TestPlans.jl"))
using .TestPlans

# Run interface tests for TestPlans 
test_fft_backend(Array) 
