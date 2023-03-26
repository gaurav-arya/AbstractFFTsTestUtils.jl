# This file contains code that was formerly part of Julia. License is MIT: https://julialang.org/license

module AbstractFFTsTestUtils

export test_fft_backend

using AbstractFFTs
using AbstractFFTs: Plan

using LinearAlgebra
using Test

"""
    test_fft_backend(ArrayType=Array; test_real=true, test_inplace=true) 

Run tests to verify correctness of FFT functions using a particular 
backend plan implementation. The backend implementation is assumed to be loaded 
prior to calling this function.

# Arguments

- `ArrayType`: determines the `AbstractArray` implementation for
  which the correctness tests are run. Arrays are constructed via
  `convert(ArrayType, ...)`.
- `test_real=true`: whether to test real-to-complex and complex-to-real FFTs.
- `test_inplace=true`: whether to test in-place plans. 
"""
function test_fft_backend(ArrayType=Array; test_real=true, test_inplace=true)
    @testset "fft correctness" begin
        # DFT along last dimension, results computed using FFTW
        for test_case in (
            (; _x = collect(1:7), dims = 1,
             _fftw_fft = [28.0 + 0.0im,
                          -3.5 + 7.267824888003178im,
                          -3.5 + 2.7911568610884143im,
                          -3.5 + 0.7988521603655248im,
                          -3.5 - 0.7988521603655248im,
                          -3.5 - 2.7911568610884143im,
                          -3.5 - 7.267824888003178im]),
            (; _x = collect(1:8), dims = 1,
             _fftw_fft = [36.0 + 0.0im,
                          -4.0 + 9.65685424949238im,
                          -4.0 + 4.0im,
                          -4.0 + 1.6568542494923806im,
                          -4.0 + 0.0im,
                          -4.0 - 1.6568542494923806im,
                          -4.0 - 4.0im,
                          -4.0 - 9.65685424949238im]),
            (; _x = collect(reshape(1:8, 2, 4)), dims = 2,
             _fftw_fft = [16.0+0.0im  -4.0+4.0im  -4.0+0.0im  -4.0-4.0im;
                          20.0+0.0im  -4.0+4.0im  -4.0+0.0im  -4.0-4.0im]),
            (; _x = collect(reshape(1:9, 3, 3)), dims = 2,
             _fftw_fft = [12.0+0.0im  -4.5+2.598076211353316im  -4.5-2.598076211353316im;
                          15.0+0.0im  -4.5+2.598076211353316im  -4.5-2.598076211353316im;
                          18.0+0.0im  -4.5+2.598076211353316im  -4.5-2.598076211353316im]),
            (; _x = collect(reshape(1:8, 2, 2, 2)), dims = 1:2,
             _fftw_fft = cat([10.0 + 0.0im -4.0 + 0.0im; -2.0 + 0.0im 0.0 + 0.0im],
                             [26.0 + 0.0im -4.0 + 0.0im; -2.0 + 0.0im 0.0 + 0.0im],
                             dims=3)),
            (; _x = collect(1:7) + im * collect(8:14), dims = 1,
             _fftw_fft = [28.0 + 77.0im,
                          -10.76782488800318 + 3.767824888003175im,
                          -6.291156861088416 - 0.7088431389115883im,
                          -4.298852160365525 - 2.7011478396344746im,
                          -2.7011478396344764 - 4.298852160365524im,
                          -0.7088431389115866 - 6.291156861088417im,
                          3.767824888003177 - 10.76782488800318im]),
            (; _x = collect(reshape(1:8, 2, 2, 2)) + im * reshape(9:16, 2, 2, 2), dims = 1:2,
             _fftw_fft = cat([10.0 + 42.0im -4.0 - 4.0im; -2.0 - 2.0im 0.0 + 0.0im],
                             [26.0 + 58.0im -4.0 - 4.0im; -2.0 - 2.0im 0.0 + 0.0im],
                             dims=3)),
        )
            _x, dims, _fftw_fft = test_case
            x = convert(ArrayType, _x) # dummy array that will be passed to plans
            x_complex = convert(ArrayType, complex.(x)) # for testing complex FFTs
            x_complexfloat = convert(ArrayType, complex.(float.(x))) # for testing in-place complex FFTs
            fftw_fft = convert(ArrayType, _fftw_fft)

            # FFT
            y = AbstractFFTs.fft(x_complex, dims)
            @test y ≈ fftw_fft
            if test_inplace
                @test AbstractFFTs.fft!(copy(x_complexfloat), dims) ≈ fftw_fft
            end
            # test plan_fft and also inv and plan_inv of plan_ifft, which should all give 
            # functionally identical plans
            plans_to_test = (plan_fft(x, dims), inv(plan_ifft(x, dims)), 
                             AbstractFFTs.plan_inv(plan_ifft(x, dims)))
            for P in plans_to_test
                @test mul!(similar(y), P, copy(x_complexfloat)) ≈ fftw_fft
            end
            if test_inplace
                plans_to_test = (plans_to_test..., plan_fft!(similar(x_complexfloat), dims))
            end
            for P in plans_to_test 
                @test eltype(P) <: Complex
                @test P * copy(x_complex) ≈ fftw_fft
                @test P \ (P * copy(x_complex)) ≈ x_complex
                @test fftdims(P) == dims
            end

            # BFFT
            fftw_bfft = prod(size(x_complex, d) for d in dims) .* x_complex
            @test AbstractFFTs.bfft(y, dims) ≈ fftw_bfft
            if test_inplace
                @test AbstractFFTs.bfft!(copy(y), dims) ≈ fftw_bfft
            end
            P = plan_bfft(similar(y), dims)
            @test mul!(similar(x_complexfloat), P, copy(y)) ≈ fftw_bfft
            plans_to_test = if test_inplace
                (P, plan_bfft!(similar(y), dims))
            else
                (P,)
            end
            for P in plans_to_test
                @test eltype(P) <: Complex
                @test P * copy(y) ≈ fftw_bfft
                @test P \ (P * copy(y)) ≈ y
                @test fftdims(P) == dims
            end

            # IFFT
            fftw_ifft = x_complex
            @test AbstractFFTs.ifft(y, dims) ≈ fftw_ifft
            if test_inplace
                @test AbstractFFTs.ifft!(copy(y), dims) ≈ fftw_ifft
            end
            plans_to_test = (plan_ifft(x, dims), inv(plan_fft(x, dims)), 
                             AbstractFFTs.plan_inv(plan_fft(x, dims)))
            for P in plans_to_test
                @test mul!(similar(x_complexfloat), P, copy(y)) ≈ fftw_ifft
            end
            if test_inplace
                plans_to_test = (plans_to_test..., plan_ifft!(similar(x_complexfloat), dims))
            end
            for P in plans_to_test
                @test eltype(P) <: Complex
                @test P * copy(y) ≈ fftw_ifft
                @test P \ (P * copy(y)) ≈ y
                @test fftdims(P) == dims
            end

            if test_real && (eltype(x) <: Real) 
                x_real = float.(x) # for testing in-place real FFTs
                # RFFT
                fftw_rfft = selectdim(fftw_fft, first(dims), 1:(size(fftw_fft, first(dims)) ÷ 2 + 1))
                ry = AbstractFFTs.rfft(x, dims)
                @test ry ≈ fftw_rfft
                for P in (plan_rfft(similar(x), dims), inv(plan_irfft(similar(ry), size(x, first(dims)), dims)), 
                          AbstractFFTs.plan_inv(plan_irfft(similar(ry), size(x, first(dims)), dims)))
                    @test eltype(P) <: Real
                    @test P * x ≈ fftw_rfft
                    @test mul!(similar(ry), P, copy(x_real)) ≈ fftw_rfft
                    @test P \ (P * x) ≈ x
                    @test fftdims(P) == dims
                end

                # BRFFT
                fftw_brfft = prod(size(x, d) for d in dims) .* x
                @test AbstractFFTs.brfft(ry, size(x, first(dims)), dims) ≈ fftw_brfft
                P = plan_brfft(similar(ry), size(x, first(dims)), dims)
                @test P * ry ≈ fftw_brfft
                @test mul!(similar(x_real), P, copy(ry)) ≈ fftw_brfft
                @test P \ (P * ry) ≈ ry
                @test fftdims(P) == dims

                # IRFFT
                fftw_irfft = x
                @test AbstractFFTs.irfft(ry, size(x, first(dims)), dims) ≈ fftw_irfft
                for P in (plan_irfft(similar(ry), size(x, first(dims)), dims), inv(plan_rfft(similar(x), dims)), 
                          AbstractFFTs.plan_inv(plan_rfft(similar(x), dims)))
                    @test P * ry ≈ fftw_irfft
                    @test mul!(similar(x_real), P, copy(ry)) ≈ fftw_irfft
                    @test P \ (P * ry) ≈ ry
                    @test fftdims(P) == dims
                end
            end
        end
    end
end

end
