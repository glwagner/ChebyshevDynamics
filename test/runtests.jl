using
  ChebyshevDynamics,
  Test

include("test_bvp.jl")

@testset "Utilities" begin 
  @test testcheb()
  @test testuni()
  @test testcoeff()
  @test testvalue()
end

@testset "Dynamics" begin 
  @test testbvp()
end
