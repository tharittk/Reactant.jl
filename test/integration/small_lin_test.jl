using LinearAlgebra, Reactant, Test

@testset "map" begin
    A = rand(Float64, 3, 4, 5)
    B = rand(Float64, 3, 4, 5)
    C = rand(Float64, 3, 4, 5)
    D = rand(Float64, 3, 4, 5)
    
    A_ra = Reactant.to_rarray(A)
    B_ra = Reactant.to_rarray(B)
    C_ra = Reactant.to_rarray(C)
    D_ra = Reactant.to_rarray(D)

    m_hlo = @jit Reactant.Ops.map(+, A_ra, B_ra)
    @test m_hlo ≈ map(+, A, B)
 
    m_hlo = @jit Reactant.Ops.map(+, A_ra, B_ra, C_ra)
    @test m_hlo ≈ map(+, A, B, C)

    m_hlo = @jit Reactant.Ops.map(*, A_ra, B_ra, C_ra, D_ra)
    @test m_hlo ≈ map(*, A, B, C, D)

    # anonymous function
    E = [[1 2];[4 3]]
    E_ra = Reactant.to_rarray(E)
    m_hlo = @jit Reactant.Ops.map((x) -> x^2, E_ra)
    @test m_hlo ≈ map( (x) -> x^2, E)

    F = [[5 6];[7 8]]
    F_ra = Reactant.to_rarray(F)
    m_hlo = @jit Reactant.Ops.map((x, y) -> x^2 + y, E_ra, F_ra)
    @test m_hlo ≈ map((x, y) -> x^2 + y, E, F)   
end

@testset "mapreduce" begin
    # A = rand(Float64, 3, 4, 5)
    # B = rand(Float64, 3, 4, 5)
    # C = rand(Float64, 3, 4, 5)
    # D = rand(Float64, 3, 4, 5)
    
    # A_ra = Reactant.to_rarray(A)
    # B_ra = Reactant.to_rarray(B)
    # C_ra = Reactant.to_rarray(C)
    # D_ra = Reactant.to_rarray(D)

    A = [1 2; 3 4]
    A_ra = Reactant.to_rarray(A)

    mr_hlo = @jit Base.mapreduce((x) -> x*3, +, A_ra; dims = 1:2)
    @show mr_hlo
    # @test mr_hlo ≈ mapreduce(map_f, +, A; dims = 1:2)
end

