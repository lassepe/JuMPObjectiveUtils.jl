using Test: @test, @testset, @test_throws
using JuMPObjectiveUtils: NonlinearJuMPObjective, QuadraticJuMPObjective, add_objective!

import JuMP
import Ipopt

function solve_toy_problem(objective)
    opt_model = JuMP.Model(Ipopt.Optimizer)

    x = JuMP.@variable(opt_model, [1:2, 1:10])
    u = JuMP.@variable(opt_model, [1:2, 1:10])

    JuMP.@constraint(opt_model, x[:, 1] .== [1.0, 1.0])
    JuMP.@constraint(opt_model, [t = 1:9], x[:, t + 1] .== x[:, t] .+ u[:, t])

    add_objective!(opt_model, objective, x, u)
    JuMP.optimize!(opt_model)

    (; x = JuMP.value.(x), u = JuMP.value.(u))
end

function quadratic_cost(x, u; kwargs...)
    sum(x .^ 2) + sum(u .^ 2)
end

function nonlinear_cost(x, u; kwargs...)
    sum(x .^ 2) * sum(u .^ 2)
end

@testset "Tests" begin
    @testset "quadratic" begin
        quaddratic_solution = solve_toy_problem(QuadraticJuMPObjective(quadratic_cost))
        nonlinear_solution = solve_toy_problem(NonlinearJuMPObjective(quadratic_cost))
        @test quaddratic_solution.x ≈ nonlinear_solution.x
        @test quaddratic_solution.u ≈ nonlinear_solution.u
    end

    @testset "nonlinear" begin
        @test_throws ErrorException solve_toy_problem(QuadraticJuMPObjective(nonlinear_cost))
        solve_toy_problem(NonlinearJuMPObjective(nonlinear_cost))
    end
end
