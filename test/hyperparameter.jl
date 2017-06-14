MODHP = MLKernels.HyperParameters

info("Testing ", MOD.OpenBound)
for T in (FloatingPointTypes..., IntegerTypes...)
    if T <: Integer
        @test_throws ErrorException MOD.OpenBound(one(T))
    else
        if T <: AbstractFloat
            @test_throws ErrorException MOD.OpenBound(convert(T,  NaN))
            @test_throws ErrorException MOD.OpenBound(convert(T,  Inf))
            @test_throws ErrorException MOD.OpenBound(convert(T, -Inf))
        end

        for x in (1,2)
            B = MOD.OpenBound(convert(T,x))

            @test B.value  == x
            @test eltype(B) == T
        end

        for U in FloatingPointTypes
            B = MOD.OpenBound(one(T))

            B_u = convert(MOD.OpenBound{U}, B)
            @test B_u.value == one(U)
            @test eltype(B_u) == U
        end

        B = MOD.OpenBound(zero(T))

        @test checkvalue(-one(T), B) == true
        @test checkvalue(zero(T), B) == false
        @test checkvalue(one(T),  B) == false

        @test checkvalue(B, -one(T)) == false
        @test checkvalue(B, zero(T)) == false
        @test checkvalue(B, one(T))  == true

        @test string(B) == string("OpenBound(", zero(T), ")")
        @test show(DevNull, B) == nothing
    end
end

info("Testing ", MOD.ClosedBound)
for T in (FloatingPointTypes..., IntegerTypes...)
    if T <: AbstractFloat
        @test_throws ErrorException MOD.ClosedBound(convert(T,  NaN))
        @test_throws ErrorException MOD.ClosedBound(convert(T,  Inf))
        @test_throws ErrorException MOD.ClosedBound(convert(T, -Inf))
    end

    for x in (1,2)
        B = MOD.ClosedBound(convert(T,x))

        @test B.value  == x
        @test eltype(B) == T
    end

    for U in (FloatingPointTypes..., IntegerTypes...)
        B = MOD.ClosedBound(one(T))

        B_u = convert(MOD.ClosedBound{U}, B)
        @test B_u.value == one(U)
        @test eltype(B_u) == U
    end

    B = MOD.ClosedBound(zero(T))

    @test checkvalue(-one(T), B) == true
    @test checkvalue(zero(T), B) == true
    @test checkvalue(one(T),  B) == false

    @test checkvalue(B, -one(T)) == false
    @test checkvalue(B, zero(T)) == true
    @test checkvalue(B, one(T))  == true

    @test string(B) == string("ClosedBound(", zero(T), ")")
    @test show(DevNull, B) == nothing
end

info("Testing ", MOD.NullBound)
for T in (FloatingPointTypes..., IntegerTypes...)
    @test eltype(NullBound(T)) == T

    for U in (FloatingPointTypes..., IntegerTypes...)
        B = MOD.NullBound(T)

        B_u = convert(MOD.NullBound{U}, B)
        @test eltype(B_u) == U
    end

    B = MOD.NullBound(T)

    @test checkvalue(-one(T), B) == true
    @test checkvalue(zero(T), B) == true
    @test checkvalue(one(T),  B) == true

    @test checkvalue(B, -one(T)) == true
    @test checkvalue(B, zero(T)) == true
    @test checkvalue(B, one(T))  == true

    @test string(B) == string("NullBound(", T, ")")
    @test show(DevNull, B) == nothing
end

info("Testing ", MOD.Interval)
for T in FloatingPointTypes
    @test_throws ErrorException MOD.Interval(MOD.ClosedBound(one(T)), MOD.OpenBound(one(T)))
    @test_throws ErrorException MOD.Interval(MOD.OpenBound(one(T)),   MOD.ClosedBound(one(T)))
    @test_throws ErrorException MOD.Interval(MOD.OpenBound(one(T)),   MOD.OpenBound(one(T)))

    a = MOD.ClosedBound(one(T))
    b = MOD.ClosedBound(one(T))
    I = MOD.Interval(a,b)
    @test I.a == a
    @test I.b == b
    @test eltype(I) == T

    I = MOD.interval(nothing, nothing)
    @test I.a == NullBound(Float64)
    @test I.b == NullBound(Float64)
    @test eltype(I) == Float64

    for a in (NullBound(T), ClosedBound(-one(T)), OpenBound(-one(T)))
        for b in (NullBound(T), ClosedBound(one(T)), OpenBound(one(T)))
            I = MOD.Interval(a, b)
            @test I.a == a
            @test I.b == b
            @test eltype(I) == T

            if typeof(a) <: NullBound
                if typeof(b) <: NullBound
                    @test I == MOD.interval(T)
                    @test string(I) == string("interval(", T, ")")
                else
                    @test I == MOD.interval(nothing, b)
                    @test string(I) == string("interval(nothing,", string(b), ")")
                end
            else
                if typeof(b) <: NullBound
                    @test I == MOD.interval(a,nothing)
                    @test string(I) == string("interval(", string(a), ",nothing)")
                else
                    @test I == MOD.interval(a, b)
                    @test string(I) == string("interval(", string(a), ",", string(b), ")")
                end
            end

            @test MOD.checkvalue(I, convert(T,-2)) == (typeof(a) <: NullBound ? true  : false)
            @test MOD.checkvalue(I, -one(T))       == (typeof(a) <: OpenBound ? false : true)
            @test MOD.checkvalue(I, zero(T))       == true
            @test MOD.checkvalue(I,  one(T))       == (typeof(b) <: OpenBound ? false : true)
            @test MOD.checkvalue(I, convert(T,2))  == (typeof(b) <: NullBound ? true  : false)
        end
    end

    a = convert(T,7.6)
    b = convert(T,23)
    c = convert(T,13)

    @test show(DevNull, MOD.interval(MOD.ClosedBound(one(T)), MOD.ClosedBound(one(T)))) == nothing

end

info("Testing ", MOD.checkvalue)
for T in FloatingPointTypes
    l = convert(T,-10)
    a = -one(T)
    b = one(T)
    u = convert(T,10)
    for A in (NullBound(T), ClosedBound(a), OpenBound(a))
        T_a = typeof(A)
        for B in (NullBound(T), ClosedBound(b), OpenBound(b))
            T_b = typeof(B)
            I = MODHP.interval(A,B)

            for x in linspace(l,a,20)
                if T_a <: NullBound || T_a <: ClosedBound && x == a
                    @test MOD.checkvalue(I,x) == true
                else 
                    @test MOD.checkvalue(I,x) == false
                end
            end

            for x in linspace(a,b,20)
                if x == a
                    @test MOD.checkvalue(I,x) == T_a <: OpenBound ? false : true
                elseif x == b
                    @test MOD.checkvalue(I,x) == T_b <: OpenBound ? false : true
                else
                    @test MOD.checkvalue(I,x) == true
                end
            end

            for x in linspace(b,u,20)
                if T_b <: NullBound || T_b <: ClosedBound && x == b
                    @test MOD.checkvalue(I,x) == true
                else 
                    @test MOD.checkvalue(I,x) == false
                end
            end
        end
    end

end


info("Testing ", MOD.interval)
@test typeof(MOD.interval(nothing, nothing)) == MOD.Interval{Float64,NullBound{Float64},NullBound{Float64}}
for T in FloatingPointTypes
    for a in (NullBound(T), ClosedBound(-one(T)), OpenBound(-one(T)))
        for b in (NullBound(T), ClosedBound(one(T)), OpenBound(one(T)))
            null_a = typeof(a) <: NullBound
            null_b = typeof(b) <: NullBound

            I = interval(null_a ? nothing : a, null_b ? nothing : b)

            if null_a && null_b
                @test typeof(I) == MOD.Interval{Float64,NullBound{Float64},NullBound{Float64}}
            else
                @test I.a == a
                @test I.b == b
                @test eltype(T) == T
            end
        end
    end
end

info("Testing ", MODHP.theta)
#=
for T in FloatingPointTypes
    for a in (NullBound(T), ClosedBound(-one(T)), OpenBound(-one(T)))
        for b in (NullBound(T), ClosedBound(one(T)), OpenBound(one(T)))
            I = MOD.Interval(a, b)

            if typeof(a) <: NullBound
                if typeof(b) <: NullBound

                elseif typeof(b) <: ClosedBound

                else

                end
            elseif typeof(a) <: ClosedBound
                if typeof(b) <: NullBound

                elseif typeof(b) <: ClosedBound

                else

                end
            else
                if typeof(b) <: NullBound

                elseif typeof(b) <: ClosedBound

                else

                end
            end
        end
    end
end
=#


info("Testing ", MODHP.eta)
for T in FloatingPointTypes
    for a in (NullBound(T), ClosedBound(-one(T)), OpenBound(-one(T)))
        for b in (NullBound(T), ClosedBound(one(T)), OpenBound(one(T)))
            I = MODHP.interval(a,b)
            for x in linspace(-convert(T,-0.9),convert(T,0.9),10)
                @test_approx_eq MODHP.eta(I,MODHP.theta(I,x)) x
            end
        end
    end
end


info("Testing ", MOD.lowerboundtheta)
info("Testing ", MOD.upperboundtheta)
for T in FloatingPointTypes
    for a in (NullBound(T), ClosedBound(-one(T)), OpenBound(-one(T)))
        for b in (NullBound(T), ClosedBound(one(T)), OpenBound(one(T)))
            I = MOD.Interval(a, b)

            if typeof(a) <: NullBound
                if typeof(b) <: NullBound
                    @test MOD.lowerboundtheta(I) == convert(T,-Inf)
                    @test MOD.upperboundtheta(I) == convert(T,Inf)
                elseif typeof(b) <: ClosedBound
                    @test MOD.lowerboundtheta(I) == convert(T,-Inf)
                    @test MOD.upperboundtheta(I) == I.b.value
                else
                    @test MOD.lowerboundtheta(I) == convert(T,-Inf)
                    @test MOD.upperboundtheta(I) == convert(T,Inf)
                end
            elseif typeof(a) <: ClosedBound
                if typeof(b) <: NullBound
                    @test MOD.lowerboundtheta(I) == I.a.value
                    @test MOD.upperboundtheta(I) == convert(T,Inf)
                elseif typeof(b) <: ClosedBound
                    @test MOD.lowerboundtheta(I) == I.a.value
                    @test MOD.upperboundtheta(I) == I.b.value
                else
                    @test MOD.lowerboundtheta(I) == convert(T,-Inf)
                    @test MOD.upperboundtheta(I) == log(I.b.value - I.a.value)
                end
            else
                if typeof(b) <: NullBound
                    @test MOD.lowerboundtheta(I) == convert(T,-Inf)
                    @test MOD.upperboundtheta(I) == convert(T,Inf)
                elseif typeof(b) <: ClosedBound
                    @test MOD.lowerboundtheta(I) == convert(T,-Inf)
                    @test MOD.upperboundtheta(I) == log(I.b.value - I.a.value)
                else
                    @test MOD.lowerboundtheta(I) == convert(T,-Inf)
                    @test MOD.upperboundtheta(I) == convert(T,Inf)
                end
            end
        end
    end
end

info("Testing ", MOD.HyperParameter)
for T in (FloatingPointTypes..., IntegerTypes...)
    I = interval(nothing, ClosedBound(zero(T)))

    @test_throws ErrorException HyperParameter(one(T), I)

    P = HyperParameter(one(T), interval(nothing, ClosedBound(one(T))))
    @test getindex(P.value) == one(T)
    @test eltype(P) == T

    @test getvalue(P) == one(T)

    @test checkvalue(P, convert(T,2)) == false
    @test checkvalue(P, zero(T)) == true

    setvalue!(P, zero(T))
    @test getindex(P.value) == zero(T)
    @test getvalue(P) == zero(T)

    if T in FloatingPointTypes
        for a in (NullBound(T), ClosedBound(-one(T)), OpenBound(-one(T)))
            for b in (NullBound(T), ClosedBound(one(T)), OpenBound(one(T)))
                I = MODHP.Interval(a, b)
                P = HyperParameter(zero(T),I)

                @test_approx_eq MODHP.gettheta(P) MODHP.theta(I, zero(T))

                MODHP.settheta!(P, MODHP.theta(I, convert(T,0.5)))
                @test_approx_eq MODHP.gettheta(P) MODHP.theta(I, convert(T,0.5))
                @test_approx_eq MODHP.getvalue(P) convert(T,0.5)
            end
        end
    end

    P1 = HyperParameter(zero(T), interval(T))
    P2 = HyperParameter(one(T),  interval(T))
    for op in (isless, ==, +, -, *, /)
        @test op(P1, one(T)) == op(getvalue(P1), one(T))
        @test op(one(T), P1) == op(one(T), getvalue(P1))
        @test op(P1, P2)     == op(getvalue(P1), getvalue(P2))
    end

    @test show(DevNull, P) == nothing
end
