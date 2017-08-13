using Base.Test

@testset "isempty" begin
    @test isempty(AddressTrie())
end

@testset "push!, in, delete!, getindex, isempty" begin

    t = AddressTrie()
    push!(t, ())
    @test () in t

    push!(t, ("foo",))
    @test ("foo",) in t
    @test () in t[("foo",)]

    push!(t, ("foo", 1))
    @test ("foo",) in t
    @test ("foo", 1) in t
    @test (1,) in t[("foo",)]
    @test () in t[("foo", 1)]

    push!(t, ("foo", 2))
    @test ("foo",) in t
    @test ("foo", 1) in t
    @test ("foo", 2) in t

    push!(t, ("foo", 1, "X"))
    @test ("foo",) in t
    @test ("foo", 1) in t
    @test ("foo", 1, "X") in t
    @test ("foo", 2) in t
    @test (1,) in t[("foo",)]
    @test (2,) in t[("foo",)]
    @test () in t[("foo", 1)]
    @test ("X",) in t[("foo", 1)]

    push!(t, ("baz", 1))
    @test !(("baz",) in t)
    @test ("baz", 1) in t
    @test (1,) in t[("baz",)]
    @test () in t[("baz", 1)]

    delete!(t, ())
    @test !(() in t)

    delete!(t, ("foo",))
    @test !(("foo",) in t)
    @test ("foo", 1) in t # the entire subtrie is not deleted, only the one element
  
    delete!(t, ("foo", 1))
    delete!(t, ("foo", 1, "X"))
    delete!(t, ("foo", 2))
    delete!(t, ("baz", 1))
    @test isempty(t)
end


@testset "shorthand for single-element addresses" begin
    t = AddressTrie()

    push!(t, "foo")
    @test ("foo",) in t
    @test "foo" in t
    @test () in t[("foo",)]
    @test () in t["foo"]

    push!(t, ("foo", 1))
    @test "foo" in t
    @test ("foo", 1) in t
    @test 1 in t[("foo",)]

    delete!(t, "foo")
    @test !(("foo",) in t)
    @test ("foo", 1) in t # the entire subtrie is not deleted, only the one element

    delete!(t, ("foo", 1))
    @test isempty(t)
end

@testset "iterator" begin

    t = AddressTrie()
    push!(t, ())
    push!(t, "foo")
    push!(t, ("foo", 1))
    push!(t, ("foo", 2))
    push!(t, ("bar", "x", "y", "z"))
    push!(t, ("foo", 2, 3))
    push!(t, ("foo", 3, 4, 5))

    state = start(t)
    @test !done(t, state)

    item, state = next(t, state)
    @test !done(t, state)
    @test item == ()

    item, state = next(t, state)
    @test !done(t, state)
    @test item == ("foo",)

    item, state = next(t, state)
    @test !done(t, state)
    @test item == ("foo", 1)

    item, state = next(t, state)
    @test !done(t, state)
    @test item == ("foo", 2)

    item, state = next(t, state)
    @test !done(t, state)
    @test item == ("foo", 2, 3)

    item, state = next(t, state)
    @test !done(t, state)
    @test item == ("foo", 3, 4, 5)

    item, state = next(t, state)
    @test done(t, state)
    @test item == ("bar", "x", "y", "z")

    # remove all items and start new iterator, checking it is done
    delete!(t, ())
    delete!(t, "foo")
    delete!(t, ("foo", 1))
    delete!(t, ("foo", 2))
    delete!(t, ("bar", "x", "y", "z"))
    delete!(t, ("foo", 2, 3))
    delete!(t, ("foo", 3, 4, 5))
    state = start(t)
    @test done(t, state)

    # add new items, check the order, which is based on insertion, except that
    # the root address gets visited first
    push!(t, ("baz",))
    push!(t, ("foo", 1))
    push!(t, "foo")

    state = start(t)
    @test !done(t, state)
    
    item, state = next(t, state)
    @test !done(t, state)
    @test item == ("baz",)

    item, state = next(t, state)
    @test !done(t, state)
    @test item == ("foo",)

    item, state = next(t, state)
    @test done(t, state)
    @test item == ("foo", 1)

end

@testset "iterator for .. in .. " begin

    t = AddressTrie()
    push!(t, "foo")
    push!(t, ("foo", 1))
    push!(t, ("foo", 2))
    push!(t, ("bar", "x", "y", "z"))

    arr = []
    for addr in t
        push!(arr, addr)
    end
    @test length(arr) == 4
    @test arr[1] == ("foo",)
    @test arr[2] == ("foo", 1)
    @test arr[3] == ("foo", 2)
    @test arr[4] == ("bar", "x", "y", "z")
    
end

@testset "disjoint" begin

    t1 = AddressTrie()
    push!(t1, "foo")
    push!(t1, ("foo", 1))

    t2 = AddressTrie()
    push!(t2, "bar")
    @test disjoint(t1, t2)

    push!(t2, ("foo", 1))
    @test !disjoint(t1, t2)
end

@testset "constructor" begin

    t = AddressTrie("foo", ("bar", 1), (5,))
    @test ("foo",) in t
    @test ("bar", 1) in t
    @test (5,) in t
end
