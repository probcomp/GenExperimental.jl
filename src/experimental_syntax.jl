using Gen

@program foo(x::Int, y::Float64) begin
    a = flip(0.5) ~ "a"
    a + x
end
    
@program foo2(x::Int, y::Float64) begin
    t = Trace()
    z = @generate(t, foo(x, y))
    println("nested trace")
    print(t)
    println("call toambient trace")
    z2 = @generate(foo(x, y))
    println(z2)
    x + y + z + z2 + normal(0.5, 0.1) ~ "b"
end

# TODO add an @curtrace() macro which returns the current trace, for debugging purposes..

print()
println("calling foo1")
t = Trace()
@generate(t, foo(1, 2.))
print(t)

print()
println("calling foo2")
t = Trace()
@generate(t, foo2(1, 2.0))
print(t)
