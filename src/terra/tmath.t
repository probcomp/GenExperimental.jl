local cmath = terralib.includec("math.h")
local cfloat = terralib.includec("float.h")

tmath = {}
for k,v in pairs(cmath) do 
    tmath[k] = v
end

tmath.logsumexp = terra(x : &double, n : int) : double
    var maximum : double = -(cfloat.__DBL_MAX__)
    for i=0,n do
        if x[i] > maximum then
            maximum = x[i]
        end
    end
    var sum = 0.0
    for i=0,n do
        sum = sum + cmath.exp(x[i] - maximum)
    end
    return cmath.log(sum) + maximum
end

return tmath
