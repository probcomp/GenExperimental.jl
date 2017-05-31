stdio = terralib.includec("stdio.h")
stdlib = terralib.includec("stdlib.h")
printf = stdio.printf

rand = require("rand")
T = require("trace").T

-- load primitives
primitives = require("primitives")
flip = primitives.flip
normal = primitives.normal

terra test_model([T], weight : float)
    var coin1 = flip(weight, "coin1")
    var coin2 = flip(weight, "coin2")
    printf("coin1: %d, coin2: %d\n", coin1, coin2)
    return coin1
end

terra doit()
    rand.seedtime()
    var trace : Trace
    trace:init() -- TODO use better syntax here
    var result = test_model(&trace, 0.5)
    printf("result=%d\n", result)
    printf("log_weight=%f\n", trace.log_weight)
end
doit()

-- print(test_model:printpretty())

-- linear regression

terra linreg([T], x : &double, n: int)
    var slope = normal(0.0, 2.0, "slope")
    var intercept = normal(0.0, 2.0, "intercept")
    var y = [&double](stdlib.malloc(sizeof(double) * n))
    var name : int8[100]
    for i=0,n do
        var outlier = primitives._flip(0.1)
        var mean = intercept + slope * x[i]
        stdio.snprintf(name, 100, "y_%d", i)
        if outlier then
            y[i] = normal(mean, 10.0, name)
        else 
            y[i] = normal(mean, 0.1, name)
        end
    end
    stdlib.free(y)
end

print(linreg:printpretty())

struct LinregResult {
    log_weight : double
    slope : double
    intercept : double
}

terra linreg_sample(x : &double, y : &double, n : int) : LinregResult
    var trace : Trace
    trace:init()
    var name : int8[100]
    for i=0,10 do
        stdio.snprintf(name, 100, "y_%d", i)
        -- printf("putting %s\n", name)
        trace:put_double(name, y[i])
    end
    linreg(&trace, x, 10)
    var result : LinregResult
    trace:get_double("slope", &result.slope)
    trace:get_double("intercept", &result.intercept)
    result.log_weight = trace.log_weight
    return result
end


terra run_linreg()
    var x : double[10]
    for i=0,10 do
        x[i] = -3.0 + i
    end
    var y : double[10]
    for i=0,10 do
        y[i] = 3.0 - i
    end
    y[7] = 4 -- comment/uncomment for no outlier / outlier
    var num_samples = 10000
    var results = [&LinregResult](stdlib.malloc(num_samples*sizeof(LinregResult)))
    var log_weights = [&double](stdlib.malloc(num_samples*sizeof(double)))
    for i=0,num_samples do
        var result = linreg_sample(x, y, 10)
        -- printf("slope=%0.3f, intercept=%0.3f, log_weight=%0.3f\n", result.slope, result.intercept, result.log_weight)
        results[i] = result
        log_weights[i] = result.log_weight
    end
    var k = rand.log_categorical(log_weights, num_samples)
    var slope = results[k].slope
    var intercept = results[k].intercept
    printf("slope=%0.3f, intercept=%0.3f\n", slope, intercept)
    stdlib.free(results)
    stdlib.free(log_weights)
end

for i=1,100 do
    run_linreg()
end
