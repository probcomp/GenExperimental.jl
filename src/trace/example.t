require("rand")
require("trace")
stdio = terralib.includec("stdio.h")
printf = stdio.printf

-- model = terra()
-- var weight = uniform() #bias
-- var coin = flip(weight)
-- end

terra _model(_constraints : &Trace, _requested : &Request)
    var _trace = new_trace()
    var _log_weight : double = 0.0
    
    var weight : float
    if trace_has(_constraints, "bias") then
        weight = trace_get(_constraints, "bias")
        _log_weight = _log_weight + 0.0 -- TODO _beta(bias, a, b) 
    else
        weight = uniform()
        if request_has(_requested, "bias") then
            _trace = trace_add(_trace, "bias", weight)
        end
    end

    var coin : bool = flip(weight)
    printf("weight=%f, coin=%i\n", weight, coin)
    return _trace, _log_weight
end

terra doit()
    seedtime()
    var constraints = new_trace()
    var request = new_trace()
    request = request_add(request, "bias")
    printf("reqyest has bias=%d\n", request_has(request, "bias"))
    var trace, w = _model(constraints, request)
    printf("trace has bias=%d\n", trace_has(trace, "bias"))
    printf("response[bias] = %f\n", trace_get(trace, "bias"), w)
    -- TODO: free them
end

doit()
