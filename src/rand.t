local C = terralib.includecstring [[
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void seedtime()
{
   time_t t;
   srand((unsigned) time(&t));
}
]]

local math = require("tmath")

rand = {}

rand.seedtime = terra() 
    C.seedtime()
end

rand.uniform = terra() : double
    return ([double](C.rand()) / [double](C.RAND_MAX))
end

rand.log_categorical = terra(x : &double, n : int) : int
    var log_denom = math.logsumexp(x, n)
    var r = rand.uniform()
    var accum : double = 0.0
    var i : int
    for i=0,n do
        accum = accum + math.exp(x[i] - log_denom)
        if accum > r then
            return i
        end
    end
    return n-1
end

return rand
