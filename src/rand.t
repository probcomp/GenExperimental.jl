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

local terra seedtime() 
    C.seedtime()
end

local terra uniform() : float
    return ([float](C.rand()) / [float](C.RAND_MAX))
end

rand = {
    seedtime = seedtime,
    uniform = uniform
}

return rand
