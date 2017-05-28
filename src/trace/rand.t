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

terra seedtime() 
    C.seedtime()
end

terra uniform() : float
    return ([float](C.rand()) / [float](C.RAND_MAX))
end

terra flip(weight : float) : bool
    return uniform() <= weight
end


main = terra()
    var x = uniform()
    C.printf("%f\n", x)
end
main()

