rand = require("rand")
tmath = require("tmath")
trace = require("trace")
stdio = terralib.includec("stdio.h")

P = {}

P._flip = terra(p: double) : bool
        return rand.uniform() <= p 
    end
P.flip = trace.makeModule(
    P._flip,
    terra(val : bool, p: double) : double
        if val then
            return tmath.log(p)
        else
            return tmath.log(1.0 - p)
        end
    end)

P._normal = terra(mu: double, sigma: double) : double
            var u:double, v:double, x:double, y:double, q:double
            repeat
                u = 1.0 - rand.uniform()
                v = 1.7156 * (rand.uniform() - 0.5)
                x = u - 0.449871
                y = tmath.fabs(v) + 0.386595
                q = x*x + y*(0.196*y - 0.25472*x)
            until not(q >= 0.27597 and (q > 0.27846 or v*v > -4 * u * u * tmath.log(u)))
            var result = mu + sigma*v/u
            -- stdio.printf("normal simulate mu=%0.3f, sigma=%0.3f -> x=%0.3f\n", mu, sigma, result)
            return result
        end
P.normal = trace.makeModule(
        P._normal,
        -- from quicksand
        terra(x: double, mu: double, sigma: double) : double
            -- stdio.printf("normal regenerate, x=%0.3f, mu=%0.3f, sigma=%0.3f\n", x, mu, sigma)
            var xminusmu = x - mu
            return -.5*(1.8378770664093453 + 2*tmath.log(sigma) + xminusmu*xminusmu/(sigma*sigma))
        end)

return P
