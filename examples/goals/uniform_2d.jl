# need to import these if you are going to use your own modules
import Gen.simulate
import Gen.regenerate

# module for 2D uniform simulate

immutable Uniform2D <: Gen.Module{Point} end

function simulate(::Uniform2D, xmin::Real, xmax::Real, ymin::Real, ymax::Real)
    (x, log_weight_x) = simulate(Gen.Uniform(), xmin, xmax)
    (y, log_weight_y) = simulate(Gen.Uniform(), ymin, ymax)
    (Point(x, y), log_weight_x + log_weight_y)
end

function regenerate(::Uniform2D, point::Point, xmin::Real, xmax::Real, ymin::Real, ymax::Real)
    log_weight_x = regenerate(Gen.Uniform(), point.x, xmin, xmax)
    log_weight_y = regenerate(Gen.Uniform(), point.y, ymin, ymax)
    log_weight_x + log_weight_y
end

register_module(:uniform_2d, Uniform2D())
uniform_2d = (xmin::Real, xmax::Real, ymin::Real, ymax::Real) -> simulate(Uniform2D(), xmin, xmax, ymin, ymax)[1]
