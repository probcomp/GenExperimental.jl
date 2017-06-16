abstract Obstacle

immutable Point
    x::Float64
    y::Float64
end

array(point::Point) = [point.x, point.y]

function dist(a::Point, b::Point)
    dx = a.x - b.x
    dy = a.y - b.y
    sqrt(dx * dx + dy * dy)
end

type Scene
    xmin::Float64
    xmax::Float64
    ymin::Float64
    ymax::Float64
    obstacles::Array{Obstacle,1}
    function Scene(xmin::Real, xmax::Real, ymin::Real, ymax::Real)
        new(xmin, xmax, ymin, ymax, [])
    end
end

function add!(scene::Scene, object::Obstacle)
    push!(scene.obstacles, object)
end


function ccw(a::Point, b::Point, c::Point)
    (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)
end

function line_intersects_line(a1::Point, a2::Point, b1::Point, b2::Point)
        # http://algs4.cs.princeton.edu/91primitives/
        if ccw(a1, a2, b1) * ccw(a1, a2, b2) > 0
            return false
        end
        if ccw(b1, b2, a1) * ccw(b1, b2, a2) > 0
            return false
        end
        return true
end

function line_of_site(scene::Scene, a::Point, b::Point)
    for obstacle in scene.obstacles
        if intersects_path(obstacle, a, b)
            return false
        end
    end
    return true
end

immutable Polygon <: Obstacle
    vertices::Array{Point,1}
end

function intersects_path(poly::Polygon, path_start::Point, path_end::Point)
    n = length(poly.vertices)
    for start_vertex_idx=1:n
        end_vertex_idx = start_vertex_idx % n + 1 # loop over to 1
        v1 = poly.vertices[start_vertex_idx]
        v2 = poly.vertices[end_vertex_idx]
        if line_intersects_line(v1, v2, path_start, path_end)
            return true
        end
    end
    return false
end

immutable Wall <: Obstacle
    start::Point
    orientation::Int # x is 1, y is 2
    length::Float64
    thickness::Float64
    height::Float64
    poly::Polygon
    name::Symbol # for serialization to Javascript for visualization, to preserve classname (TODO do this generically. can we override serialization?)
    function Wall(start::Point, orientation::Int, length::Float64,
                  thickness::Float64, height::Float64)
        if orientation != 1 && orientation != 2
            error("orientation must be either 1 (x) or 2 (y)")
        end
        vertices = Array{Point,1}(4)
        vertices[1] = start
        dx = orientation == 1 ? length : thickness
        dy = orientation == 2 ? length : thickness
        vertices[2] = Point(start.x + dx, start.y)
        vertices[3] = Point(start.x + dx, start.y + dy)
        vertices[4] = Point(start.x, start.y + dy)
        poly = Polygon(vertices)
        new(start, orientation, length, thickness, height, poly, :Wall)
    end
end

function intersects_path(wall::Wall, path_start::Point, path_end::Point)
    intersects_path(wall.poly, path_start, path_end)
end

immutable Tree <: Obstacle
    center::Point
    size::Float64
    poly::Polygon
    name::Symbol # for serialization to Javascript for visualization, to preserve classname (TODO do this generically. can we override serialization?)
    function Tree(center::Point; size::Float64=10.)
        vertices = Array{Point,1}(4)
        vertices[1] = Point(center.x - size/2, center.y - size/2)
        vertices[2] = Point(center.x + size/2, center.y - size/2)
        vertices[3] = Point(center.x + size/2, center.y + size/2)
        vertices[4] = Point(center.x - size/2, center.y + size/2)
        poly = Polygon(vertices)
        new(center, size, poly, :Tree)
    end
end

function intersects_path(tree::Tree, path_start::Point, path_end::Point)
    intersects_path(tree.poly, path_start, path_end)
end
