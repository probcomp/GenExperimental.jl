# generic 

immutable RRTNode{C,U}
    conf::C
    # the previous configuration and the control needed to produce this node's
    # configuration from the previous configuration may be null if it is the
    # root node
    parent::Nullable{RRTNode{C,U}}
    control::Nullable{U}
    dt::Float64
end

type RRTTree{C,U}
    nodes::Array{RRTNode{C,U}, 1}
    function RRTTree(root_conf::C)
        nodes = Array{RRTNode{C,U},1}()
        push!(nodes, RRTNode(root_conf, Nullable{RRTNode{C,U}}(), Nullable{U}(), NaN))
        new(nodes)
    end
end

function add_node!{C,U}(tree::RRTTree{C,U}, parent::RRTNode{C,U}, new_conf::C, control::U, dt::Float64)
    node = RRTNode(new_conf, Nullable{RRTNode{C,U}}(parent), Nullable{U}(control), dt)
    push!(tree.nodes, node)
end

function root{C,U}(tree::RRTTree{C,U})
    return tree.nodes[1]
end

abstract RRTScheme{C,U}

function nearest_neighbor{C,U}(scheme::RRTScheme{C,U}, conf::C,
                               tree::RRTTree{C,U})
    nearest::RRTNode{C,U} = root(tree)
    best_dist::Float64 = dist(nearest.conf, conf)
    for node::RRTNode{C,U} in tree.nodes
        d = dist(node.conf, conf)
        if d < best_dist
            best_dist = d
            nearest = node
        end
    end
    nearest
end

immutable SelectControlResult{C,U}
    start_conf::C
    new_conf::C # we will have to test if it is close to the goal or not TODO separate function does this
    control::U
    failed::Bool # new_conf is undefined in this case
end

function rrt{C,U}(scheme::RRTScheme{C,U}, init::C, iters::Int, dt::Float64)
    tree = RRTTree{C,U}(init) # init is the root of tree
    for iter=1:iters
        rand_conf::C = random_config(scheme)
        near_node::RRTNode{C,U} = nearest_neighbor(scheme, rand_conf, tree)
        result = select_control(scheme, rand_conf, near_node.conf, dt)
        if !result.failed
            add_node!(tree, near_node, result.new_conf, result.control, dt)
        end
    end
    tree
end

# TODO: do it for a non-holonomic car :) -- three point turns... going in reverse..

# specialized for point-agent in plane

immutable Point
    x::Float64
    y::Float64
end

function dist(a::Point, b::Point)
    dx = a.x - b.x
    dy = a.y - b.y
    sqrt(dx * dx + dy * dy)
end

abstract Obstacle

immutable Scene
    xmin::Float64
    xmax::Float64
    ymin::Float64
    ymax::Float64
    obstacles::Array{Obstacle,1}
end

immutable Polygon <: Obstacle
    vertices::Array{Point,1}
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

using PyCall
@pyimport matplotlib.path as mplPath
@pyimport matplotlib.patches as patches
function render(poly::Polygon)
    points = map((p) -> Float64[p.x, p.y], poly.vertices)
    path = mplPath.Path(points)
    ax = plt[:gca]()
    patch = patches.PathPatch(path, facecolor="black")
    ax[:add_patch](patch)
end

immutable HolonomicPointRRTScheme <: RRTScheme{Point,Point}
    scene::Scene
end

function random_config(scheme::HolonomicPointRRTScheme)
    x = rand() * (scheme.scene.xmax - scheme.scene.xmin) + scheme.scene.xmin
    y = rand() * (scheme.scene.ymax - scheme.scene.ymin) + scheme.scene.ymin
    Point(x, y)
end


function select_control(scheme::HolonomicPointRRTScheme, 
                        target_conf::Point, start_conf::Point, dt::Float64)


    dist_to_target = dist(start_conf, target_conf)
    diff = Point(target_conf.x - start_conf.x, target_conf.y - start_conf.y)
    distance_to_move = min(dt, dist_to_target)
    scale = distance_to_move / dist_to_target
    control = Point(scale * diff.x, scale * diff.y)

    obstacles = scheme.scene.obstacles

    # go in the direction of target_conf from start_conf 
    new_conf = Point(start_conf.x + control.x, start_conf.y + control.y)

    # test the obstacles
    failed = false
    for obstacle in obstacles
        if intersects_path(obstacle, start_conf, new_conf)
            # NOTE: could do more intelligent things like backtrack until you succeed
            failed = true
            break
        end
    end
    SelectControlResult(start_conf, new_conf, control, failed)
end

# example
using PyPlot

function render(scheme::HolonomicPointRRTScheme, 
                tree::RRTTree{Point,Point})
    scene = scheme.scene
    for node in tree.nodes
        if !isnull(node.parent)
            # it is not the root
            x1 = get(node.parent).conf.x
            y1 = get(node.parent).conf.y
            x2 = node.conf.x
            y2 = node.conf.y
            plt[:plot]([x1, x2], [y1, y2], color="k")
        end
    end
    for obstacle in scene.obstacles
        render(obstacle)
    end
    ax = plt[:gca]()
    ax[:set_xlim](scene.xmin, scene.xmax)
    ax[:set_ylim](scene.ymin, scene.ymax)
end

# plot them
obstacles = [Polygon([Point(30, 30), Point(80, 30), Point(80, 35), Point(30, 35)])] # one tree in the center of hte mpap
scene = Scene(0, 100, 0, 100, obstacles)
scheme = HolonomicPointRRTScheme(scene)
plt[:figure](figsize=(30, 10))
for (i, iters) in enumerate([100, 1000, 2000])
    println("iters: $iters")
    plt[:subplot](1, 3, i)
    println("rrt..")
    @time tree = rrt(scheme, Point(50, 50), iters, 1.)
    println("rendering..")
    render(scheme, tree)
end
plt[:savefig]("rrt.png")








