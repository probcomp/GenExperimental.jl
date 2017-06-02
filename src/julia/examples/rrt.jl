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

function rrt{C,U}(scheme::RRTScheme{C,U}, init::C, iters::Int, dt::Float64)
    tree = RRTTree{C,U}(init) # init is the root of tree
    for iter=1:iters
        rand_conf::C = random_config(scheme)
        near_node::RRTNode{C,U} = nearest_neighbor(scheme, rand_conf, tree)
        control::U = select_control(scheme, rand_conf, near_node.conf, dt) # u
        new_conf::C = predict_config(scheme, near_node.conf, control, dt)
        add_node!(tree, near_node, new_conf, control, dt)
        # record control so the sequence of controls needed can be reconstructed
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


immutable HolonomicPointControl
    # defines a direction (ux and uy together form a unit vector)
    ux::Float64
    uy::Float64
    function HolonomicPointControl(dx::Float64, dy::Float64)
        length = sqrt(dx * dx + dy * dy)
        new(dx / length, dy / length)
    end
end

immutable HolonomicPointRRTScheme <: RRTScheme{Point,HolonomicPointControl}
    xmin::Float64
    xmax::Float64
    ymin::Float64
    ymax::Float64
end

function random_config(scheme::HolonomicPointRRTScheme)
    x = rand() * (scheme.xmax - scheme.xmin) + scheme.xmin
    y = rand() * (scheme.ymax - scheme.ymin) + scheme.ymin
    Point(x, y)
end

function select_control(scheme::HolonomicPointRRTScheme, 
                       target_conf::Point, start_conf::Point, dt::Float64)
    # don't overshoot the point:
    d = dist(target_conf, start_conf)
    local scale::Float64
    if d < dt
        scale = d / dt
    else
        scale = 1.0
    end
    # go in the direction of target_conf from start_conf 
    # TODO take into account constriants! what if we fail?
    HolonomicPointControl(scale * (target_conf.x - start_conf.x),
                          scale * (target_conf.y - start_conf.y))
end

function predict_config(scheme::HolonomicPointRRTScheme, conf::Point, 
                        control::HolonomicPointControl, dt::Float64)
    x = conf.x + control.ux * dt
    y = conf.y + control.uy * dt
    Point(x, y)
end

# TODO add constraints!!

# example
using PyPlot

function render(scheme::HolonomicPointRRTScheme, 
                tree::RRTTree{Point,HolonomicPointControl})
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
    ax = plt[:gca]()
    ax[:set_xlim](scheme.xmin, scheme.xmax)
    ax[:set_ylim](scheme.ymin, scheme.ymax)
end

# plot them
scheme = HolonomicPointRRTScheme(0, 100, 0, 100)
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








