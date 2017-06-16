
# --- Generic RRT --------------------

immutable RRTNode{C,U}
    conf::C
    # the previous configuration and the control needed to produce this node's
    # configuration from the previous configuration may be null if it is the
    # root node
    parent::Nullable{RRTNode{C,U}}
    control::Nullable{U}
    cost_from_start::Float64 # cost (e.g. distance) of moving from start to us
end

type RRTTree{C,U}
    nodes::Array{RRTNode{C,U}, 1}
    function RRTTree(root_conf::C)
        nodes = Array{RRTNode{C,U},1}()
        push!(nodes, RRTNode(root_conf, Nullable{RRTNode{C,U}}(), Nullable{U}(), 0.0))
        new(nodes)
    end
end

function Base.show{C,U}(io::IO, tree::RRTTree{C,U})
    write(io, "RRTTree with $(length(tree.nodes)) nodes")
end

function add_node!{C,U}(tree::RRTTree{C,U}, parent::RRTNode{C,U}, new_conf::C, control::U, cost_from_start::Float64)
    node = RRTNode(new_conf, Nullable{RRTNode{C,U}}(parent), Nullable{U}(control), cost_from_start)
    push!(tree.nodes, node)
    return node
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
    new_conf::C
    control::U
    failed::Bool # new_conf is undefined in this case
    cost::Float64 # cost of this control action (e.g. distance)
end

function rrt{C,U}(scheme::RRTScheme{C,U}, init::C, iters::Int, dt::Float64)
    tree = RRTTree{C,U}(init) # init is the root of tree
    for iter=1:iters
        rand_conf::C = random_config(scheme)
        near_node::RRTNode{C,U} = nearest_neighbor(scheme, rand_conf, tree)
        result = select_control(scheme, rand_conf, near_node.conf, dt)
        if !result.failed
            cost_from_start = near_node.cost_from_start + result.cost
            add_node!(tree, near_node, result.new_conf, result.control, cost_from_start)
        end
    end
    tree
end

# --- RRT scheme for Holonomic 2D point --------------------

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
    cost = distance_to_move
    SelectControlResult(start_conf, new_conf, control, failed, cost)
end

function rrt_demo()
    # plot them
    obstacles = [Polygon([Point(30, 30), Point(80, 30), Point(80, 35), Point(30, 35)])] # one tree in the center of hte mpap
    scene = Scene(0, 100, 0, 100, obstacles)
    scheme = HolonomicPointRRTScheme(scene)
    plt[:figure](figsize=(30, 10))
    rendering = PlotRendering()
    for (i, iters) in enumerate([100, 1000, 2000])
        println("iters: $iters")
        plt[:subplot](1, 3, i)
        println("rrt..")
        @time tree = rrt(scheme, Point(50, 50), iters, 1.)
        println("rendering..")
        render(rendering, scene)
        render(rendering, tree)
    end
    plt[:savefig]("rrt.png")
end
#rrt_demo()


