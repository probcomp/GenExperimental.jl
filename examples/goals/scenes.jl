function generate_scene_a()
    trace = Trace()
    intervene!(trace, "is-tree-1", true)
    intervene!(trace, "tree-1", Tree(Point(30, 20), 10.))
    intervene!(trace, "is-tree-2", true)
    intervene!(trace, "tree-2", Tree(Point(83, 80), 10.))
    intervene!(trace, "is-tree-3", true)
    intervene!(trace, "tree-3", Tree(Point(80, 40), 10.))

    wall_height = 30.
    intervene!(trace, "is-wall-1", true)
    intervene!(trace, "wall-1", Wall(Point(20., 40.), 1, 40., 2., wall_height))
    intervene!(trace, "is-wall-2", true)
    intervene!(trace, "wall-2", Wall(Point(60., 40.), 2, 40., 2., wall_height))
    intervene!(trace, "is-wall-3", true)
    intervene!(trace, "wall-3", Wall(Point(60.-15., 80.), 1, 15. + 2., 2., wall_height))
    intervene!(trace, "is-wall-4", true)
    intervene!(trace, "wall-4", Wall(Point(20., 80.), 1, 15., 2., wall_height))
    intervene!(trace, "is-wall-5", true)
    intervene!(trace, "wall-5", Wall(Point(20., 40.), 2, 40., 2., wall_height))

    boundary_wall_height = 2.
    intervene!(trace, "is-wall-6", true)
    intervene!(trace, "wall-6", Wall(Point(0., 0.), 1, 100., 2., boundary_wall_height))
    intervene!(trace, "is-wall-7", true)
    intervene!(trace, "wall-7", Wall(Point(100., 0.), 2, 100., 2., boundary_wall_height))
    intervene!(trace, "is-wall-8", true)
    intervene!(trace, "wall-8", Wall(Point(0., 100.), 1, 100., 2., boundary_wall_height))
    intervene!(trace, "is-wall-9", true)
    intervene!(trace, "wall-9", Wall(Point(0., 0.), 2, 100., 2., boundary_wall_height))

    # prevent the program from adding new wall or trees
    intervene!(trace, "is-tree-4", false)
    intervene!(trace, "is-wall-10", false)

    # add the drone starting position
    constrain!(trace, "start", Point(90., 10.))
    return trace
end

function generate_scene_b()
    trace = generate_scene_a()

    # change the walls to add a bottom passageway
    wall_height = 30.
    delete!(trace, "wall-1")
    intervene!(trace, "wall-1", Wall(Point(20., 40.), 1, 15., 2., wall_height))
    delete!(trace, "is-tree-10")
    delete!(trace, "is-wall-10")
    intervene!(trace, "is-wall-10", true)
    intervene!(trace, "wall-10", Wall(Point(60.- 15, 40.), 1, 15., 2., wall_height))
    intervene!(trace, "is-wall-11", false)
    return trace
end
