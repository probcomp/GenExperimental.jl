#declare T_Stone34 =
texture {
    pigment {
        gradient x
        turbulence 0.75
        frequency 6
        scale 0.5
        rotate <0, -45, 30>
        color_map {
            [0.02 color rgbt <0.85, 0.85, 0.85, 0.0> ]
            [0.10 color rgbt <1.00, 1.00, 1.00, 0.0> ]
            [1.00 color rgbt <0.85, 0.85, 0.85, 0.0> ]
        }
    }
}
