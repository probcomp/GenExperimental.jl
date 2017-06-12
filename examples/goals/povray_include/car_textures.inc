#include "colors.inc"
#include "textures.inc"
#include "glass.inc"
#include "metals.inc"
#include "golds.inc"
#include "stones.inc"
#include "woods.inc"
#include "shapes.inc"
#include "shapes2.inc"
#include "functions.inc"
#include "math.inc"
#include "transforms.inc"
#declare Car_Body_Texture = 
         texture{ pigment{ color rgb<1,1,1>*1.2}
                //normal { radial sine_wave frequency 30 scale 0.25 }
                  finish { phong 1}
                 }
#declare Car_Top_Texture = 
         texture { pigment{ color rgb< 1, 0.95, 0.95>*1.1  } // very light red  
                // normal { bumps 0.5 scale 0.05 }
                   finish { diffuse 0.9 phong 1 reflection 0.00}
                 } // end of texture 
#declare Car_Seat_Texture = 
          texture { pigment{ color rgb< 1, 0.90, 0.85>*1.2  }// very light brown  
                   normal { bumps 0.15 scale 0.05 }
                   finish { phong 1 reflection 0.00}
                 } // end of texture 
#declare Car_Inside_Texture=
          texture { pigment{ color rgb< 1, 0.87, 0.75>  }// very light brown  
                // normal { bumps 0.5 scale 0.05 }
                   finish { phong 1 reflection 0.00}
                 } // end of texture 

#declare Wheel_Rubber_Texture = 
         texture { pigment{ color rgb< 1, 1, 1>*0.15 } //  color Gray25
                   normal { bumps 0.5 scale 0.008 }
                   finish { phong 0.2 reflection 0.00}
                 } // end of texture 
#declare Wheel_Metal_Texture = 
         texture { Chrome_Metal
                   // pigment{ color rgb<1,0,0>} 
                    normal { bumps 0.5 scale 0.005 } 
                    finish { phong 1}
                 } // end of texture 
#declare Bumper_Metal_Texture = 
         texture { Polished_Chrome
                   // pigment{ color rgb<1,0,0>} 
                   finish {   phong 1}
                 } // end of texture 
#declare Chassis_Inside_Texture = 
         texture { pigment{ color rgb< 1, 1, 1>*0.45 } //  color Gray55
                // normal { bumps 0.5 scale 0.05 }
                   finish { phong 1 reflection 0.00}
                 } // end of texture 
