#declare Stem_Texture = 
 texture{ pigment{ color rgb< 0.70, 0.5, 0.30>*0.25 } 
          normal { bumps 1.00 scale <0.025,0.075,0.025> }
          finish { phong 0.2 reflection 0.00}         
        } // end of texture 
//------------------------------------------------------- 
#declare Leaves_Texture_1 = 
 texture{ pigment{ color rgbf< 0.25, 0.36, 0.0, 0.1>*1.3  }   
          normal { bumps 0.15 scale 0.05 }
          finish { phong 1 reflection 0.00}
        } // end of texture 
//-------------------------------------------------------- 
#declare Leaves_Texture_2 = 
 texture{ pigment{ color rgbf< 0.25, 0.35, 0.0, 0.1>*1.2  }   
          normal { bumps 0.15 scale 0.05 }
          finish { phong 0.2 reflection 0.00}
        } // end of texture 
