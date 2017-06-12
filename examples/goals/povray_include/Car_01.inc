// POV-Ray 3.6 / 3.7 include file "Car_01.inc" 
// author: Friedrich A, Lohmueller, Nov-2006 / May-2014
// homepage: http://www.f-lohmueller.de/
// email: Friedrich.Lohmueller_aT_t-online.de
//------------------------------------------------------------------------
#ifndef( Car_01_Inc_Temp)
#declare Car_01_Inc_Temp = version;
#version 3.6;

//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------
#ifndef(  Shapes_Inc_Temp)
#include "shapes.inc"                                             
#end

//------------------------------------------------------------------------------ /////////  
//---------------------------------------------------------------/////////// 
#macro Car_01 ( Front_Wheel_Angle__, // 
                Coupe_ON__, //

                Car_Body_Texture__, //
                Car_Top_Texture__, // 
                Car_Seat_Texture__, //
                Car_Inside_Texture__, //
                Wheel_Rubber_Texture__,
                Wheel_Metal_Texture__, 
                Bumper_Metal_Texture__,
                Chassis_Inside_Texture__ //
              ) //-------------------------------------------------- 
// -----------------------------------------------------------------
#local D = 0.0001; 
//-----------------------------------------
#local Wheel_Scale = 0.85; 
#local Base_Wheel_R = 0.45; 
#local Wheel_R = Wheel_Scale* Base_Wheel_R; 
//----------------------------------------- object Wheel 
#local Wheel_ = 
union{ 
 intersection{
  torus{ 0.35,0.15  rotate<90,0,0> }
  cylinder{ <0,0,-0.20>,<0,0,0.20>, 0.47}
  texture{Wheel_Rubber_Texture__}
 }// end intersection
 sphere{ <0,0,0>,0.45 scale<1,1,0.25> 
          texture {Wheel_Metal_Texture__}  
 } // end of sphere -----------------------
translate<0,0.45,0>
scale Wheel_Scale 
} // end of union ------------------------- end of object Wheel 
//------------------------------------------------------------------

// -----------------------------------------------------------------
#local Car_Lenght = 5.00; // total length
#local Car_Low = 0.25; // base height
#local Car_Tween = 1.00; // tween height
#local Car_Top = 1.60; // top height
#local Front_X = 0.70; // length over front wheels 
#local Body_H =Car_Top- Car_Low;  
// -----------------------------------------------------------------
#local Half = 0.90; // Halbe Spurweite
#local Z_Over = 0.10; // Halbe Spurweite Überhang
#local Half_Brutto = Half + Z_Over; 
#local Axis_Distance = 3.20; // Achs-Abstand
//--------------------------------------------
#local Chassis_Base_R = 0.15; 
//------------------------------------------------------------------

//------------------------------------------------------------------ macro Wheels 
#macro Wheels (Front_Wheel___Angle)  
//------------------------------------ Chassis
union{
object{ Wheel_ rotate<0,Front_Wheel___Angle,0> 
        translate<0,0,-Half>} 
object{ Wheel_ scale<1,1,-1> rotate<0,Front_Wheel___Angle,0> 
        translate<0,0, Half>} 
object{ Wheel_ translate<-Axis_Distance,0,-Half+0.05>} 
object{ Wheel_ scale<1,1,-1> translate<-Axis_Distance,0, Half-0.05>} 
} // end of union  ------------ 
#end // ------------------------------------------------------------ end of macro "Wheels"
//----------------------------------------------------------------------------------------

#local Seat = //---------------------------------------------------- Seat
union{ 
 object{ Round_Box(<0.0,-0.085,-0.35>,<0.50,0.075,0.035>, 0.05, 0) }
 object{ Round_Box(<0.0,-0.075,-0.35>,<0.50,0.085,0.350>, 0.05, 0) rotate<0,0,100> }
         texture{Car_Seat_Texture}
       } // --------------------------------------------------------- end of "Seat"

//------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------- assembled car

union{ 
object { Wheels ( Front_Wheel_Angle__) }
difference{ //----------------------------------------------------------------------------- car base
  object{ Round_Box(<-Car_Lenght+Front_X,Car_Low,-Half_Brutto>,<Front_X,Car_Tween,Half_Brutto>, 0.15, 0) }  
  union{ 
  cylinder{ <0,0,-Half_Brutto-D>,<0,0,-Half_Brutto/2>,Wheel_R+0.12  translate<             0,Wheel_R-0.05,0> }        
  cylinder{ <0,0, Half_Brutto/2>,<0,0, Half_Brutto+D>,Wheel_R+0.12  translate<             0,Wheel_R-0.05,0> }        
  cylinder{ <0,0,-Half_Brutto-D>,<0,0,-Half_Brutto/2>,Wheel_R+0.12 scale<1,0.5,1> translate<-Axis_Distance,Wheel_R-0.05,0> }
  cylinder{ <0,0, Half_Brutto/2>,<0,0, Half_Brutto+D>,Wheel_R+0.12 scale<1,0.5,1> translate<-Axis_Distance,Wheel_R-0.05,0> }
  texture{ Chassis_Inside_Texture__ } } 
intersection{ // inside space !!!
 object{ Round_Box( //------------------------- upper front
        <-Axis_Distance-0.5,          0.01,-0.20>,
        <           -0.52,Body_H-0.03, 0.20>, Chassis_Base_R, 0)  
        scale<1,1,(Half_Brutto-0.05)*1/0.20+D>*0.94
     matrix<1  , 0, 0, //  matrix-shear_y_to_x
           -0.55, 1, 0,
            0  , 0, 1,
            0  , 0, 0>
        translate<0.15,Car_Low+0.02,0.00> }
 object{ Round_Box( //------------------------- upper back
        <-Axis_Distance+1,          0.01,-0.20>,
        <            0.52,Body_H-0.03, 0.20>, Chassis_Base_R, 0)  
        scale<1,1,(Half_Brutto-0.05)*1/0.20>*0.94
     matrix<1  , 0, 0, //  matrix-shear_y_to_x
            1.25, 1, 0,
            0  , 0, 1,
            0  , 0, 0>
        translate<-1.7,Car_Low+0.02+D,0.00> }
 cylinder{ <0,0,-Half_Brutto-D>,<0,0, Half_Brutto+D>,Wheel_R+0.14  translate<             0,Wheel_R-0.05,0> inverse }        
 cylinder{ <0,0,-Half_Brutto-D>,<0,0, Half_Brutto+D>,Wheel_R+0.14  translate<-Axis_Distance,Wheel_R-0.05,0> inverse }        
 texture{ Car_Inside_Texture }        
 } // end upper inside space
} // end of diff ----------------------------------------------------------------------------

 
difference{ //-------------------------------------------------------------------------------// Car Top:
intersection{
 object{ Round_Box( //------------------------- upper front
        <-Axis_Distance-0.5,          0,-0.20>,
        <           -0.50,Body_H-0.05, 0.20>, Chassis_Base_R+0.001, 0)  
        scale<1,1,(Half_Brutto-0.05)*1/0.20+D>
     matrix<1  , 0, 0, //  matrix-shear_y_to_x
           -0.55, 1, 0,
            0  , 0, 1,
            0  , 0, 0>
        translate<0.15,Car_Low+0.02,0.00>}
 object{ Round_Box( //------------------------- upper back
        <-Axis_Distance+1,          0,-0.20>,
        <            0.50,Body_H-0.05, 0.20>, Chassis_Base_R+0.001, 0)  
        scale<1,1,(Half_Brutto-0.05)*1/0.20>
     matrix<1  , 0, 0, //  matrix-shear_y_to_x
            1.25, 1, 0,
            0  , 0, 1,
            0  , 0, 0>
        translate<-1.8,Car_Low+0.02+D,0.00>}
 cylinder{ <0,0,-Half_Brutto-D>,<0,0, Half_Brutto+D>,Wheel_R+0.13  translate<             0,Wheel_R-0.05,0> inverse }        
 cylinder{ <0,0,-Half_Brutto-D>,<0,0, Half_Brutto+D>,Wheel_R+0.13  translate<-Axis_Distance,Wheel_R-0.05,0> inverse }        
// texture{ pigment{ color rgb 0.6}
 texture{ Car_Top_Texture__ }        
} // end upper outside
intersection{
 object{ Round_Box( //------------------------- upper front
        <-Axis_Distance-0.5,          0,-0.20>,
        <           -0.52,Body_H-0.05, 0.20>, Chassis_Base_R, 0)  
        scale<1,1,(Half_Brutto-0.05)*1/0.20+D>*0.98
     matrix<1  , 0, 0, //  matrix-shear_y_to_x
           -0.55, 1, 0,
            0  , 0, 1,
            0  , 0, 0>
        translate<0.15,Car_Low+0.02,0.00>}
 object{ Round_Box( //------------------------- upper back
        <-Axis_Distance+1,          0,-0.20>,
        <            0.52,Body_H-0.05, 0.20>, Chassis_Base_R, 0)  
        scale<1,1,(Half_Brutto-0.05)*1/0.20>*0.98
     matrix<1  , 0, 0, //  matrix-shear_y_to_x
            1.25, 1, 0,
            0  , 0, 1,
            0  , 0, 0>
        translate<-1.8,Car_Low+0.02+D,0.00> }
 cylinder{ <0,0,-Half_Brutto-D>,<0,0, Half_Brutto+D>,Wheel_R+0.14  translate<             0,Wheel_R-0.05,0> inverse }        
 cylinder{ <0,0,-Half_Brutto-D>,<0,0, Half_Brutto+D>,Wheel_R+0.14  translate<-Axis_Distance,Wheel_R-0.05,0> inverse }        
 texture{ Car_Inside_Texture }         
} // end upper inside

// ------------------------------------------------------------- side window front
// linear prism in z-direction: from ,to ,number of points (first = last)
prism { -Half_Brutto-2*D ,Half_Brutto+2*D  , 5
       <-0.80, 0.00>,  // first point
       <-0.03, 0.00>, 
       <-0.26, 0.42>, 
       <-0.80, 0.42>, 
       <-0.80, 0.00>   // last point = first point!!!!
       rotate<-90,0,0> scale<1,1,-1> //turns prism in z direction! Don't change this line! 

       texture { pigment{ color White} 
                 finish { diffuse 0.9 phong 1.0}
               } // end of texture

        translate<-0.95,Car_Tween-0.04+D,0>
       } // ---------------------------------------------------------
// linear prism in z-direction: ------------------------------- side window back half
prism { -Half_Brutto-2*D ,Half_Brutto+2*D  , 6
       <-0.85, 0.00>,  // first point
       <-1.52, 0.00>, 
       <-1.52, 0.30>, 
       <-1.42, 0.42>, 
       <-0.85, 0.42>, 
       <-0.85, 0.00>   // last point = first point!!!!
       rotate<-90,0,0> scale<1,1,-1> //turns prism in z direction! Don't change this line! 

       texture { pigment{ color White} 
                 finish { diffuse 0.9 phong 1.0}
               } // end of texture

        translate<-0.95,Car_Tween-0.04+D,0>
       } // ---------------------------------------------------------
// ------------------------------------------------------------------ front window hole
object { // Round_Box(A, B, WireRadius, UseMerge)
         Round_Box(<-0.5,0,-Half_Brutto+0.10>,<0.5,0.47,Half_Brutto-0.10>, 0.05   , 0)  
         scale<1,1,1>  rotate<0, 0,20> translate<-0.85,Car_Tween-0.01+D,0>
       } // ---------------------------------------------------------
// ------------------------------------------------------------------ backside window hole
object { // Round_Box(A, B, WireRadius, UseMerge)
         Round_Box(<-0.5,0.10,-Half_Brutto+0.20>,<0.5,0.45,Half_Brutto-0.20>, 0.10   , 0)  
         scale<1,1,1>  rotate<0, 0,-10> translate<-2.8,Car_Tween-0.04+D,0>
       } // ---------------------------------------------------------
//-------------------------------------- Coupé 
#if (Coupe_ON__ = 1)
box{ <-3.50,0,-Half_Brutto-0.05>,<-1.20,0.70,Half_Brutto+0.05> translate<0,Car_Tween+0.01+D,0>}  
box{ <-1,0,-Half_Brutto-0.05>,< 1,0.70,Half_Brutto+0.05>rotate<0,0,20> translate<-1.0,Car_Top-0.12+D,0>}  
#end // Coupé 

} // end top difference

//-------------------------------------------------------------------------------------------
//--------------------------------------------------------------------- adding window glass! 
//--------------------------------------------------------------------- adding window glass! 
difference{ 
intersection{
 object{ Round_Box( //------------------------- upper front
        <-Axis_Distance-0.5,          0,-0.20>,
        <           -0.52,Body_H-0.05, 0.20>, 0.201, 1)  
        scale<1,1,(Half_Brutto-0.05)*1/0.20+D>*0.98
     matrix<1  , 0, 0, //  matrix-shear_y_to_x
           -0.55, 1, 0,
            0  , 0, 1,
            0  , 0, 0>
        translate<0.15,Car_Low+0.02,0.00>}
 // Coupé            
 // #if (Coupe_ON__ = 0) 
 object{ Round_Box( //------------------------- upper back
        <-Axis_Distance+1,          0,-0.201>,
        <            0.52,Body_H-0.05, 0.201>, 0.201, 1)  
        scale<1,1,(Half_Brutto-0.05)*1/0.20>*0.98
     matrix<1  , 0, 0, //  matrix-shear_y_to_x
            1.25, 1, 0,
            0  , 0, 1,
            0  , 0, 0>
        translate<-1.8,Car_Low+0.02+D,0.00>}
// #end
 cylinder{ <0,0,-Half_Brutto-D>,<0,0, Half_Brutto+D>,Wheel_R+0.125  translate<             0,Wheel_R-0.05,0> inverse }        
 cylinder{ <0,0,-Half_Brutto-D>,<0,0, Half_Brutto+D>,Wheel_R+0.125  translate<-Axis_Distance,Wheel_R-0.05,0> inverse }        
} // intersection 
intersection{
 object{ Round_Box( //------------------------- upper front
        <-Axis_Distance-0.5,          0,-0.20>,
        <           -0.565,Body_H-0.08, 0.20>, Chassis_Base_R, 1)  
        scale<1,1,(Half_Brutto-0.05)*1/0.20+D>*0.96
     matrix<1  , 0, 0, //  matrix-shear_y_to_x
           -0.55, 1, 0,
            0  , 0, 1,
            0  , 0, 0>
        translate<0.15,Car_Low+0.02,0.00>}
 object{ Round_Box( //------------------------- upper back
        <-Axis_Distance+1,          0,-0.20>,
        <            0.52,Body_H-0.08, 0.20>, Chassis_Base_R, 1)  
        scale<1,1,(Half_Brutto-0.05)*1/0.20>*0.96
     matrix<1  , 0, 0, //  matrix-shear_y_to_x
            1.25, 1, 0,
            0  , 0, 1,
            0  , 0, 0>
        translate<-1.8,Car_Low+0.02+D,0.00>}
 cylinder{ <0,0,-Half_Brutto-D>,<0,0, Half_Brutto+D>,Wheel_R+0.122  translate<             0,Wheel_R-0.05,0> inverse }        
 cylinder{ <0,0,-Half_Brutto-D>,<0,0, Half_Brutto+D>,Wheel_R+0.122  translate<-Axis_Distance,Wheel_R-0.05,0> inverse }        
} // end upper inside

// ------------------------------------------------------------------ Glass off for Coupé 
#if (Coupe_ON__ = 1)
// clipped_by{ 
union{ 
   box{ <-4.50,0,-Half_Brutto-0.05>,<-1.80,0.80,Half_Brutto+0.05> translate<0,Car_Tween-0.01+D,0>}  
   box{ <-2,0,-Half_Brutto-0.05>,< 0,0.90,Half_Brutto+0.05>rotate<0,0,00> translate<-1.20,Car_Top-0.23+D,0>}  
   box{ <0,0,-Half_Brutto-0.05>,< 0.5,0.90,Half_Brutto+0.05>rotate<0,0,20> translate<-1.20,Car_Top-0.23+D,0>}  
// // inverse 
pigment{ color Clear} 
}// end union
// } // end clipped_by
#end // Coupé 
   
// texture{ pigment{ color rgb 0.6} }
       texture{ pigment{ rgbf <0.98, 0.98, 0.98, 0.9>*0.9 }
                finish { diffuse 0.1 reflection 0.2  
                         specular 0.8 roughness 0.0003 phong 1 phong_size 400}
               } // end of texture ------------------------------------------- 

} // end upper inside  
//------------------------------------------------------------------ end window glass  


//---------------------------------------------------------- add lights: 
union{
object { Round_Box(<-0.025,-0.08,-0.22>,
                   < 0.025, 0.08, 0.22>, 0.025, 0)  
         texture{ pigment{ rgbf <0.98, 0.98, 0.98, 0.9> }
                  finish { reflection 0.35  
                           specular 0.8 roughness 0.0003 
                           phong 1 phong_size 400}
                 } // end of texture -------------------------
         scale<1,1,1>  rotate<0, 0,0> 
         translate<+0.02,0,0>
       } // ---------------------------------------------------------
object { Round_Box(<-0.025,-0.09,-0.23>,
                   < 0.025, 0.09, 0.23>, 0.015, 0)  
         texture{ Polished_Chrome}
         scale<1,1,1>  rotate<0, 0,0> 
         translate<0,0,0>
       } // ---------------------------------------------------------
   translate<Front_X ,0.80,-Half_Brutto+0.35>} 

union{
object { Round_Box(<-0.025,-0.08,-0.22>,
                   < 0.025, 0.08, 0.22>, 0.025, 0)  
         texture{ pigment{ rgbf <0.98, 0.98, 0.98, 0.9> }
                  finish { reflection 0.35  
                           specular 0.8 roughness 0.0003 
                           phong 1 phong_size 400}
                 } // end of texture -------------------------
         scale<1,1,1>  rotate<0, 0,0> 
         translate<+0.02,0,0>
       } // ---------------------------------------------------------
object { Round_Box(<-0.025,-0.09,-0.23>,
                   < 0.025, 0.09, 0.23>, 0.015, 0)  
         texture{ Polished_Chrome}
         scale<1,1,1>  rotate<0, 0,0> 
         translate<0,0,0>
       } // ---------------------------------------------------------
   translate<Front_X ,0.80,-Half_Brutto+0.35> scale<1,1,-1>} 


union{  // blinker right front:
object { Round_Box(<-0.100,-0.08,-0.015>,
                   < 0.025, 0.08, 0.22>, 0.035, 0)  
         texture{ pigment{ rgbf <0.98, 0.75, 0.55, 0.9> }
                  finish { reflection 0.35  
                           specular 0.8 roughness 0.0003 
                           phong 1 phong_size 400}
                 } // end of texture -------------------------
         scale<1,0.5,1>  rotate<0, 0,0> 
         translate<+0.02,0,-0.02>
       } // ---------------------------------------------------------
 
object { Round_Box(<-0.125,-0.09,-0.00>,
                   < 0.025, 0.09, 0.23>, 0.035, 0)  
         texture{ Polished_Chrome}
         scale<1,0.5,1>  rotate<0, 0,0> 
         translate<0,0,0>
       } // ---------------------------------------------------------
  translate<Front_X ,0.60,-Half_Brutto+0.00>} 

union { // blinker left front
object { Round_Box(<-0.100,-0.08,-0.015>,
                   < 0.025, 0.08, 0.22>, 0.035, 0)  
         texture{ pigment{ rgbf <0.98, 0.75, 0.55, 0.9> }
                  finish { reflection 0.35  
                           specular 0.8 roughness 0.0003 
                           phong 1 phong_size 400}
                 } // end of texture -------------------------
         scale<1,0.5,1>  rotate<0, 0,0> 
         translate<+0.02,0,-0.02>
       } // ---------------------------------------------------------
 
object { Round_Box(<-0.125,-0.09,-0.00>,
                   < 0.025, 0.09, 0.23>, 0.035, 0)  
         texture{ Polished_Chrome}
         scale<1,0.5,1>  rotate<0, 0,0> 
         translate<0,0,0>
       } // ---------------------------------------------------------
   translate<Front_X ,0.60,-Half_Brutto+0.00> scale<1,1,-1>} 

//------------------------------------------------------------------ end of front lights
//------------------------------------------------------------------ backside lights

union { // blinker left backside
object { Round_Box(<-0.100,-0.08,-0.025>,
                   < 0.025, 0.08, 0.22>, 0.035, 0)  
         texture{ pigment{ rgbf <0.98, 0.75, 0.55, 0.9> }
                  finish { reflection 0.35  
                           specular 0.8 roughness 0.0003 
                           phong 1 phong_size 400}
                 } // end of texture -------------------------
         scale<1,0.5,1>  rotate<0, 0,0> 
         translate<+0.02,0,-0.02>
       } // ---------------------------------------------------------
 
object { Round_Box(<-0.125,-0.09,-0.00>,
                   < 0.025, 0.09, 0.23>, 0.035, 0)  
         texture{ Polished_Chrome}
         scale<1,0.5,1>  rotate<0, 0,0> 
         translate<0,0,0>
       } // ---------------------------------------------------------
  scale<-1,1,1> translate<-Car_Lenght+Front_X ,0.60,-Half_Brutto+0.04> } 

union { // blinker right backside
object { Round_Box(<-0.100,-0.08,-0.025>,
                   < 0.025, 0.08, 0.22>, 0.025, 0)  
         texture{ pigment{ rgbf <0.98, 0.75, 0.55, 0.9> }
                  finish { reflection 0.35  
                           specular 0.8 roughness 0.0003 
                           phong 1 phong_size 400}
                 } // end of texture -------------------------
         scale<1,0.5,1>  rotate<0, 0,0> 
         translate<+0.02,0,-0.02>
       } // ---------------------------------------------------------
 
object { Round_Box(<-0.125,-0.09,-0.015>,
                   < 0.025, 0.09, 0.23>, 0.035, 0)  
         texture{ Polished_Chrome}
         scale<1,0.5,1>  rotate<0, 0,0> 
         translate<0,0,0>
       } // ---------------------------------------------------------
  scale<-1,1,1> translate<-Car_Lenght+Front_X ,0.60,-Half_Brutto+0.04>  scale<1,1,-1>} 

union { // red light right backside
object { Round_Box(<-0.100,-0.08,-0.025>,
                   < 0.025, 0.08, 0.22>, 0.025, 0)  
         texture{ pigment{ rgbf <0.98, 0.15, 0.15, 0.9> }
                  finish { reflection 0.35  
                           specular 0.8 roughness 0.0003 
                           phong 1 phong_size 400}
                 } // end of texture -------------------------
         scale<1,0.75,1>  rotate<0, 0,0> 
         translate<+0.02,0,-0.02>
       } // ---------------------------------------------------------
 
object { Round_Box(<-0.125,-0.09,-0.015>,
                   < 0.025, 0.09, 0.23>, 0.035, 0)  
         texture{ Polished_Chrome}
         scale<1,0.75,1>  rotate<0, 0,0> 
         translate<0,0,0>
       } // ---------------------------------------------------------
  scale<-1,1,1> translate<-Car_Lenght+Front_X ,0.75,-Half_Brutto+0.04> } 
union { // red light left backside
object { Round_Box(<-0.100,-0.08,-0.025>,
                   < 0.025, 0.08, 0.22>, 0.035, 0)  
         texture{ pigment{ rgbf <0.98, 0.15, 0.15, 0.9> }
                  finish { reflection 0.35  
                           specular 0.8 roughness 0.0003 
                           phong 1 phong_size 400}
                 } // end of texture -------------------------
         scale<1,0.75,1>  rotate<0, 0,0> 
         translate<+0.02,0,-0.02>
       } // ---------------------------------------------------------
 
object { Round_Box(<-0.125,-0.09,-0.015>,
                   < 0.025, 0.09, 0.23>, 0.025, 0)  
         texture{ Polished_Chrome}
         scale<1,0.75,1>  rotate<0, 0,0> 
         translate<0,0,0>
       } // ---------------------------------------------------------
  scale<-1,1,1> translate<-Car_Lenght+Front_X ,0.75,-Half_Brutto+0.04> scale<1,1,-1> } 
                                                     

//------------------------------------------------------------------ end of backside lights

//--------------------------------------------------------------------------------  bumpers 
object { // Round_Box(A, B, WireRadius, UseMerge) // front
         Round_Box(<-0.02,0,-Half_Brutto-0.04>,<0.30,0.15, Half_Brutto+0.04>, 0.030   , 0)  
         clipped_by{ box{ <0,-D,-Half_Brutto-0.06>,<0.33,0.16, Half_Brutto+0.05> } } 
          texture{ Bumper_Metal_Texture }
         scale<1,1,1>  rotate<0, 0,0> translate<Front_X-0.20,Car_Low+0.05,0>
       } // ---------------------------------------------------------

object { // Round_Box(A, B, WireRadius, UseMerge) // back
         Round_Box(<-0.02,0,-Half_Brutto-0.04>,<0.30,0.15, Half_Brutto+0.04>, 0.030   , 0)  
         clipped_by{ box{ <0,-D,-Half_Brutto-0.06>,<0.33,0.16, Half_Brutto+0.05> } } 
          texture{ Bumper_Metal_Texture }
         scale<1,1,1>  rotate<0, 0,0> translate<-Car_Lenght+Front_X-0.10,Car_Low+0.05,0>
       } // ---------------------------------------------------------



//------------------------------------------------------------------ end of backside lights
#if (Coupe_ON__ = 1) // back cover
object { // Round_Box(A, B, WireRadius, UseMerge)
         Round_Box(<-0.50,-0.06,-Half_Brutto+0.10>,<0,0.06,Half_Brutto-0.10>, 0.05   , 0)  
         
         texture{ Car_Top_Texture__}
                  
         scale<1,1,1>  rotate<0,0,-5> translate<-2.57,Car_Tween-0.05,0>
       } // ---------------------------------------------------------
#end 
// front console 
object { // Round_Box(A, B, WireRadius, UseMerge)
         Round_Box(<-0.40,-0.06,-Half_Brutto+0.10>,<0,0.06,Half_Brutto-0.10>, 0.05   , 0)  
         
         texture{ pigment{ color rgb<1,1,1>*0.75}} 
                  
         scale<1,1,1>  rotate<0,0,15> translate<-0.70,Car_Tween-0.05,0>
       } // ---------------------------------------------------------
// steering wheel 
union{ 
 torus{ 0.20,0.015 } 
 cylinder{ <0,0,0>,<0.20,0,0>,0.0075 rotate<0,0*120,0>} 
 cylinder{ <0,0,0>,<0.20,0,0>,0.0075 rotate<0,1*120,0>} 
 cylinder{ <0,0,0>,<0.20,0,0>,0.0075 rotate<0,2*120,0>} 
 cylinder{ <0,0,0>,<0,-0.80,0>,0.025 }
 sphere{ <0,0,0>,0.045 scale<1,0.5,1> translate<0,0.01,0>}
 texture{ Chassis_Inside_Texture__}
 rotate<0,0,45>
 translate<-1.08,0.90,Half_Brutto/2>} 

//------------------------------------------------------------------ Seats
 object{ Seat scale<1,1,1> rotate<0, 0,5> translate<-1.50,Car_Low+0.25,Half_Brutto/2> } 
 object{ Seat scale<1,1,1> rotate<0, 0,5> translate<-1.50,Car_Low+0.25,Half_Brutto/2> scale<1,1,-1> } 

 object{ Seat scale<1,1,0.8> rotate<0, 0,7> translate<-2.40,Car_Low+0.25,-Half_Brutto*0.6> } 
 object{ Seat scale<1,1,0.8> rotate<0, 0,7> translate<-2.40,Car_Low+0.25,0> } 
 object{ Seat scale<1,1,0.8> rotate<0, 0,7> translate<-2.40,Car_Low+0.25, Half_Brutto*0.6> } 

//------------------------------------------------------------------ end of Seats


texture{ Car_Body_Texture__ } 
}// end of union
// --------------------------------------------------------------------------------------
#end// of macro ------------------------------------------------------// end of macro  "Car_01"




//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------
// sample: 
/*
//----------------------------------------------------------------------------- textures
#declare Xxx_Texture = 
   texture { //Chrome_Metal}  
             pigment{ color rgb<1,1,1> } 
             finish { phong 0.1} 
           } // end of texture
//---------------------------------------------------------------------------------------
#include "Car_01.inc" 
//-------------------------------------------------------------------------------------// 
object{ Car_01(     , // sample of VAL
 
                   ) //----------------------------------------------------------------//
        texture { Xyz_Texture }
        scale <1,1,1>*1
        rotate<0,0,0> 
        translate<0.00,0.00, 0.00>}
//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------
*/


#version Car_01_Inc_Temp;
#end
//------------------------------------- end of include file

