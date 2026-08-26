// Фланец вал-корпус с шестью болтами
outer_d = 70; bore_d = 20; thickness = 6; hub_d = 34; hub_h = 12;
bolt_circle = 54; hole_d = 5.5; n = 6;

difference() {
  union() {
    cylinder(h=thickness, r=outer_d/2);
    translate([0,0,thickness]) cylinder(h=hub_h, r=hub_d/2);
  }
  translate([0,0,-1]) cylinder(h=thickness+hub_h+2, r=bore_d/2);
  for (i = [0:5])
    rotate([0,0,i*60]) translate([bolt_circle/2,0,0])
      cylinder(h=thickness*3, r=hole_d/2, center=true);
}
