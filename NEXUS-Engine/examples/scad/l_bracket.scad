// Базовый L-кронштейн (стартовая деталь для маховика данных)
width = 40; height = 45; thickness = 4; hole_d = 5; rib = 6; leg = 35;

difference() {
  union() {
    cube([width, thickness, height], center=false);
    cube([width, leg, thickness], center=false);
    translate([width/2 - rib/2, 0, 0]) cube([rib, leg, leg], center=false);
  }
  translate([width/2, thickness/2, height*0.72]) rotate([90, 0, 0])
    translate([0, 0, -thickness]) cylinder(h=thickness*3, r=hole_d/2);
  translate([width/2, leg*0.65, thickness/2]) cylinder(h=thickness*3, r=hole_d/2, center=true);
}
