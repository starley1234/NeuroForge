// Кронштейн полки — пример для nexus optimize
width = 40;          // ширина
height = 45;         // высота стойки
thickness = 6.0;     // толщина стенки  <- силовой параметр
rib = 8.0;           // ребро жёсткости <- силовой параметр
leg = 35;            // вылет полки
hole_d = 5;          // отверстие крепления

difference() {
  union() {
    cube([width, thickness, height]);
    cube([width, leg, thickness]);
    translate([width/2 - rib/2, 0, 0]) cube([rib, leg, leg]);
  }
  translate([width/2, thickness/2, height*0.72]) rotate([90,0,0])
    translate([0,0,-thickness]) cylinder(h=thickness*3, r=hole_d/2, $fn=24);
}
