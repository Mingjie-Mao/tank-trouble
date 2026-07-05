function init(dim)
{
   inited = true;
   distrib = new Array(dim.x);
   old_distrib = new Array(dim.x);
   flags = new Array(dim.x);
   vel = new Array(dim.x);
   var _loc3_ = 0;
   while(_loc3_ < dim.x)
   {
      distrib[_loc3_] = new Array(dim.y);
      old_distrib[_loc3_] = new Array(dim.y);
      flags[_loc3_] = new Array(dim.y);
      vel[_loc3_] = new Array(dim.y);
      _loc3_ = _loc3_ + 1;
   }
   center = {x:dim.x / 2,y:dim.y / 2};
   radiusSq = (dim.x / 2 - 2) * (dim.x / 2 - 2);
   var _loc2_ = 0;
   while(_loc2_ < distrib[0].length)
   {
      _loc3_ = 0;
      while(_loc3_ < distrib.length)
      {
         if((_loc3_ - center.x) * (_loc3_ - center.x) + (_loc2_ - center.y) * (_loc2_ - center.y) <= radiusSq)
         {
            flags[_loc3_][_loc2_] = FLUID;
            particles[particles.length] = {x:_loc3_ - center.x,y:_loc2_ - center.y};
         }
         else
         {
            flags[_loc3_][_loc2_] = BOUNDARY;
         }
         distrib[_loc3_][_loc2_] = new Array(Q);
         old_distrib[_loc3_][_loc2_] = new Array(Q);
         var _loc1_ = 0;
         while(_loc1_ < Q)
         {
            distrib[_loc3_][_loc2_][_loc1_] = w[_loc1_];
            old_distrib[_loc3_][_loc2_][_loc1_] = w[_loc1_];
            _loc1_ = _loc1_ + 1;
         }
         _loc3_ = _loc3_ + 1;
      }
      _loc2_ = _loc2_ + 1;
   }
}
function stream(inn, out)
{
   var _loc3_ = 1;
   while(_loc3_ < inn[0].length - 1)
   {
      var _loc4_ = 1;
      while(_loc4_ < inn.length - 1)
      {
         if(flags[_loc4_][_loc3_] == FLUID)
         {
            var _loc1_ = 0;
            while(_loc1_ < Q)
            {
               var _loc2_ = {x:_loc4_ - e[_loc1_].x,y:_loc3_ - e[_loc1_].y};
               if(flags[_loc2_.x][_loc2_.y] == BOUNDARY)
               {
                  out[_loc4_][_loc3_][_loc1_] = inn[_loc4_][_loc3_][e_opp[_loc1_]];
               }
               else
               {
                  out[_loc4_][_loc3_][_loc1_] = inn[_loc2_.x][_loc2_.y][_loc1_];
               }
               _loc1_ = _loc1_ + 1;
            }
         }
         _loc4_ = _loc4_ + 1;
      }
      _loc3_ = _loc3_ + 1;
   }
}
function collision(inn)
{
   var _loc8_ = 1;
   while(_loc8_ < inn[0].length - 1)
   {
      var _loc9_ = 1;
      while(_loc9_ < inn.length - 1)
      {
         if(flags[_loc9_][_loc8_] == FLUID)
         {
            var _loc5_ = 0;
            var _loc1_ = 0;
            while(_loc1_ < Q)
            {
               _loc5_ += inn[_loc9_][_loc8_][_loc1_];
               _loc1_ = _loc1_ + 1;
            }
            var _loc2_ = {x:0,y:0};
            _loc1_ = 1;
            while(_loc1_ < Q)
            {
               _loc2_.x += inn[_loc9_][_loc8_][_loc1_] * e[_loc1_].x;
               _loc2_.y += inn[_loc9_][_loc8_][_loc1_] * e[_loc1_].y;
               _loc1_ = _loc1_ + 1;
            }
            if(_loc9_ == 5 && _loc8_ > 5)
            {
               _loc2_.x += Math.random() * 0.1 * tau;
            }
            else if(_loc9_ == 5)
            {
               _loc2_.x -= Math.random() * 0.1 * tau;
            }
            vel[_loc9_][_loc8_] = _loc2_;
            var _loc7_ = _loc5_ - 1.5 * (_loc2_.x * _loc2_.x + _loc2_.y * _loc2_.y);
            _loc1_ = 0;
            while(_loc1_ < Q)
            {
               var _loc4_ = _loc2_.x * e[_loc1_].x + _loc2_.y * e[_loc1_].y;
               var _loc6_ = w[_loc1_] * (_loc7_ + 3 * _loc4_ + 4.5 * _loc4_ * _loc4_);
               inn[_loc9_][_loc8_][_loc1_] -= 1 / tau * (inn[_loc9_][_loc8_][_loc1_] - _loc6_);
               _loc1_ = _loc1_ + 1;
            }
         }
         else
         {
            vel[_loc9_][_loc8_] = {x:0,y:0};
         }
         _loc9_ = _loc9_ + 1;
      }
      _loc8_ = _loc8_ + 1;
   }
}
function advect()
{
   var _loc3_ = 0;
   while(_loc3_ < particles.length)
   {
      var _loc1_ = {x:Math.round(particles[_loc3_].x + center.x),y:Math.round(particles[_loc3_].y + center.y)};
      var _loc2_ = {x:particles[_loc3_].x + center.x - _loc1_.x,y:particles[_loc3_].y + center.y - _loc1_.y};
      particles[_loc3_].x += vel[_loc1_.x][_loc1_.y].x * (1 - _loc2_.x) * (1 - _loc2_.y) + vel[_loc1_.x + 1][_loc1_.y].x * _loc2_.x * (1 - _loc2_.y) + vel[_loc1_.x][_loc1_.y + 1].x * (1 - _loc2_.x) * _loc2_.y + vel[_loc1_.x + 1][_loc1_.y + 1].x * _loc2_.x * _loc2_.y;
      particles[_loc3_].y += vel[_loc1_.x][_loc1_.y].y * (1 - _loc2_.x) * (1 - _loc2_.y) + vel[_loc1_.x + 1][_loc1_.y].y * _loc2_.x * (1 - _loc2_.y) + vel[_loc1_.x][_loc1_.y + 1].y * (1 - _loc2_.x) * _loc2_.y + vel[_loc1_.x + 1][_loc1_.y + 1].y * _loc2_.x * _loc2_.y;
      if(particles[_loc3_].x * particles[_loc3_].x + particles[_loc3_].y * particles[_loc3_].y >= radiusSq - 4)
      {
         particles[_loc3_].x = 0;
         particles[_loc3_].y = 0;
      }
      _loc3_ = _loc3_ + 1;
   }
}
function display()
{
   clear();
   var _loc2_ = 0;
   while(_loc2_ < particles.length)
   {
      lineStyle(20 * (_root.SCALE / 50),16777215,30);
      moveTo(particles[_loc2_].x * 8 * (_root.SCALE / 50),particles[_loc2_].y * 8 * (_root.SCALE / 50));
      lineTo(particles[_loc2_].x * 8 * (_root.SCALE / 50) + 1,particles[_loc2_].y * 8 * (_root.SCALE / 50));
      _loc2_ = _loc2_ + 1;
   }
}
var distrib;
var old_distrib;
var flags;
var vel;
var e = new Array({x:0,y:0},{x:1,y:0},{x:1,y:1},{x:0,y:1},{x:-1,y:1},{x:-1,y:0},{x:-1,y:-1},{x:0,y:-1},{x:1,y:-1});
var e_opp = new Array(0,5,6,7,8,1,2,3,4);
var w = new Array(0.4444444444444444,0.1111111111111111,0.027777777777777776,0.1111111111111111,0.027777777777777776,0.1111111111111111,0.027777777777777776,0.1111111111111111,0.027777777777777776);
var Q = 9;
var FLUID = 0;
var BOUNDARY = 1;
var dt = 10;
var viscosity = 0.02;
var omega = 2 / (6 * viscosity + 1);
var tau = 1 / omega;
var center;
var radiusSq;
var particles = new Array();
var inited = false;
onEnterFrame = function()
{
   var _loc1_ = old_distrib;
   old_distrib = distrib;
   distrib = _loc1_;
   stream(old_distrib,distrib);
   collision(distrib);
   advect();
   display();
};
