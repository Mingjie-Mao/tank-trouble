function init(dim)
{
   inited = true;
   dens = new Array(dim.x);
   old_dens = new Array(dim.x);
   flags = new Array(dim.x);
   var _loc1_ = 0;
   while(_loc1_ < dim.x)
   {
      dens[_loc1_] = new Array(dim.y);
      old_dens[_loc1_] = new Array(dim.y);
      flags[_loc1_] = new Array(dim.y);
      _loc1_ = _loc1_ + 1;
   }
   center = {x:dim.x / 2,y:dim.y / 2};
   radiusSq = (dim.x / 2 - 2) * (dim.x / 2 - 2);
   var _loc2_ = 0;
   while(_loc2_ < dens[0].length)
   {
      _loc1_ = 0;
      while(_loc1_ < dens.length)
      {
         if((_loc1_ - center.x) * (_loc1_ - center.x) + (_loc2_ - center.y) * (_loc2_ - center.y) <= radiusSq)
         {
            flags[_loc1_][_loc2_] = FLUID;
         }
         else
         {
            flags[_loc1_][_loc2_] = BOUNDARY;
         }
         dens[_loc1_][_loc2_] = 0;
         old_dens[_loc1_][_loc2_] = 0;
         _loc1_ = _loc1_ + 1;
      }
      _loc2_ = _loc2_ + 1;
   }
}
function diffuse(inn, out)
{
   var _loc5_ = dt * viscosity;
   var _loc6_ = 0;
   while(_loc6_ < 2)
   {
      var _loc3_ = 1;
      while(_loc3_ < inn[0].length - 1)
      {
         var _loc2_ = 1;
         while(_loc2_ < inn.length - 1)
         {
            if(flags[_loc2_][_loc3_] == FLUID)
            {
               out[_loc2_][_loc3_] = (inn[_loc2_][_loc3_] + _loc5_ * (out[_loc2_ + 1][_loc3_] + out[_loc2_ - 1][_loc3_] + out[_loc2_][_loc3_ + 1] + out[_loc2_][_loc3_ - 1])) / (1 + 4 * _loc5_);
            }
            _loc2_ = _loc2_ + 1;
         }
         _loc3_ = _loc3_ + 1;
      }
      _loc3_ = 1;
      while(_loc3_ < inn[0].length - 1)
      {
         _loc2_ = 1;
         while(_loc2_ < inn.length - 1)
         {
            if(flags[_loc2_][_loc3_] == BOUNDARY)
            {
               out[_loc2_][_loc3_] = getBoundary(out,_loc2_,_loc3_);
            }
            _loc2_ = _loc2_ + 1;
         }
         _loc3_ = _loc3_ + 1;
      }
      _loc6_ = _loc6_ + 1;
   }
}
function getBoundary(inn, x, y)
{
   var _loc1_ = 0;
   var _loc4_ = 0;
   if(flags[x - 1][y] == FLUID)
   {
      _loc1_ = _loc1_ + 1;
      _loc4_ += inn[x - 1][y];
   }
   if(flags[x + 1][y] == FLUID)
   {
      _loc1_ = _loc1_ + 1;
      _loc4_ += inn[x + 1][y];
   }
   if(flags[x][y - 1] == FLUID)
   {
      _loc1_ = _loc1_ + 1;
      _loc4_ += inn[x][y - 1];
   }
   if(flags[x][y + 1] == FLUID)
   {
      _loc1_ = _loc1_ + 1;
      _loc4_ += inn[x][y + 1];
   }
   if(_loc1_ == 0)
   {
      return 0;
   }
   return _loc4_ / _loc1_;
}
function display()
{
   clear();
   var _loc3_ = 0;
   while(_loc3_ < dens[0].length)
   {
      var _loc2_ = 0;
      while(_loc2_ < dens.length)
      {
         if(flags[_loc2_][_loc3_] == FLUID)
         {
            lineStyle(4 * (_root.SCALE / 50),16711680,10 * dens[_loc2_][_loc3_]);
            moveTo((_loc2_ - center.x) * 4 * (_root.SCALE / 50),(_loc3_ - center.y) * 4 * (_root.SCALE / 50));
            lineTo((_loc2_ - center.x) * 4 * (_root.SCALE / 50) + 1,(_loc3_ - center.y) * 4 * (_root.SCALE / 50));
         }
         _loc2_ = _loc2_ + 1;
      }
      _loc3_ = _loc3_ + 1;
   }
}
var dens;
var old_dens;
var flags;
var FLUID = 0;
var BOUNDARY = 1;
var dt = 1;
var viscosity = 0.1;
var center;
var radiusSq;
var inited = false;
onEnterFrame = function()
{
   var _loc1_ = old_dens;
   old_dens = dens;
   dens = _loc1_;
   diffuse(old_dens,dens);
   display();
};
