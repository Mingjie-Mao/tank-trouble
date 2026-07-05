function init()
{
   inited = true;
   center = {x:owner.x,y:owner.y};
   var _loc2_ = {x:0,y:0};
   hexagons = new Array(SIZE);
   var _loc3_ = 0;
   while(_loc3_ < SIZE)
   {
      hexagons[_loc3_] = new Array(SIZE);
      _loc3_ = _loc3_ + 1;
   }
   _loc3_ = 0;
   while(_loc3_ < SIZE)
   {
      var _loc1_ = 0;
      while(_loc1_ < SIZE)
      {
         _loc2_.x = center.x - (SIZE - 1) / 2 * HEXSIZE + _loc3_ * HEXSIZE + (_loc1_ % 2 != 1 ? (- HEXSIZE) / 4 : HEXSIZE / 4);
         _loc2_.y = center.y - (SIZE - 1) / 2 * (HEXSIZE * 0.85) + _loc1_ * (HEXSIZE * 0.85);
         hexagons[_loc3_][_loc1_] = {x:_loc2_.x,y:_loc2_.y,size:0,shake:0,impactNum:0};
         _loc1_ = _loc1_ + 1;
      }
      _loc3_ = _loc3_ + 1;
   }
}
function impact(x, y, impactNum)
{
   var _loc7_ = 10000;
   var _loc11_ = undefined;
   var _loc6_ = 0;
   while(_loc6_ < SIZE)
   {
      var _loc4_ = 0;
      while(_loc4_ < SIZE)
      {
         var _loc3_ = hexagons[_loc6_][_loc4_];
         var _loc5_ = (_loc3_.x - x) * (_loc3_.x - x) + (_loc3_.y - y) * (_loc3_.y - y);
         if(_loc5_ < _loc7_)
         {
            _loc7_ = _loc5_;
            _loc11_ = _loc3_;
         }
         _loc4_ = _loc4_ + 1;
      }
      _loc6_ = _loc6_ + 1;
   }
   _loc11_.shake = 5;
   _loc11_.impactNum = impactNum;
   var _loc10_ = 0;
   while(_loc10_ < 50)
   {
      p = _root.game.createEmptyMovieClip("particle-" + _root.game.getNextHighestDepth(),_root.game.getNextHighestDepth());
      var _loc9_ = Math.random() * 360;
      var _loc8_ = 3 * (0.5 + 2 * Math.random()) * (_root.SCALE / 50);
      p.lineStyle((Math.random() * 2 + 1) * (_root.SCALE / 50),shieldColor);
      p.moveTo(0,0);
      p.lineTo(1,0);
      p.xspeed = Math.cos(_loc9_) * _loc8_;
      p.yspeed = Math.sin(_loc9_) * _loc8_;
      p.x = x + p.xspeed;
      p.y = y + p.yspeed;
      p._x = p.x;
      p._y = p.y;
      p.lifetime = 12;
      p.alpha = 20;
      p.onEnterFrame = function()
      {
         if(_root.frozen)
         {
            return undefined;
         }
         this.x += this.xspeed;
         this.y += this.yspeed;
         this._x = this.x;
         this._y = this.y;
         this._alpha = this.alpha;
         this.xspeed *= 0.9000000000000002;
         this.yspeed *= 0.9000000000000002;
         this.lifetime = this.lifetime - 1;
         if(this.lifetime <= 0)
         {
            this.alpha -= 2;
         }
         if(this.alpha <= 0)
         {
            this.removeMovieClip();
         }
      };
      _loc10_ = _loc10_ + 1;
   }
}
var hexagons;
var firstRowEven = true;
var inited = false;
var center;
var shieldColor = shield.shieldColor;
var SIZE = 8;
var HEXSIZE = 20 * (_root.SCALE / 50);
onEnterFrame = function()
{
   if(_root.frozen)
   {
      return undefined;
   }
   if(!inited)
   {
      return undefined;
   }
   var _loc8_ = 0;
   while(_loc8_ < SIZE)
   {
      var _loc5_ = 0;
      while(_loc5_ < SIZE)
      {
         var _loc2_ = hexagons[_loc8_][_loc5_];
         _root.game.mazebg.localToGlobal(_loc2_);
         if(shield.hitTest(_loc2_.x,_loc2_.y,true))
         {
            _root.game.globalToLocal(_loc2_);
            _loc2_.size = Math.min(_loc2_.size + 0.2 * HEXSIZE,HEXSIZE);
         }
         else
         {
            _root.game.globalToLocal(_loc2_);
            _loc2_.size = Math.max(_loc2_.size - 0.1 * HEXSIZE,0);
         }
         _loc2_.shake = Math.max(_loc2_.shake - 1,0);
         if(_loc2_.shake > 0 && _loc2_.shake < 3)
         {
            if(hexagons[_loc8_ - 1][_loc5_].impactNum < _loc2_.impactNum)
            {
               hexagons[_loc8_ - 1][_loc5_].shake = 5;
               hexagons[_loc8_ - 1][_loc5_].impactNum = _loc2_.impactNum;
            }
            if(hexagons[_loc8_ + 1][_loc5_].impactNum < _loc2_.impactNum)
            {
               hexagons[_loc8_ + 1][_loc5_].shake = 5;
               hexagons[_loc8_ + 1][_loc5_].impactNum = _loc2_.impactNum;
            }
            var _loc3_ = _loc5_ % 2 != (!firstRowEven ? 0 : 1) ? -1 : 0;
            if(hexagons[_loc8_ + _loc3_][_loc5_ - 1].impactNum < _loc2_.impactNum)
            {
               hexagons[_loc8_ + _loc3_][_loc5_ - 1].shake = 5;
               hexagons[_loc8_ + _loc3_][_loc5_ - 1].impactNum = _loc2_.impactNum;
            }
            if(hexagons[_loc8_ + 1 + _loc3_][_loc5_ - 1].impactNum < _loc2_.impactNum)
            {
               hexagons[_loc8_ + 1 + _loc3_][_loc5_ - 1].shake = 5;
               hexagons[_loc8_ + 1 + _loc3_][_loc5_ - 1].impactNum = _loc2_.impactNum;
            }
            if(hexagons[_loc8_ + _loc3_][_loc5_ + 1].impactNum < _loc2_.impactNum)
            {
               hexagons[_loc8_ + _loc3_][_loc5_ + 1].shake = 5;
               hexagons[_loc8_ + _loc3_][_loc5_ + 1].impactNum = _loc2_.impactNum;
            }
            if(hexagons[_loc8_ + 1 + _loc3_][_loc5_ + 1].impactNum < _loc2_.impactNum)
            {
               hexagons[_loc8_ + 1 + _loc3_][_loc5_ + 1].shake = 5;
               hexagons[_loc8_ + 1 + _loc3_][_loc5_ + 1].impactNum = _loc2_.impactNum;
            }
         }
         _loc5_ = _loc5_ + 1;
      }
      _loc8_ = _loc8_ + 1;
   }
   if(owner.x - center.x >= HEXSIZE)
   {
      var _loc12_ = hexagons.shift();
      hexagons[SIZE - 1] = _loc12_;
      _loc5_ = 0;
      while(_loc5_ < SIZE)
      {
         hexagons[SIZE - 1][_loc5_].x = hexagons[SIZE - 2][_loc5_].x + HEXSIZE;
         hexagons[SIZE - 1][_loc5_].y = hexagons[SIZE - 2][_loc5_].y;
         hexagons[SIZE - 1][_loc5_].size = 0;
         hexagons[SIZE - 1][_loc5_].shake = 0;
         _loc5_ = _loc5_ + 1;
      }
      center.x += HEXSIZE;
   }
   else if(owner.x - center.x <= - HEXSIZE)
   {
      var _loc11_ = hexagons.pop();
      hexagons.unshift(_loc11_);
      _loc5_ = 0;
      while(_loc5_ < SIZE)
      {
         hexagons[0][_loc5_].x = hexagons[1][_loc5_].x - HEXSIZE;
         hexagons[0][_loc5_].y = hexagons[1][_loc5_].y;
         hexagons[0][_loc5_].size = 0;
         hexagons[0][_loc5_].shake = 0;
         _loc5_ = _loc5_ + 1;
      }
      center.x -= HEXSIZE;
   }
   if(owner.y - center.y >= HEXSIZE * 0.85)
   {
      _loc8_ = 0;
      while(_loc8_ < SIZE)
      {
         var _loc10_ = hexagons[_loc8_].shift();
         hexagons[_loc8_][SIZE - 1] = _loc10_;
         hexagons[_loc8_][SIZE - 1].x = hexagons[_loc8_][SIZE - 2].x + (SIZE % 2 != (!firstRowEven ? 0 : 1) ? (- HEXSIZE) / 2 : HEXSIZE / 2);
         hexagons[_loc8_][SIZE - 1].y = hexagons[_loc8_][SIZE - 2].y + HEXSIZE * 0.85;
         hexagons[_loc8_][SIZE - 1].size = 0;
         hexagons[_loc8_][SIZE - 1].shake = 0;
         _loc8_ = _loc8_ + 1;
      }
      center.y += HEXSIZE * 0.85;
      firstRowEven = !firstRowEven;
   }
   else if(owner.y - center.y <= - HEXSIZE * 0.85)
   {
      _loc8_ = 0;
      while(_loc8_ < SIZE)
      {
         var _loc9_ = hexagons[_loc8_].pop();
         hexagons[_loc8_].unshift(_loc9_);
         hexagons[_loc8_][0].x = hexagons[_loc8_][1].x + (!firstRowEven ? (- HEXSIZE) / 2 : HEXSIZE / 2);
         hexagons[_loc8_][0].y = hexagons[_loc8_][1].y - HEXSIZE * 0.85;
         hexagons[_loc8_][0].size = 0;
         hexagons[_loc8_][0].shake = 0;
         _loc8_ = _loc8_ + 1;
      }
      center.y -= HEXSIZE * 0.85;
      firstRowEven = !firstRowEven;
   }
   clear();
   _loc8_ = 0;
   while(_loc8_ < SIZE)
   {
      _loc5_ = 0;
      while(_loc5_ < SIZE)
      {
         _loc2_ = hexagons[_loc8_][_loc5_];
         lineStyle(1,5592405,50);
         beginFill(shieldColor,50);
         var _loc7_ = Math.random() * (_root.SCALE / 50) - _root.SCALE / 50 / 2;
         var _loc6_ = Math.random() * (_root.SCALE / 50) - _root.SCALE / 50 / 2;
         moveTo(_loc2_.x + _loc2_.shake * _loc7_,_loc2_.y + _loc2_.shake * _loc6_ + _loc2_.size / 2);
         var _loc4_ = 0;
         while(_loc4_ < 7)
         {
            lineTo(_loc2_.x + _loc2_.shake * _loc7_ + Math.sin(_loc4_ / 6 * 2 * 3.141592653589793) * _loc2_.size / 2,_loc2_.y + _loc2_.shake * _loc6_ + Math.cos(_loc4_ / 6 * 2 * 3.141592653589793) * _loc2_.size / 2);
            _loc4_ = _loc4_ + 1;
         }
         _loc5_ = _loc5_ + 1;
      }
      _loc8_ = _loc8_ + 1;
   }
};
