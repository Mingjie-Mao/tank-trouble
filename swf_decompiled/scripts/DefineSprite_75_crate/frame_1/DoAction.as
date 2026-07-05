function hitCheck(mc, point)
{
   localToGlobal(point);
   if(mc.hitTest(point.x,point.y,false))
   {
      return true;
   }
   return false;
}
onEnterFrame = function()
{
   if(_root.frozen)
   {
      return undefined;
   }
   var _loc3_ = 0;
   while(_loc3_ < _root.TANKS)
   {
      if(_root.game["tank" + _loc3_].alive && (hitCheck(_root.game["tank" + _loc3_],{x:-10,y:-10}) || hitCheck(_root.game["tank" + _loc3_],{x:-10,y:10}) || hitCheck(_root.game["tank" + _loc3_],{x:10,y:10}) || hitCheck(_root.game["tank" + _loc3_],{x:10,y:-10})))
      {
         if(_root.game["tank" + _loc3_].currentWeapon == "bullet")
         {
            if(_root.soundOn)
            {
               _root.soundClick.start();
            }
            _root.setWeapon(_root.game["tank" + _loc3_],weapon);
            _root.numberOfCrates = _root.numberOfCrates - 1;
            _root.reachable[pos].used = false;
            this.removeMovieClip();
         }
      }
      _loc3_ = _loc3_ + 1;
   }
   _xscale = _xscale + scaleSpeed;
   _yscale = _yscale + scaleSpeed;
   if(_xscale > targetScale)
   {
      scaleSpeed -= scaleSpeedDiff;
   }
   if(_xscale - targetScale < 0 && scaleSpeed < 0 && !landed)
   {
      if(_root.soundOn)
      {
         _root.soundCrateLand.start();
      }
      rotSpeed = 0;
      scaleSpeed = 0;
      _xscale = targetScale;
      _yscale = targetScale;
      landed = true;
      var _loc4_ = 0;
      while(_loc4_ < _root.NUMBEROFDUSTCLOUDS)
      {
         _root.game.mazebg.createEmptyMovieClip("dust" + _root.numberOfCrates + "-" + _loc4_,_root.game.mazebg.getNextHighestDepth());
         s = _root.game.mazebg["dust" + _root.numberOfCrates + "-" + _loc4_];
         this.swapDepths(s);
         s.lineStyle(10 * (_root.SCALE / 50),11184810,40 + random(20));
         s.moveTo(0,0);
         s.lineTo(0,1);
         s.xspeed = (Math.random() * 2 - 1) * (_root.SCALE / 50);
         s.yspeed = (Math.random() * 2 - 1) * (_root.SCALE / 50);
         s.x = _X + s.xspeed * (Math.random() * 3 + 1) + (Math.random() * 2 - 1) * (_root.SCALE / 50);
         s.y = _Y + s.yspeed * (Math.random() * 3 + 1) + (Math.random() * 2 - 1) * (_root.SCALE / 50);
         s._x = s.x;
         s._y = s.y;
         s.onEnterFrame = function()
         {
            if(_root.frozen)
            {
               return undefined;
            }
            this._xscale += 2;
            this._yscale += 2;
            this._alpha -= 4 - Math.random() * 2;
            this.xspeed *= 0.85;
            this.yspeed *= 0.85;
            this.x += this.xspeed;
            this.y += this.yspeed;
            this._x = this.x;
            this._y = this.y;
            if(this._alpha <= 0)
            {
               this.removeMovieClip();
            }
         };
         _loc4_ = _loc4_ + 1;
      }
   }
};
