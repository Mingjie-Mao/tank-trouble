function hitCheck(mc, point)
{
   localToGlobal(point);
   if(mc.hitTest(point.x,point.y,true))
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
   i = 0;
   while(i < _root.GATLINGHITCHECKINTERVALS)
   {
      previousX = x;
      previousY = y;
      x += xSpeed;
      y += ySpeed;
      _X = x;
      _Y = y;
      if(hitCheck(_root.game.mazemc,{x:0,y:0}))
      {
         if(_root.soundOn)
         {
            _root["soundBounce" + random(2)].start();
         }
         x = previousX;
         y = previousY;
         x -= xSpeed;
         y += ySpeed;
         _X = x;
         _Y = y;
         if(hitCheck(_root.game.mazemc,{x:0,y:0}))
         {
            hitOnXInvert = true;
         }
         else
         {
            hitOnXInvert = false;
         }
         x = previousX;
         y = previousY;
         x += xSpeed;
         y -= ySpeed;
         _X = x;
         _Y = y;
         if(hitCheck(_root.game.mazemc,{x:0,y:0}))
         {
            hitOnYInvert = true;
         }
         else
         {
            hitOnYInvert = false;
         }
         if(hitOnXInvert && !hitOnYInvert)
         {
            ySpeed = - ySpeed;
         }
         else if(hitOnYInvert && !hitOnXInvert)
         {
            xSpeed = - xSpeed;
         }
         else
         {
            xSpeed = - xSpeed;
            ySpeed = - ySpeed;
         }
         x = previousX;
         y = previousY;
         x += xSpeed;
         y += ySpeed;
      }
      i++;
   }
   _X = x;
   _Y = y;
   if(deadly == 0)
   {
      var i = 0;
      while(i < _root.TANKS)
      {
         if(_root.game["tank" + i].alive && hitCheck(_root.game["tank" + i],{x:0,y:0}))
         {
            _root.registerHit(owner,_root.game["tank" + i]);
            _root.destroyTank(i);
            this.removeMovieClip();
         }
         i++;
      }
   }
   if(deadly > 0)
   {
      deadly--;
   }
   lifetime--;
   if(lifetime <= 0)
   {
      var _loc3_ = 0;
      while(_loc3_ < _root.NUMBEROFSMOKECLOUDS)
      {
         s = _root.game.createEmptyMovieClip("smokegatlingbullet" + _root.game.getNextHighestDepth(),_root.game.getNextHighestDepth());
         s.lineStyle(2 * (_root.SCALE / 50),Math.round(random(4)) * 1118481,10 + random(20));
         s.moveTo(0,0);
         s.lineTo(0,1);
         s.xspeed = xSpeed * _root.GATLINGHITCHECKINTERVALS + 0.25 * (Math.random() * 8 - 4) * (_root.SCALE / 50);
         s.yspeed = ySpeed * _root.GATLINGHITCHECKINTERVALS + 0.25 * (Math.random() * 8 - 4) * (_root.SCALE / 50);
         s.x = _X;
         s.y = _Y;
         s._x = s.x;
         s._y = s.y;
         s.hitCheck = function(mc, point)
         {
            this.localToGlobal(point);
            if(mc.hitTest(point.x,point.y,true))
            {
               return true;
            }
            return false;
         };
         s.onEnterFrame = function()
         {
            if(_root.frozen)
            {
               return undefined;
            }
            this._xscale += 2;
            this._yscale += 2;
            this._alpha -= 15 - Math.random() * 2;
            this.xspeed *= 0.93;
            this.yspeed *= 0.93;
            this.x += this.xspeed;
            this.y += this.yspeed;
            this._x = this.x;
            this._y = this.y;
            if(this.hitCheck(_root.game.mazemc,{x:0,y:0}))
            {
               this.xspeed *= 0.25;
               this.yspeed *= 0.25;
            }
            if(this._alpha <= 0)
            {
               this.removeMovieClip();
            }
         };
         _loc3_ = _loc3_ + 1;
      }
      this.removeMovieClip();
   }
};
