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
   while(i < _root.BULLETHITCHECKINTERVALS)
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
   if(deadly == 0 && this == firstBullet && lastBullet instanceof MovieClip)
   {
      sparkMC.clear();
      var _loc8_ = lastBullet.x - this.x;
      var _loc7_ = lastBullet.y - this.y;
      var _loc6_ = Math.sqrt(_loc8_ * _loc8_ + _loc7_ * _loc7_);
      sparkMC.points = new Array();
      var _loc10_ = (- _loc7_) / _loc6_;
      var _loc9_ = _loc8_ / _loc6_;
      sparkMC.waveCounter += 0.5;
      _loc8_ = _loc8_ / _loc6_ * 5;
      _loc7_ = _loc7_ / _loc6_ * 5;
      var _loc3_ = 0;
      while(_loc3_ <= _loc6_ / 5)
      {
         var _loc5_ = Math.cos(_loc3_ / 5 + sparkMC.waveCounter) * Math.min(1,Math.min(_loc3_ / 5,(_loc6_ / 5 - _loc3_) / 5)) * 10;
         sparkMC.points[_loc3_] = {x:this.x + _loc3_ * _loc8_ + _loc10_ * _loc5_,y:this.y + _loc3_ * _loc7_ + _loc9_ * _loc5_};
         _loc3_ = _loc3_ + 1;
      }
      _loc3_ = 0;
      while(_loc3_ < 2)
      {
         sparkMC.moveTo(sparkMC.points[0].x,sparkMC.points[0].y);
         var i = 1;
         while(i < sparkMC.points.length - 1)
         {
            sparkMC.lineStyle(Math.random() * 2 + 1,Math.random() >= 0.5 ? 10066431 : 6711039,Math.random() * 30 + 70);
            sparkMC.lineTo(sparkMC.points[i].x + Math.random() * 10 - 5,sparkMC.points[i].y + Math.random() * 10 - 5);
            i++;
         }
         if(sparkMC.points.length >= 2)
         {
            sparkMC.lineTo(sparkMC.points[i].x,sparkMC.points[i].y);
         }
         _loc3_ = _loc3_ + 1;
      }
      if(!sparkMC.shooting && Math.random() > 0.85)
      {
         sparkMC.shooting = true;
         var i = Math.floor(Math.random() * sparkMC.points.length);
         sparkMC.shootingPoints = new Array();
         sparkMC.shootingPoints[sparkMC.shootingPoints.length] = {x:sparkMC.points[i].x,y:sparkMC.points[i].y};
      }
      if(sparkMC.shooting)
      {
         sparkMC.moveTo(sparkMC.shootingPoints[0].x,sparkMC.shootingPoints[0].y);
         sparkMC.shootingPoints[sparkMC.shootingPoints.length] = {x:sparkMC.shootingPoints[sparkMC.shootingPoints.length - 1].x + (Math.random() >= 0.5 ? -1 : 1) * (Math.random() * 10 + 5),y:sparkMC.shootingPoints[sparkMC.shootingPoints.length - 1].y + (Math.random() >= 0.5 ? -1 : 1) * (Math.random() * 10 + 5)};
         var i = 0;
         while(i < sparkMC.shootingPoints.length)
         {
            sparkMC.lineStyle((sparkMC.shootingPoints.length - i) / 2,Math.random() >= 0.5 ? 10066431 : 6711039,100);
            sparkMC.lineTo(sparkMC.shootingPoints[i].x + Math.random() * 5 - 2.5,sparkMC.shootingPoints[i].y + Math.random() * 5 - 2.5);
            i++;
         }
         if(sparkMC.shootingPoints.length == 6)
         {
            sparkMC.shooting = false;
         }
      }
   }
   if(deadly > 0)
   {
      deadly--;
   }
   lifetime--;
   if(lifetime <= 0)
   {
      owner.electricReady = true;
      _root.setWeapon(owner,"bullet");
      if(_root.soundOn)
      {
         _root.soundPoof.start();
      }
      var _loc4_ = 0;
      while(_loc4_ < _root.NUMBEROFSMOKECLOUDS * 2)
      {
         s = _root.game.createEmptyMovieClip("smokebullet" + _root.game.getNextHighestDepth(),_root.game.getNextHighestDepth());
         s.lineStyle(5 * (_root.SCALE / 50),Math.round(random(4)) * 1118481,10 + random(20));
         s.moveTo(0,0);
         s.lineTo(0,1);
         s.xspeed = xSpeed * _root.BULLETHITCHECKINTERVALS + 0.5 * (Math.random() * 8 - 4) * (_root.SCALE / 50);
         s.yspeed = ySpeed * _root.BULLETHITCHECKINTERVALS + 0.5 * (Math.random() * 8 - 4) * (_root.SCALE / 50);
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
         _loc4_ = _loc4_ + 1;
      }
      sparkMC.removeMovieClip();
      this.removeMovieClip();
   }
};
