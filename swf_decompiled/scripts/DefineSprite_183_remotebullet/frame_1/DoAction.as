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
   if(owner.alive)
   {
      if(owner.mouseTank)
      {
         var _loc5_ = _root.game.mazemc._xmouse - _X;
         var _loc4_ = _root.game.mazemc._ymouse - _Y;
         var _loc6_ = Math.sqrt(Math.pow(_loc5_,2) + Math.pow(_loc4_,2));
         if(_loc5_ < 0)
         {
            if(_loc4_ < 0)
            {
               aimAngle = -3.141592653589793 + Math.atan(_loc4_ / _loc5_);
            }
            else
            {
               aimAngle = 3.141592653589793 + Math.atan(_loc4_ / _loc5_);
            }
         }
         else if(_loc5_ > 0)
         {
            aimAngle = Math.atan(_loc4_ / _loc5_);
         }
         else if(_loc4_ < 0)
         {
            aimAngle = -1.5707963267948966;
         }
         else
         {
            aimAngle = 1.5707963267948966;
         }
         _rotation = (aimAngle + 1.5707963267948966) * 180 / 3.141592653589793;
      }
      else
      {
         if(Key.isDown(owner.KEYTURNLEFT))
         {
            turnLeft = true;
         }
         else
         {
            turnLeft = false;
         }
         if(Key.isDown(owner.KEYTURNRIGHT))
         {
            turnRight = true;
         }
         else
         {
            turnRight = false;
         }
         turnSize = 0;
         if(turnLeft)
         {
            turnSize = - turnSpeed;
         }
         if(turnRight)
         {
            turnSize += turnSpeed;
         }
         _rotation = _rotation + turnSize;
      }
   }
   xSpeed = Math.cos((_rotation - 90) * 3.141592653589793 / 180) * _root.REMOTESPEED / _root.REMOTEHITCHECKINTERVALS * (_root.SCALE / 50);
   ySpeed = Math.sin((_rotation - 90) * 3.141592653589793 / 180) * _root.REMOTESPEED / _root.REMOTEHITCHECKINTERVALS * (_root.SCALE / 50);
   i = 0;
   while(i < _root.REMOTEHITCHECKINTERVALS)
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
            _rotation = 180 - _rotation;
         }
         else if(hitOnYInvert && !hitOnXInvert)
         {
            xSpeed = - xSpeed;
            _rotation = - _rotation;
         }
         else
         {
            xSpeed = - xSpeed;
            ySpeed = - ySpeed;
            _rotation = _rotation + 180;
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
            if(owner == _root.game["tank" + i])
            {
               _root.game["tank" + i].scoreboard.subtract(_root.TANKS - 1);
               _root.loadVariables("includes/updateGameStatistics.php?tankScrapped=" + _root.TANKS,"POST");
               var _loc3_ = 0;
               while(_loc3_ < _root.TANKS)
               {
                  if(_loc3_ != i)
                  {
                     _root.game["tank" + _loc3_].scoreboard.add(1);
                  }
                  _loc3_ = _loc3_ + 1;
               }
            }
            else
            {
               _root.game["tank" + i].scoreboard.subtract(1);
               _root.loadVariables("includes/updateGameStatistics.php?tankScrapped=" + _root.TANKS,"POST");
               owner.scoreboard.add(1);
            }
            owner.remoteControlling = false;
            _root.setWeapon(owner,"bullet");
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
      owner.remoteControlling = false;
      _root.setWeapon(owner,"bullet");
      if(_root.soundOn)
      {
         _root.soundPoof.start();
      }
      _loc3_ = 0;
      while(_loc3_ < _root.NUMBEROFSMOKECLOUDS * 2)
      {
         s = _root.game.createEmptyMovieClip("smokeremotebullet" + _root.game.getNextHighestDepth(),_root.game.getNextHighestDepth());
         s.lineStyle(5 * (_root.SCALE / 50),Math.round(random(4)) * 1118481,10 + random(20));
         s.moveTo(0,0);
         s.lineTo(0,1);
         s.xspeed = xSpeed * _root.REMOTEHITCHECKINTERVALS + 0.5 * (Math.random() * 8 - 4) * (_root.SCALE / 50);
         s.yspeed = ySpeed * _root.REMOTEHITCHECKINTERVALS + 0.5 * (Math.random() * 8 - 4) * (_root.SCALE / 50);
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
