function hitCheck(mc, point)
{
   localToGlobal(point);
   if(mc.hitTest(point.x,point.y,true))
   {
      return true;
   }
   return false;
}
stop();
onEnterFrame = function()
{
   if(_root.frozen)
   {
      return undefined;
   }
   if(homing)
   {
      var _loc11_ = Math.floor(x / _root.SCALE);
      var _loc12_ = Math.floor(y / _root.SCALE);
      var _loc10_ = 1000;
      var _loc14_ = undefined;
      var _loc13_ = undefined;
      var _loc3_ = 0;
      while(_loc3_ < _root.TANKS)
      {
         var _loc9_ = Math.floor(_root.game["tank" + _loc3_].x / _root.SCALE);
         var _loc7_ = Math.floor(_root.game["tank" + _loc3_].y / _root.SCALE);
         if(_root.game["tank" + _loc3_].alive)
         {
            var _loc5_ = _root.distancesForMaze[_loc11_][_loc12_][_loc9_][_loc7_];
            if(_loc5_ < _loc10_ || _loc5_ == _loc10_ && _root.game["tank" + _loc3_] != owner)
            {
               _loc10_ = _loc5_;
               _loc14_ = _loc9_;
               _loc13_ = _loc7_;
               target = _root.game["tank" + _loc3_];
               targetColor = 16711680 * target.baseColor.r / 255 + 65280 * target.baseColor.g / 255 + 255 * target.baseColor.b / 255;
            }
         }
         _loc3_ = _loc3_ + 1;
      }
      if(!target.alive)
      {
         homing = false;
      }
      else
      {
         if(target != oldTarget)
         {
            if(_root.soundOn)
            {
               _root.soundHoming.start();
            }
         }
         oldTarget = target;
         var _loc17_ = _root.getShortestPathWithDistances(_root.maze,_root.distancesForMaze[_loc11_][_loc12_],_loc11_,_loc12_,_loc14_,_loc13_);
         var _loc16_ = undefined;
         var _loc15_ = undefined;
         if(_root.distancesForMaze[_loc11_][_loc12_][_loc14_][_loc13_] + 1 <= 1)
         {
            _loc16_ = target.x;
            _loc15_ = target.y;
         }
         else
         {
            _loc16_ = (_loc17_[0].x + 0.5) * _root.SCALE;
            _loc15_ = (_loc17_[0].y + 0.5) * _root.SCALE;
         }
         soundCounter++;
         if(_root.soundOn)
         {
            if(soundCounter > (_root.distancesForMaze[_loc11_][_loc12_][_loc14_][_loc13_] + 1) * 4)
            {
               soundCounter = 0;
               _root.soundHoming2.start();
            }
         }
         _loc3_ = 0;
         while(_loc3_ < _root.HOMINGHITCHECKINTERVALS)
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
            if(_loc16_ - x < 0)
            {
               xSpeed -= 0.12 / _root.HOMINGHITCHECKINTERVALS;
            }
            else
            {
               xSpeed += 0.12 / _root.HOMINGHITCHECKINTERVALS;
            }
            if(_loc15_ - y < 0)
            {
               ySpeed -= 0.12 / _root.HOMINGHITCHECKINTERVALS;
            }
            else
            {
               ySpeed += 0.12 / _root.HOMINGHITCHECKINTERVALS;
            }
            var _loc4_ = Math.sqrt(xSpeed * xSpeed + ySpeed * ySpeed);
            if(_loc4_ > _root.HOMINGSPEED / _root.HOMINGHITCHECKINTERVALS * (_root.SCALE / 50))
            {
               xSpeed = xSpeed / _loc4_ * _root.HOMINGSPEED / _root.HOMINGHITCHECKINTERVALS * (_root.SCALE / 50);
               ySpeed = ySpeed / _loc4_ * _root.HOMINGSPEED / _root.HOMINGHITCHECKINTERVALS * (_root.SCALE / 50);
            }
            _loc3_ = _loc3_ + 1;
         }
      }
   }
   else
   {
      _loc3_ = 0;
      while(_loc3_ < _root.HOMINGHITCHECKINTERVALS)
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
         _loc3_ = _loc3_ + 1;
      }
   }
   _X = x;
   _Y = y;
   if(xSpeed < 0)
   {
      if(ySpeed < 0)
      {
         aimAngle = -3.141592653589793 + Math.atan(ySpeed / xSpeed);
      }
      else
      {
         aimAngle = 3.141592653589793 + Math.atan(ySpeed / xSpeed);
      }
   }
   else if(xSpeed > 0)
   {
      aimAngle = Math.atan(ySpeed / xSpeed);
   }
   else if(ySpeed < 0)
   {
      aimAngle = -1.5707963267948966;
   }
   else
   {
      aimAngle = 1.5707963267948966;
   }
   _rotation = (aimAngle + 1.5707963267948966) * 180 / 3.141592653589793;
   _loc3_ = 0;
   while(_loc3_ < _root.HOMINGSMOKECLOUDS)
   {
      var _loc6_ = _root.game.getNextHighestDepth();
      _root.game.createEmptyMovieClip("homingSmoke-" + _loc6_,_root.game.getNextHighestDepth());
      s = _root.game["homingSmoke-" + _loc6_];
      if(homing && Math.random() > 0.5)
      {
         s.lineStyle(4 * (_root.SCALE / 50),targetColor,20);
      }
      else
      {
         s.lineStyle(4 * (_root.SCALE / 50),Math.round(random(4) + 6) * 1118481,20);
      }
      s.moveTo(0,0);
      s.lineTo(0,1);
      s.xspeed = - this.xSpeed + (Math.random() - 0.5) * (_root.SCALE / 50);
      s.yspeed = - this.ySpeed + (Math.random() - 0.5) * (_root.SCALE / 50);
      _loc4_ = Math.sqrt(xSpeed * xSpeed + ySpeed * ySpeed);
      if(isNaN(_loc4_))
      {
         _loc4_ = 1;
      }
      s.x = this._x + (-8 * this.xSpeed / _loc4_ + (5 * Math.random() - 2.5)) * (_root.SCALE / 50);
      s.y = this._y + (-8 * this.ySpeed / _loc4_ + (5 * Math.random() - 2.5)) * (_root.SCALE / 50);
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
         this._alpha -= 4 - Math.random() * 4;
         this.xspeed *= 0.9500000000000001;
         this.yspeed *= 0.9500000000000001;
         this.x += this.xspeed;
         this.y += this.yspeed;
         this._x = this.x;
         this._y = this.y;
         if(this._alpha <= 0)
         {
            this.removeMovieClip();
         }
      };
      _loc3_ = _loc3_ + 1;
   }
   if(deadly == 0)
   {
      _loc3_ = 0;
      while(_loc3_ < _root.TANKS)
      {
         if(_root.game["tank" + _loc3_].alive && hitCheck(_root.game["tank" + _loc3_],{x:0,y:0}))
         {
            _root.registerHit(owner,_root.game["tank" + _loc3_]);
            owner.homingReady = true;
            _root.setWeapon(owner,"bullet");
            _root.destroyTank(_loc3_);
            this.removeMovieClip();
         }
         _loc3_ = _loc3_ + 1;
      }
   }
   if(startuptime >= 0)
   {
      startuptime--;
   }
   if(startuptime == 0)
   {
      homing = true;
   }
   if(deadly > 0)
   {
      deadly--;
   }
   lifetime--;
   if(lifetime <= 0)
   {
      owner.homingReady = true;
      _root.setWeapon(owner,"bullet");
      if(_root.soundOn)
      {
         _root.soundPoof.start();
      }
      var _loc8_ = 0;
      while(_loc8_ < _root.NUMBEROFSMOKECLOUDS * 2)
      {
         s = _root.game.createEmptyMovieClip("smokebullet" + _root.game.getNextHighestDepth(),_root.game.getNextHighestDepth());
         s.lineStyle(5 * (_root.SCALE / 50),Math.round(random(4)) * 1118481,10 + random(20));
         s.moveTo(0,0);
         s.lineTo(0,1);
         s.xspeed = xSpeed * _root.HOMINGHITCHECKINTERVALS + 0.5 * (Math.random() * 8 - 4) * (_root.SCALE / 50);
         s.yspeed = ySpeed * _root.HOMINGHITCHECKINTERVALS + 0.5 * (Math.random() * 8 - 4) * (_root.SCALE / 50);
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
         _loc8_ = _loc8_ + 1;
      }
      this.removeMovieClip();
   }
};
