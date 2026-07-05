function hitCheck(mc, point)
{
   localToGlobal(point);
   if(mc.hitTest(point.x,point.y,true))
   {
      return true;
   }
   return false;
}
function detonate()
{
   _root.shake += 7;
   if(_root.soundOn)
   {
      _root.soundExplosion3.start();
      _root.soundExplosion3.start();
      _root.soundExplosion3.start();
   }
   var _loc5_ = 0;
   while(_loc5_ < _root.MINEFRAGMENTS)
   {
      fragDepth = _root.game.getNextHighestDepth();
      fragName = "fragfragment" + fragDepth;
      frag = _root.game.attachMovie("fragbombfragment",fragName,fragDepth);
      frag.x = _X;
      frag.y = _Y;
      frag._x = frag.x;
      frag._y = frag.y;
      frag._xscale = 170 * (_root.SCALE / 50);
      frag._yscale = 170 * (_root.SCALE / 50);
      frag._rotation = random(360);
      var _loc3_ = 25 + random(25);
      frag.rotSpeed = Math.random() <= 0.5 ? - _loc3_ : _loc3_;
      var _loc6_ = random(360);
      frag.xSpeed = Math.cos((_loc6_ - 90) * 3.141592653589793 / 180) * (_root.FRAGSPEED - Math.random() * _root.FRAGSPEED + 4) / _root.FRAGHITCHECKINTERVALS * (_root.SCALE / 50);
      frag.ySpeed = Math.sin((_loc6_ - 90) * 3.141592653589793 / 180) * (_root.FRAGSPEED - Math.random() * _root.FRAGSPEED + 4) / _root.FRAGHITCHECKINTERVALS * (_root.SCALE / 50);
      frag.active = true;
      frag.owner = owner;
      _loc5_ = _loc5_ + 1;
   }
   var _loc4_ = 0;
   while(_loc4_ < _root.MINESMOKECLOUDS)
   {
      s = _root.game.createEmptyMovieClip("smokemine" + owner + "-" + _root.game.getNextHighestDepth(),_root.game.getNextHighestDepth());
      s.lineStyle(15 * (_root.SCALE / 50),Math.round(random(4)) * 1118481,40 + random(20));
      s.moveTo(0,0);
      s.lineTo(0,1);
      s.xspeed = (Math.random() * 2 - 1) * (_root.SCALE / 50);
      s.yspeed = (Math.random() * 2 - 1) * (_root.SCALE / 50);
      s.x = _X + s.xspeed * (random(6) + 1) + (random(2) - 1) * (_root.SCALE / 50);
      s.y = _Y + s.yspeed * (random(6) + 1) + (random(2) - 1) * (_root.SCALE / 50);
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
         this._alpha -= 3 - Math.random() * 2;
         this.xspeed *= 0.93;
         this.yspeed *= 0.93;
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
   this.removeMovieClip();
}
stop();
onEnterFrame = function()
{
   if(_root.frozen)
   {
      blinker.stop();
      return undefined;
   }
   i = 0;
   while(i < _root.MINEHITCHECKINTERVALS)
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
   xSpeed *= 0.75;
   ySpeed *= 0.75;
   _X = x;
   _Y = y;
   if(!landed)
   {
      if(Math.abs(xSpeed) < 0.15 && Math.abs(ySpeed) < 0.15)
      {
         landed = true;
         xSpeed = 0;
         ySpeed = 0;
         if(_root.soundOn)
         {
            _root.soundMineLand.start();
         }
         var _loc4_ = 0;
         while(_loc4_ < _root.NUMBEROFDUSTCLOUDS / 2)
         {
            s = _root.game.mazebg.createEmptyMovieClip("minedust-" + _root.game.mazebg.getNextHighestDepth(),_root.game.mazebg.getNextHighestDepth());
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
   }
   if(deadly == 0 && !armed)
   {
      var i = 0;
      while(i < _root.TANKS)
      {
         var _loc3_ = 0;
         while(_loc3_ < 6.283185307179586)
         {
            if(_root.game["tank" + i].alive && hitCheck(_root.game["tank" + i].base,{x:Math.cos(_loc3_) * 5,y:Math.sin(_loc3_) * 5}))
            {
               armed = true;
               _root.soundMineActivate.stop("soundMineActivate");
               if(_root.soundOn)
               {
                  _root.soundMineArm.start();
               }
               break;
            }
            _loc3_ += 0.5;
         }
         if(armed)
         {
            break;
         }
         i++;
      }
   }
   if(armed && !detonating)
   {
      var _loc5_ = false;
      var i = 0;
      while(i < _root.TANKS)
      {
         _loc3_ = 0;
         while(_loc3_ < 6.283185307179586)
         {
            if(_root.game["tank" + i].alive && hitCheck(_root.game["tank" + i].base,{x:Math.cos(_loc3_) * 5,y:Math.sin(_loc3_) * 5}))
            {
               _loc5_ = true;
               break;
            }
            _loc3_ += 0.5;
         }
         if(_loc5_)
         {
            break;
         }
         i++;
      }
      if(!_loc5_)
      {
         detonating = true;
         if(_root.soundOn)
         {
            _root.soundMineDetonate.start();
         }
      }
   }
   if(detonating)
   {
      if(detonateCounter > 0)
      {
         detonateCounter--;
      }
      if(detonateCounter == 0)
      {
         _root.soundMineDetonate.stop("soundMineDetonateCharge");
         detonate();
      }
   }
   if(deadly > 0)
   {
      deadly--;
   }
   if(deadly == 1)
   {
      gotoAndStop(2);
      if(_root.soundOn)
      {
         _root.soundMineActivate.start();
      }
   }
   if(deadly == 0 && !armed)
   {
      if(hideCounter > 0)
      {
         hideCounter--;
      }
      if(hideCounter == 0)
      {
         if(_alpha > 0)
         {
            _alpha = _alpha - 10;
         }
      }
   }
   else if(armed)
   {
      if(_alpha < 100)
      {
         _alpha = _alpha + 10;
      }
   }
};
