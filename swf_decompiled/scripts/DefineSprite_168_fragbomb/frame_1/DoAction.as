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
   if(_root.soundOn)
   {
      _root.soundExplosion3.start();
      _root.soundExplosion3.start();
      _root.soundExplosion3.start();
   }
   _root.shake += 7;
   owner.fragFired = false;
   owner.lastFrag = undefined;
   _root.setWeapon(owner,"bullet");
   if(level > 0)
   {
      var _loc5_ = 0;
      while(_loc5_ < _root.FRAGFRAGMENTS)
      {
         fragDepth = _root.game.getNextHighestDepth();
         fragName = "fragfragment" + fragDepth;
         frag = _root.game.attachMovie("fragbombfragment",fragName,fragDepth);
         this.swapDepths(frag);
         frag.x = _X;
         frag.y = _Y;
         frag._x = frag.x;
         frag._y = frag.y;
         frag._xscale = 170 * (_root.SCALE / 50);
         frag._yscale = 170 * (_root.SCALE / 50);
         frag._rotation = random(360);
         var _loc4_ = 25 + random(25);
         frag.rotSpeed = Math.random() <= 0.5 ? - _loc4_ : _loc4_;
         var _loc6_ = random(360);
         frag.xSpeed = Math.cos((_loc6_ - 90) * 3.141592653589793 / 180) * (_root.FRAGSPEED - Math.random() * _root.FRAGSPEED + 4) / _root.FRAGHITCHECKINTERVALS * (_root.SCALE / 50);
         frag.ySpeed = Math.sin((_loc6_ - 90) * 3.141592653589793 / 180) * (_root.FRAGSPEED - Math.random() * _root.FRAGSPEED + 4) / _root.FRAGHITCHECKINTERVALS * (_root.SCALE / 50);
         frag.active = true;
         frag.owner = owner;
         _loc5_ = _loc5_ + 1;
      }
   }
   var _loc3_ = 0;
   while(_loc3_ < _root.FRAGSMOKECLOUDS)
   {
      s = _root.game.createEmptyMovieClip("smokefrag" + owner + "-" + _loc3_,_root.game.getNextHighestDepth());
      s.lineStyle(15 * (_root.SCALE / 50) * ((level + 2) / (_root.FRAGLEVELS + 2)),Math.round(random(4)) * 1118481,40 + random(20));
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
      _loc3_ = _loc3_ + 1;
   }
   this.removeMovieClip();
}
onEnterFrame = function()
{
   if(_root.frozen)
   {
      return undefined;
   }
   if(!owner.alive)
   {
      detonate();
   }
   i = 0;
   while(i < _root.FRAGHITCHECKINTERVALS)
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
            detonate();
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
      detonate();
      this.removeMovieClip();
   }
};
