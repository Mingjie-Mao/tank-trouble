function hitCheck(mc, point)
{
   localToGlobal(point);
   if(mc.hitTest(point.x,point.y,true))
   {
      return true;
   }
   return false;
}
var linePoints = new Array();
onEnterFrame = function()
{
   if(_root.frozen)
   {
      return undefined;
   }
   if(active)
   {
      j = 0;
      while(j < _root.LASERHITCHECKINTERVALS)
      {
         previousX = x;
         previousY = y;
         x += xSpeed;
         y += ySpeed;
         if(hitCheck(_root.game.mazemc,{x:x,y:y}))
         {
            x = previousX;
            y = previousY;
            x -= xSpeed;
            y += ySpeed;
            if(hitCheck(_root.game.mazemc,{x:x,y:y}))
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
            if(hitCheck(_root.game.mazemc,{x:x,y:y}))
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
         if(deadly > 0)
         {
            deadly--;
         }
         if(deadly == 0)
         {
            var _loc3_ = 0;
            while(_loc3_ < _root.TANKS)
            {
               if(_root.game["tank" + _loc3_].alive && hitCheck(_root.game["tank" + _loc3_],{x:x,y:y}))
               {
                  _root.registerHit(owner,_root.game["tank" + _loc3_]);
                  _root.destroyTank(_loc3_);
                  j = _root.LASERHITCHECKINTERVALS;
                  lifetime = 1;
               }
               _loc3_ = _loc3_ + 1;
            }
         }
         linePoints.push({x:x,y:y});
         j++;
      }
   }
   else
   {
      j = 0;
      while(j < _root.LASERHITCHECKINTERVALS)
      {
         linePoints.shift();
         j++;
      }
      if(linePoints.length == 0)
      {
         this.removeMovieClip();
      }
   }
   clear();
   lineStyle(3 * (_root.SCALE / 50),0,30);
   moveTo(linePoints[0].x,linePoints[0].y);
   _loc3_ = 1;
   while(_loc3_ < linePoints.length)
   {
      lineTo(linePoints[_loc3_].x,linePoints[_loc3_].y);
      _loc3_ = _loc3_ + 1;
   }
   lineStyle(2 * (_root.SCALE / 50),laserColor,100);
   moveTo(linePoints[0].x,linePoints[0].y);
   _loc3_ = 1;
   while(_loc3_ < linePoints.length)
   {
      lineTo(linePoints[_loc3_].x,linePoints[_loc3_].y);
      _loc3_ = _loc3_ + 1;
   }
   lifetime--;
   if(lifetime == 0)
   {
      owner.laserReady = true;
      active = false;
      _root.setWeapon(owner,"bullet");
   }
};
