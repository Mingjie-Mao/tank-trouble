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
   if(!owner.alive)
   {
      this.removeMovieClip();
   }
   clear();
   lineStyle(1 * (_root.SCALE / 50),aimerColor,100);
   moveTo(0,0);
   _X = owner._x + Math.cos((owner._rotation - 90) * 3.141592653589793 / 180) * _root.SCALE * 4.5 / 16;
   _Y = owner._y + Math.sin((owner._rotation - 90) * 3.141592653589793 / 180) * _root.SCALE * 4.5 / 16;
   x = 0;
   y = 0;
   active = _root.AIMERACTIVE;
   xSpeed = Math.cos((owner._rotation - 90) * 3.141592653589793 / 180) * _root.AIMERLENGTH / _root.AIMERHITCHECKINTERVALS * (_root.SCALE / 50);
   ySpeed = Math.sin((owner._rotation - 90) * 3.141592653589793 / 180) * _root.AIMERLENGTH / _root.AIMERHITCHECKINTERVALS * (_root.SCALE / 50);
   hit = undefined;
   hitXSpeed = 0;
   hitYSpeed = 0;
   j = 0;
   while(j < _root.AIMERHITCHECKINTERVALS)
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
      if(active > 0)
      {
         active--;
      }
      if(active == 0 && j % 2 == 0)
      {
         var _loc3_ = 0;
         while(_loc3_ < _root.TANKS)
         {
            if(_root.game["tank" + _loc3_].alive && hitCheck(_root.game["tank" + _loc3_],{x:x,y:y}))
            {
               hit = _root.game["tank" + _loc3_];
               hitXSpeed = xSpeed;
               hitYSpeed = ySpeed;
               j = _root.AIMERHITCHECKINTERVALS;
            }
            _loc3_ = _loc3_ + 1;
         }
      }
      if(Math.random() > 0.7000000000000001)
      {
         lineStyle(3 * (_root.SCALE / 50),0,30);
         lineTo(x,y);
         lineStyle(2 * (_root.SCALE / 50),aimerColor,100);
         moveTo(previousX,previousY);
         lineTo(x,y);
      }
      else
      {
         moveTo(x,y);
      }
      j++;
   }
};
